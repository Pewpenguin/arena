use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::bootstrap;
use crate::compare::compare_all;
use crate::error::{Error, Result};
use crate::evaluate::evaluate_result;
use crate::execute::{ExecutionResult, execute_models};
use crate::judge::{Judgment, JudgmentFailure, judge_pairs};
use crate::persist::{self, Output, RunMetadata};
use crate::provider::{ModelId, ModelProvider};
use crate::stats;
use crate::task::Task;

pub struct ExecConfig {
    pub tasks: Vec<Task>,
    pub models: Vec<ModelId>,
    pub judge: Option<ModelId>,
    pub seed: u64,
    pub tasks_path: Option<PathBuf>,
    pub started_at: String,
    pub base_url: String,
}

pub fn unique_models(ids: Vec<String>) -> Result<Vec<ModelId>> {
    let models: Vec<_> = ids.into_iter().map(ModelId::new).collect();
    let mut seen = HashSet::new();
    for model in &models {
        if !seen.insert(model) {
            return Err(Error::DuplicateModel(model.clone()));
        }
    }
    Ok(models)
}

pub async fn collect_exec<P>(
    provider: &P,
    config: &ExecConfig,
    mut on_candidate: impl FnMut(&ExecutionResult),
    mut on_judgment: impl FnMut(&Judgment),
    mut on_failure: impl FnMut(&JudgmentFailure),
) -> Result<(Output, usize)>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let mut results = Vec::new();
    for task in &config.tasks {
        let executed = execute_models(provider, task, &config.models, &mut on_candidate).await?;
        for result in executed {
            results.push(evaluate_result(task, result));
        }
    }

    let comparisons = compare_all(&results);

    let mut judgments = Vec::new();
    let mut judgment_failures = Vec::new();
    if let Some(judge_model) = &config.judge {
        for task in &config.tasks {
            let task_results: Vec<_> = results
                .iter()
                .filter(|result| result.task_id == task.id)
                .cloned()
                .collect();
            let outcome = judge_pairs(
                provider,
                judge_model.clone(),
                task,
                &task_results,
                &mut on_judgment,
            )
            .await?;
            for failure in &outcome.failures {
                on_failure(failure);
            }
            judgments.extend(outcome.judgments);
            judgment_failures.extend(outcome.failures);
        }
    }

    let (statistics, ratings, bootstrap_meta) = if config.judge.is_some() {
        let statistics = stats::aggregate(&judgments, &config.models);
        let (ratings, meta) =
            bootstrap::rate_with_uncertainty(&judgments, &config.models, config.seed);
        (statistics, ratings, Some(meta))
    } else {
        (Vec::new(), Vec::new(), None)
    };

    let mut run = RunMetadata::new(
        config.models.clone(),
        config.judge.clone(),
        config.tasks_path.clone(),
        config.started_at.clone(),
        config.base_url.clone(),
    );
    if let Some(meta) = bootstrap_meta {
        run = run.with_bootstrap(meta.seed, meta.replicates, meta.valid);
    }

    let failed_pairs = judgment_failures.len();
    Ok((
        Output {
            run,
            tasks: config.tasks.clone(),
            results,
            comparisons,
            judgments,
            judgment_failures,
            statistics,
            ratings,
        },
        failed_pairs,
    ))
}

pub fn emit_output(output: &Output, path: Option<&Path>) -> Result<()> {
    match path {
        Some(path) => persist::write(path, output)?,
        None => println!("{}", persist::to_pretty_json(output)?),
    }
    Ok(())
}

pub fn complete_exec(output: Output, failed_pairs: usize, path: Option<&Path>) -> Result<()> {
    emit_output(&output, path)?;
    if failed_pairs > 0 {
        return Err(Error::IncompleteJudgments(failed_pairs));
    }
    Ok(())
}

pub async fn run_exec<P>(
    provider: &P,
    config: &ExecConfig,
    output_path: Option<&Path>,
) -> Result<()>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let (output, failed_pairs) = collect_exec(provider, config, |_| {}, |_| {}, |_| {}).await?;
    complete_exec(output, failed_pairs, output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::provider::{CompletionRequest, CompletionResponse, ProviderError};

    fn sample_task() -> Task {
        Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        }
    }

    fn config(models: &[&str], judge: Option<&str>) -> ExecConfig {
        ExecConfig {
            tasks: vec![sample_task()],
            models: models.iter().map(|id| ModelId::new(*id)).collect(),
            judge: judge.map(ModelId::new),
            seed: 0,
            tasks_path: Some(PathBuf::from("tasks.json")),
            started_at: "2026-01-02T03:04:05Z".into(),
            base_url: "https://example.test/v1".into(),
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("arena-exec-{name}-{}.json", std::process::id()))
    }

    #[derive(Clone)]
    struct OkProvider;

    impl ModelProvider for OkProvider {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            if request.model == ModelId::new("judge") {
                return Ok(CompletionResponse {
                    text: r#"{"winner":"a","reason":"ok"}"#.into(),
                });
            }
            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[derive(Clone)]
    struct FailCandidates {
        calls: Arc<AtomicUsize>,
    }

    impl ModelProvider for FailCandidates {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(ProviderError::RequestFailed("upstream down".into()))
        }
    }

    #[derive(Clone)]
    struct FailJudge;

    impl ModelProvider for FailJudge {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            if request.model == ModelId::new("judge") {
                return Ok(CompletionResponse {
                    text: "not a judgment".into(),
                });
            }
            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[test]
    fn duplicate_model_ids_are_rejected() {
        let error = unique_models(vec!["a".into(), "b".into(), "a".into()]).unwrap_err();
        match error {
            Error::DuplicateModel(id) => assert_eq!(id, ModelId::new("a")),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn unique_model_ids_are_kept_in_order() {
        assert_eq!(
            unique_models(vec!["b".into(), "a".into()]).unwrap(),
            vec![ModelId::new("b"), ModelId::new("a")]
        );
    }

    #[tokio::test]
    async fn candidate_failure_does_not_write_output() {
        let path = temp_path("candidate-fail");
        let _ = std::fs::remove_file(&path);
        let provider = FailCandidates {
            calls: Arc::new(AtomicUsize::new(0)),
        };

        let error = run_exec(&provider, &config(&["m0"], None), Some(&path))
            .await
            .unwrap_err();

        assert!(matches!(error, Error::Execute(_)), "{error}");
        assert!(!path.exists(), "failed candidate run must not write output");
        assert!(provider.calls.load(Ordering::SeqCst) > 0);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn incomplete_judgments_are_written_before_error() {
        let path = temp_path("incomplete-judge");
        let _ = std::fs::remove_file(&path);

        let error = run_exec(
            &FailJudge,
            &config(&["m0", "m1"], Some("judge")),
            Some(&path),
        )
        .await
        .unwrap_err();

        match error {
            Error::IncompleteJudgments(1) => {}
            other => panic!("unexpected error: {other}"),
        }

        let contents = std::fs::read_to_string(&path).expect("output written before error");
        let value: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(value["judgment_failures"].as_array().unwrap().len(), 1);
        assert!(value.get("judgments").is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn output_file_matches_to_pretty_json() {
        let path = temp_path("success");
        let _ = std::fs::remove_file(&path);
        let cfg = config(&["m0", "m1"], Some("judge"));

        let (output, failed_pairs) = collect_exec(&OkProvider, &cfg, |_| {}, |_| {}, |_| {})
            .await
            .unwrap();
        assert_eq!(failed_pairs, 0);
        let json = persist::to_pretty_json(&output).unwrap();
        persist::write(&path, &output).unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, json);
        assert!(written.contains("\"m0\""));
        assert!(written.contains("\"bootstrap_seed\""));
        assert!(!written.contains("api_key"));
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn to_pretty_json_encodes_a_no_judge_run() {
        let cfg = config(&["m0"], None);
        let (output, failed_pairs) = collect_exec(&OkProvider, &cfg, |_| {}, |_| {}, |_| {})
            .await
            .unwrap();
        assert_eq!(failed_pairs, 0);
        let json = persist::to_pretty_json(&output).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["results"].as_array().unwrap().len(), 1);
        assert!(parsed.get("judgments").is_none());
        assert!(json.starts_with('{'));
    }
}
