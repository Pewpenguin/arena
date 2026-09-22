use std::collections::HashSet;
use std::path::{Path, PathBuf};

use tokio::sync::mpsc;

use crate::bootstrap;
use crate::compare::compare_all;
use crate::error::{Error, Result};
use crate::evaluate::evaluate_result;
use crate::event::ExperimentEvent;
use crate::execute::{ExecutionResult, execute_models};
use crate::judge::{Judgment, JudgmentFailure, judge_pairs};
use crate::persist::{self, Output, RunMetadata};
use crate::provider::{ModelId, ModelProvider};
use crate::stats;
use crate::task::Task;

#[derive(Debug)]
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

pub fn validate_judge(models: &[ModelId], judge: Option<&ModelId>) -> Result<()> {
    if let Some(judge) = judge
        && models.iter().any(|model| model == judge)
    {
        return Err(Error::JudgeIsCandidate(judge.clone()));
    }
    Ok(())
}

pub fn expected_pairs(task_count: usize, model_count: usize) -> usize {
    let pairs_per_task = model_count.saturating_sub(1).saturating_mul(model_count) / 2;
    task_count.saturating_mul(pairs_per_task)
}

pub async fn collect_exec<P>(
    provider: &P,
    config: &ExecConfig,
    on_candidate: impl FnMut(&ExecutionResult),
    on_judgment: impl FnMut(&Judgment),
    on_failure: impl FnMut(&JudgmentFailure),
) -> Result<(Output, usize)>
where
    P: ModelProvider + Clone + Send + 'static,
{
    collect_exec_inner(
        provider,
        config,
        on_candidate,
        on_judgment,
        on_failure,
        None,
    )
    .await
}

pub async fn collect_exec_with_events<P>(
    provider: &P,
    config: &ExecConfig,
    events: mpsc::UnboundedSender<ExperimentEvent>,
) -> Result<(Output, usize)>
where
    P: ModelProvider + Clone + Send + 'static,
{
    collect_exec_inner(provider, config, |_| {}, |_| {}, |_| {}, Some(events)).await
}

fn emit_event(events: &Option<mpsc::UnboundedSender<ExperimentEvent>>, event: ExperimentEvent) {
    if let Some(tx) = events {
        let _ = tx.send(event);
    }
}

async fn collect_exec_inner<P>(
    provider: &P,
    config: &ExecConfig,
    mut on_candidate: impl FnMut(&ExecutionResult),
    mut on_judgment: impl FnMut(&Judgment),
    mut on_failure: impl FnMut(&JudgmentFailure),
    events: Option<mpsc::UnboundedSender<ExperimentEvent>>,
) -> Result<(Output, usize)>
where
    P: ModelProvider + Clone + Send + 'static,
{
    validate_judge(&config.models, config.judge.as_ref())?;
    let mut results = Vec::new();
    for task in &config.tasks {
        let executed = execute_models(provider, task, &config.models, |result| {
            on_candidate(result);
            emit_event(
                &events,
                ExperimentEvent::CandidateFinished {
                    task_id: result.task_id.clone(),
                    model: result.model.clone(),
                    duration_ms: result.duration_ms,
                },
            );
        })
        .await?;
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
            for judgment in &outcome.judgments {
                emit_event(
                    &events,
                    ExperimentEvent::PairResolved {
                        task_id: judgment.task_id.clone(),
                        model_a: judgment.model_a.clone(),
                        model_b: judgment.model_b.clone(),
                        judgment: judgment.clone(),
                    },
                );
            }
            for failure in &outcome.failures {
                on_failure(failure);
                emit_event(
                    &events,
                    ExperimentEvent::PairFailed {
                        task_id: failure.task_id.clone(),
                        model_a: failure.model_a.clone(),
                        model_b: failure.model_b.clone(),
                        failure: failure.clone(),
                    },
                );
            }
            judgments.extend(outcome.judgments);
            judgment_failures.extend(outcome.failures);
        }
    }

    let mut run = RunMetadata::new(
        config.models.clone(),
        config.judge.clone(),
        config.tasks_path.clone(),
        config.started_at.clone(),
        config.base_url.clone(),
    );
    let (judgments, judgment_failures, statistics, ratings, expected, resolved, failed) =
        if config.judge.is_some() {
            let statistics = stats::aggregate(&judgments, &config.models);
            let (ratings, meta) =
                bootstrap::rate_with_uncertainty(&judgments, &config.models, config.seed);
            run = run.with_bootstrap(&meta);
            let expected = expected_pairs(config.tasks.len(), config.models.len());
            let resolved = judgments.len();
            let failed = judgment_failures.len();
            if expected != resolved + failed {
                return Err(Error::InconsistentPairCoverage {
                    expected,
                    resolved,
                    failed,
                });
            }
            run = run
                .with_judge_coverage(expected, resolved, failed)
                .with_orientation_agreement(stats::pair_agreement(&judgments))
                .with_judge_decoding(persist::JudgeDecoding::arena_default());
            (
                Some(judgments),
                Some(judgment_failures),
                Some(statistics),
                Some(ratings),
                expected,
                resolved,
                failed,
            )
        } else {
            (None, None, None, None, 0, 0, 0)
        };

    emit_event(
        &events,
        ExperimentEvent::RunComplete {
            expected_pairs: expected,
            resolved_pairs: resolved,
            failed_pairs: failed,
        },
    );

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
        failed,
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

    use crate::event::ExperimentEvent;
    use crate::judge::JudgeDecision;
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
    async fn judge_matching_a_candidate_is_rejected_before_execution() {
        let path = temp_path("judge-is-candidate");
        let _ = std::fs::remove_file(&path);
        let provider = FailCandidates {
            calls: Arc::new(AtomicUsize::new(0)),
        };

        let error = run_exec(
            &provider,
            &config(&["model-a", "model-b"], Some("model-a")),
            Some(&path),
        )
        .await
        .unwrap_err();

        match error {
            Error::JudgeIsCandidate(id) => assert_eq!(id, ModelId::new("model-a")),
            other => panic!("unexpected error: {other}"),
        }
        assert!(!path.exists(), "rejected judge must not write output");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        let _ = std::fs::remove_file(&path);
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
        assert_eq!(value["judgments"].as_array().unwrap().len(), 0);
        assert_eq!(value["run"]["complete"], false);
        assert_eq!(value["run"]["expected_pairs"], 1);
        assert_eq!(value["run"]["resolved_pairs"], 0);
        assert_eq!(value["run"]["failed_pairs"], 1);
        assert_eq!(
            value["run"]["expected_pairs"].as_u64().unwrap(),
            value["run"]["resolved_pairs"].as_u64().unwrap()
                + value["run"]["failed_pairs"].as_u64().unwrap()
        );
        assert_eq!(
            value["run"]["orientation_agreement"],
            serde_json::json!({
                "resolved_pairs": 0,
                "orientation_agreeing_pairs": 0,
                "orientation_disagreeing_pairs": 0,
                "agreement_rate": 0.0
            })
        );
        let ratings = value["ratings"].as_array().unwrap();
        assert_eq!(ratings.len(), 2);
        assert!(
            ratings
                .iter()
                .all(|rating| rating["rating"].is_null()
                    && rating["unavailable"] == "no_comparisons")
        );
        assert_eq!(value["statistics"].as_array().unwrap().len(), 2);
        assert_eq!(
            value["run"]["judge_decoding"],
            serde_json::json!({ "temperature": 0.0, "max_tokens": 4096 })
        );
        assert_eq!(value["run"]["candidate_max_tokens"], 4096);
        assert_eq!(value["run"]["bootstrap_ran"], false);
        assert_eq!(value["run"]["bootstrap_unavailable"], "original_unrated");
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
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["run"]["complete"], true);
        assert_eq!(parsed["run"]["expected_pairs"], 1);
        assert_eq!(parsed["run"]["resolved_pairs"], 1);
        assert_eq!(parsed["run"]["failed_pairs"], 0);
        assert_eq!(parsed["run"]["bootstrap_clusters"], 1);
        assert_eq!(parsed["run"]["bootstrap_ran"], false);
        assert_eq!(parsed["run"]["bootstrap_unavailable"], "too_few_tasks");
        assert!(parsed["run"].get("bootstrap_valid").is_none());
        assert_eq!(
            parsed["run"]["judge_decoding"],
            serde_json::json!({ "temperature": 0.0, "max_tokens": 4096 })
        );
        assert_eq!(parsed["run"]["candidate_max_tokens"], 4096);
        assert_eq!(
            parsed["run"]["orientation_agreement"],
            serde_json::json!({
                "resolved_pairs": 1,
                "orientation_agreeing_pairs": 0,
                "orientation_disagreeing_pairs": 1,
                "agreement_rate": 0.0
            })
        );
        assert_eq!(parsed["judgments"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["judgment_failures"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["statistics"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["ratings"].as_array().unwrap().len(), 2);
        assert!(parsed["judgments"][0].get("raw_ab").is_some());
        assert!(parsed["judgments"][0].get("raw_ba").is_some());
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
        assert!(parsed.get("judgment_failures").is_none());
        assert!(parsed.get("statistics").is_none());
        assert!(parsed.get("ratings").is_none());
        assert!(parsed["run"].get("complete").is_none());
        assert!(parsed["run"].get("expected_pairs").is_none());
        assert!(parsed["run"].get("resolved_pairs").is_none());
        assert!(parsed["run"].get("failed_pairs").is_none());
        assert!(parsed["run"].get("orientation_agreement").is_none());
        assert!(parsed["run"].get("judge_decoding").is_none());
        assert_eq!(parsed["run"]["candidate_max_tokens"], 4096);
        assert!(parsed["run"].get("bootstrap_seed").is_none());
        assert!(parsed["run"].get("bootstrap_clusters").is_none());
        assert!(parsed["run"].get("bootstrap_ran").is_none());
        assert!(parsed["run"].get("bootstrap_unavailable").is_none());
        assert!(json.starts_with('{'));
    }

    #[tokio::test]
    async fn judge_coverage_counts_unordered_pairs_across_tasks() {
        let cfg = ExecConfig {
            tasks: vec![
                sample_task(),
                Task {
                    id: "t2".into(),
                    prompt: "q".into(),
                    evaluation: None,
                },
            ],
            models: ["m0", "m1", "m2"].into_iter().map(ModelId::new).collect(),
            judge: Some(ModelId::new("judge")),
            seed: 0,
            tasks_path: Some(PathBuf::from("tasks.json")),
            started_at: "2026-01-02T03:04:05Z".into(),
            base_url: "https://example.test/v1".into(),
        };

        let (output, failed_pairs) = collect_exec(&OkProvider, &cfg, |_| {}, |_| {}, |_| {})
            .await
            .unwrap();

        assert_eq!(expected_pairs(2, 3), 6);
        assert_eq!(failed_pairs, 0);
        let parsed: serde_json::Value =
            serde_json::from_str(&persist::to_pretty_json(&output).unwrap()).unwrap();
        assert_eq!(parsed["run"]["complete"], true);
        assert_eq!(parsed["run"]["expected_pairs"], 6);
        assert_eq!(parsed["run"]["resolved_pairs"], 6);
        assert_eq!(parsed["run"]["failed_pairs"], 0);
        assert_eq!(
            parsed["run"]["expected_pairs"].as_u64().unwrap(),
            parsed["run"]["resolved_pairs"].as_u64().unwrap()
                + parsed["run"]["failed_pairs"].as_u64().unwrap()
        );
        assert_eq!(
            parsed["run"]["orientation_agreement"],
            serde_json::json!({
                "resolved_pairs": 6,
                "orientation_agreeing_pairs": 0,
                "orientation_disagreeing_pairs": 6,
                "agreement_rate": 0.0
            })
        );
        assert_eq!(parsed["judgments"].as_array().unwrap().len(), 6);
        assert_eq!(parsed["judgment_failures"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["ratings"].as_array().unwrap().len(), 3);
    }

    #[derive(Clone)]
    struct PrefersM0;

    impl ModelProvider for PrefersM0 {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            if request.model == ModelId::new("judge") {
                let a_is_m0 = request.prompt.contains("<response_a>\nm0\n</response_a>");
                let winner = if a_is_m0 { "a" } else { "b" };
                return Ok(CompletionResponse {
                    text: format!(r#"{{"winner":"{winner}","reason":"prefers m0"}}"#),
                });
            }
            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[tokio::test]
    async fn orientation_agreement_summary_counts_agreeing_pairs_once() {
        let (output, failed_pairs) = collect_exec(
            &PrefersM0,
            &config(&["m0", "m1"], Some("judge")),
            |_| {},
            |_| {},
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(failed_pairs, 0);
        let parsed: serde_json::Value =
            serde_json::from_str(&persist::to_pretty_json(&output).unwrap()).unwrap();
        assert_eq!(parsed["run"]["complete"], true);
        assert_eq!(
            parsed["run"]["orientation_agreement"],
            serde_json::json!({
                "resolved_pairs": 1,
                "orientation_agreeing_pairs": 1,
                "orientation_disagreeing_pairs": 0,
                "agreement_rate": 1.0
            })
        );
        let stats = parsed["statistics"].as_array().unwrap();
        let endpoint_agree: u64 = stats
            .iter()
            .map(|stat| stat["agreement_count"].as_u64().unwrap())
            .sum();
        assert_eq!(endpoint_agree, 2);
        assert_eq!(
            parsed["run"]["orientation_agreement"]["orientation_agreeing_pairs"],
            1
        );
    }

    #[tokio::test]
    async fn judge_run_with_zero_expected_pairs_keeps_empty_collections() {
        let (output, failed_pairs) = collect_exec(
            &OkProvider,
            &config(&["m0"], Some("judge")),
            |_| {},
            |_| {},
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(failed_pairs, 0);
        let parsed: serde_json::Value =
            serde_json::from_str(&persist::to_pretty_json(&output).unwrap()).unwrap();
        assert_eq!(parsed["run"]["complete"], true);
        assert_eq!(parsed["run"]["expected_pairs"], 0);
        assert_eq!(parsed["run"]["resolved_pairs"], 0);
        assert_eq!(parsed["run"]["failed_pairs"], 0);
        assert_eq!(
            parsed["run"]["orientation_agreement"]["agreement_rate"],
            0.0
        );
        assert_eq!(parsed["judgments"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["judgment_failures"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["statistics"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["ratings"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["ratings"][0]["unavailable"], "no_comparisons");
        assert!(parsed["ratings"][0]["rating"].is_null());
        assert!(parsed["ratings"][0].get("rating_lower").is_none());
        assert!(parsed["ratings"][0].get("rating_upper").is_none());
    }

    #[derive(Clone)]
    struct FailM0M1Judge;

    impl ModelProvider for FailM0M1Judge {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            if request.model == ModelId::new("judge") {
                let involves_m0 = request.prompt.contains("\nm0\n");
                let involves_m1 = request.prompt.contains("\nm1\n");
                if involves_m0 && involves_m1 {
                    return Ok(CompletionResponse {
                        text: "not a judgment".into(),
                    });
                }
                return Ok(CompletionResponse {
                    text: r#"{"winner":"a","reason":"ok"}"#.into(),
                });
            }
            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[tokio::test]
    async fn incomplete_judge_run_keeps_resolved_and_failed_coverage() {
        let (output, failed_pairs) = collect_exec(
            &FailM0M1Judge,
            &config(&["m0", "m1", "m2"], Some("judge")),
            |_| {},
            |_| {},
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(failed_pairs, 1);
        let parsed: serde_json::Value =
            serde_json::from_str(&persist::to_pretty_json(&output).unwrap()).unwrap();
        assert_eq!(parsed["run"]["complete"], false);
        assert_eq!(parsed["run"]["expected_pairs"], 3);
        assert_eq!(parsed["run"]["resolved_pairs"], 2);
        assert_eq!(parsed["run"]["failed_pairs"], 1);
        assert_eq!(parsed["judgments"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["judgment_failures"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["ratings"].as_array().unwrap().len(), 3);
        assert_eq!(parsed["run"]["orientation_agreement"]["resolved_pairs"], 2);
        assert_eq!(
            parsed["run"]["orientation_agreement"]["orientation_disagreeing_pairs"],
            2
        );
    }

    async fn drain_events(
        mut rx: mpsc::UnboundedReceiver<ExperimentEvent>,
    ) -> Vec<ExperimentEvent> {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        events
    }

    fn candidate_finished(events: &[ExperimentEvent]) -> Vec<&ExperimentEvent> {
        events
            .iter()
            .filter(|event| matches!(event, ExperimentEvent::CandidateFinished { .. }))
            .collect()
    }

    fn pair_resolved(events: &[ExperimentEvent]) -> Vec<&ExperimentEvent> {
        events
            .iter()
            .filter(|event| matches!(event, ExperimentEvent::PairResolved { .. }))
            .collect()
    }

    fn pair_failed(events: &[ExperimentEvent]) -> Vec<&ExperimentEvent> {
        events
            .iter()
            .filter(|event| matches!(event, ExperimentEvent::PairFailed { .. }))
            .collect()
    }

    #[tokio::test]
    async fn candidate_finished_emits_one_event_with_task_and_model() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (output, failed_pairs) =
            collect_exec_with_events(&OkProvider, &config(&["m0"], None), tx)
                .await
                .unwrap();
        assert_eq!(failed_pairs, 0);
        let events = drain_events(rx).await;
        let finished = candidate_finished(&events);
        assert_eq!(finished.len(), 1);
        match finished[0] {
            ExperimentEvent::CandidateFinished {
                task_id,
                model,
                duration_ms: _,
            } => {
                assert_eq!(task_id, "t1");
                assert_eq!(*model, ModelId::new("m0"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        assert_eq!(output.results[0].task_id, "t1");
        assert_eq!(output.results[0].model, ModelId::new("m0"));
    }

    #[tokio::test]
    async fn pair_resolution_emits_one_event_for_the_unordered_pair() {
        let (tx, rx) = mpsc::unbounded_channel();
        collect_exec_with_events(&PrefersM0, &config(&["m0", "m1"], Some("judge")), tx)
            .await
            .unwrap();
        let events = drain_events(rx).await;
        assert_eq!(pair_resolved(&events).len(), 1);
        assert!(pair_failed(&events).is_empty());
        match pair_resolved(&events)[0] {
            ExperimentEvent::PairResolved {
                task_id,
                model_a,
                model_b,
                judgment,
            } => {
                assert_eq!(task_id, "t1");
                assert_eq!(*model_a, ModelId::new("m0"));
                assert_eq!(*model_b, ModelId::new("m1"));
                assert_eq!(judgment.model_a, ModelId::new("m0"));
                assert_eq!(judgment.model_b, ModelId::new("m1"));
                assert_eq!(judgment.winner, JudgeDecision::A);
                assert!(judgment.agreement);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn orientation_disagreement_emits_one_resolved_draw() {
        let (tx, rx) = mpsc::unbounded_channel();
        collect_exec_with_events(&OkProvider, &config(&["m0", "m1"], Some("judge")), tx)
            .await
            .unwrap();
        let events = drain_events(rx).await;
        assert_eq!(pair_resolved(&events).len(), 1);
        assert!(pair_failed(&events).is_empty());
        match pair_resolved(&events)[0] {
            ExperimentEvent::PairResolved { judgment, .. } => {
                assert_eq!(judgment.winner, JudgeDecision::Draw);
                assert!(!judgment.agreement);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn orientation_failure_emits_one_pair_failed_and_no_resolved_pair() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (output, failed_pairs) =
            collect_exec_with_events(&FailJudge, &config(&["m0", "m1"], Some("judge")), tx)
                .await
                .unwrap();
        assert_eq!(failed_pairs, 1);
        let events = drain_events(rx).await;
        assert!(pair_resolved(&events).is_empty());
        assert_eq!(pair_failed(&events).len(), 1);
        match pair_failed(&events)[0] {
            ExperimentEvent::PairFailed {
                task_id,
                model_a,
                model_b,
                failure,
            } => {
                assert_eq!(task_id, "t1");
                assert_eq!(*model_a, ModelId::new("m0"));
                assert_eq!(*model_b, ModelId::new("m1"));
                assert_eq!(failure.task_id, "t1");
                assert_eq!(failure.model_a, ModelId::new("m0"));
                assert_eq!(failure.model_b, ModelId::new("m1"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        assert_eq!(output.judgments.as_ref().map(Vec::len), Some(0));
        assert_eq!(output.judgment_failures.as_ref().map(Vec::len), Some(1));
    }

    #[tokio::test]
    async fn run_complete_follows_pair_events_with_correct_counts() {
        let (tx, rx) = mpsc::unbounded_channel();
        collect_exec_with_events(
            &FailM0M1Judge,
            &config(&["m0", "m1", "m2"], Some("judge")),
            tx,
        )
        .await
        .unwrap();
        let events = drain_events(rx).await;
        let last = events.last().expect("events");
        match last {
            ExperimentEvent::RunComplete {
                expected_pairs,
                resolved_pairs,
                failed_pairs,
            } => {
                assert_eq!(*expected_pairs, 3);
                assert_eq!(*resolved_pairs, 2);
                assert_eq!(*failed_pairs, 1);
                assert_eq!(*resolved_pairs + *failed_pairs, *expected_pairs);
            }
            other => panic!("RunComplete must be last, got {other:?}"),
        }
        let complete_at = events
            .iter()
            .position(|event| matches!(event, ExperimentEvent::RunComplete { .. }))
            .unwrap();
        let last_pair = events
            .iter()
            .rposition(|event| {
                matches!(
                    event,
                    ExperimentEvent::PairResolved { .. } | ExperimentEvent::PairFailed { .. }
                )
            })
            .unwrap();
        assert!(last_pair < complete_at);
        assert_eq!(pair_resolved(&events).len(), 2);
        assert_eq!(pair_failed(&events).len(), 1);
        assert_eq!(candidate_finished(&events).len(), 3);
    }

    #[tokio::test]
    async fn no_judge_run_emits_no_pair_events_and_zero_pair_counts() {
        let (tx, rx) = mpsc::unbounded_channel();
        collect_exec_with_events(&OkProvider, &config(&["m0", "m1"], None), tx)
            .await
            .unwrap();
        let events = drain_events(rx).await;
        assert_eq!(candidate_finished(&events).len(), 2);
        assert!(pair_resolved(&events).is_empty());
        assert!(pair_failed(&events).is_empty());
        match events.last() {
            Some(ExperimentEvent::RunComplete {
                expected_pairs,
                resolved_pairs,
                failed_pairs,
            }) => {
                assert_eq!(*expected_pairs, 0);
                assert_eq!(*resolved_pairs, 0);
                assert_eq!(*failed_pairs, 0);
            }
            other => panic!("expected RunComplete, got {other:?}"),
        }
        let ids: Vec<_> = candidate_finished(&events)
            .into_iter()
            .map(|event| match event {
                ExperimentEvent::CandidateFinished { task_id, model, .. } => {
                    (task_id.as_str(), model.clone())
                }
                _ => unreachable!(),
            })
            .collect();
        assert!(ids.contains(&("t1", ModelId::new("m0"))));
        assert!(ids.contains(&("t1", ModelId::new("m1"))));
    }
}
