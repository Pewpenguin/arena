use std::collections::HashSet;
use std::time::Duration;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use arena::bootstrap;
use arena::cli::{Cli, Command};
use arena::compare::compare_all;
use arena::error::{Error, Result};
use arena::evaluate::evaluate_result;
use arena::execute::execute_models;
use arena::judge::judge_pairs;
use arena::persist::{self, Output, RunMetadata};
use arena::provider::{CompletionRequest, ModelId, ModelProvider, OpenAICompatibleProvider};
use arena::stats;
use arena::task;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run { model, prompt } => {
            let provider = OpenAICompatibleProvider::from_env()?;
            let response = provider
                .complete(CompletionRequest {
                    model: ModelId::new(model),
                    prompt,
                })
                .await?;
            println!("{}", response.text);
        }
        Command::Exec {
            tasks: tasks_path,
            models,
            output,
            judge,
            seed,
        } => {
            let models: Vec<_> = models.into_iter().map(ModelId::new).collect();
            let mut seen = HashSet::new();
            for model in &models {
                if !seen.insert(model) {
                    return Err(Error::DuplicateModel(model.clone()));
                }
            }

            let started_at = persist::utc_timestamp();
            let provider = OpenAICompatibleProvider::from_env()?;
            let tasks = task::load(&tasks_path)?;
            let judge = judge.map(ModelId::new);

            let mut results = Vec::new();
            let candidates = progress_bar((tasks.len() * models.len()) as u64);
            for task in &tasks {
                candidates.set_message(format!("task {}  exec", task.id));
                let executed = execute_models(&provider, task, &models, |result| {
                    candidates.println(format!(
                        "exec  {}  {}  {}ms",
                        task.id, result.model, result.duration_ms
                    ));
                    candidates.inc(1);
                })
                .await?;
                for result in executed {
                    results.push(evaluate_result(task, result));
                }
            }
            candidates.finish_and_clear();

            let comparisons = compare_all(&results);

            let mut judgments = Vec::new();
            let mut judgment_failures = Vec::new();
            if let Some(judge_model) = &judge {
                let pair_total = pair_count(tasks.len(), models.len());
                let judges = progress_bar(pair_total);
                for task in &tasks {
                    let task_results: Vec<_> = results
                        .iter()
                        .filter(|result| result.task_id == task.id)
                        .cloned()
                        .collect();

                    judges.set_message(format!("task {}  judge", task.id));
                    let outcome = judge_pairs(
                        &provider,
                        judge_model.clone(),
                        task,
                        &task_results,
                        |judgment| {
                            judges.println(format!(
                                "judge  {}  {} vs {}  {}ms",
                                judgment.task_id,
                                judgment.model_a,
                                judgment.model_b,
                                judgment.duration_ms
                            ));
                            judges.inc(1);
                        },
                    )
                    .await?;
                    for failure in &outcome.failures {
                        let detail = failure
                            .orientations
                            .iter()
                            .map(|item| {
                                format!("{} {}", item.orientation.as_str(), item.kind.as_str())
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        judges.println(format!(
                            "judge  {}  {} vs {}  failed  {}",
                            failure.task_id, failure.model_a, failure.model_b, detail
                        ));
                        judges.inc(failure.orientations.len() as u64);
                    }
                    judgments.extend(outcome.judgments);
                    judgment_failures.extend(outcome.failures);
                }
                judges.finish_and_clear();
            }

            let statistics = stats::aggregate(&judgments);
            let (ratings, bootstrap) = bootstrap::rate_with_uncertainty(&judgments, seed);
            let mut run = RunMetadata::new(
                models,
                judge,
                Some(tasks_path),
                started_at,
                provider.base_url(),
            );
            if let Some(meta) = bootstrap {
                run = run.with_bootstrap(meta.seed, meta.replicates, meta.valid);
            }
            let failed_pairs = judgment_failures.len();
            let output_data = Output {
                run,
                tasks,
                results,
                comparisons,
                judgments,
                judgment_failures,
                statistics,
                ratings,
            };

            if let Some(path) = output {
                persist::write(path, &output_data)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&output_data)?);
            }

            if failed_pairs > 0 {
                return Err(Error::IncompleteJudgments(failed_pairs));
            }
        }
    }

    Ok(())
}

fn pair_count(tasks: usize, models: usize) -> u64 {
    let n = models as u64;
    tasks as u64 * n.saturating_sub(1) * n / 2 * 2
}

fn progress_bar(len: u64) -> ProgressBar {
    let bar = ProgressBar::new(len);
    bar.set_style(
        ProgressStyle::with_template("{spinner} {elapsed_precise} {pos}/{len} {wide_msg}")
            .expect("progress template"),
    );
    bar.enable_steady_tick(Duration::from_millis(100));
    bar
}
