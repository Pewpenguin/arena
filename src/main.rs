use std::time::Duration;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use arena::cli::{Cli, Command};
use arena::compare::compare_all;
use arena::error::Result;
use arena::evaluate::evaluate_result;
use arena::execute::execute;
use arena::judge::judge_pair;
use arena::persist::{self, Output};
use arena::provider::{CompletionRequest, DeepInfraProvider, ModelId, ModelProvider};
use arena::stats;
use arena::task;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run { model, prompt } => {
            let provider = DeepInfraProvider::from_env()?;
            let response = provider
                .complete(CompletionRequest {
                    model: ModelId::new(model),
                    prompt,
                })
                .await?;
            println!("{}", response.text);
        }
        Command::Exec {
            tasks,
            models,
            output,
            judge,
        } => {
            let provider = DeepInfraProvider::from_env()?;
            let tasks = task::load(tasks)?;
            let models: Vec<_> = models.into_iter().map(ModelId::new).collect();

            let mut results = Vec::new();
            let candidates = progress_bar((tasks.len() * models.len()) as u64);
            for task in &tasks {
                for model in &models {
                    candidates.set_message(format!("task {}  exec  {model}", task.id));
                    let result = execute(&provider, model.clone(), task).await?;
                    candidates.println(format!(
                        "exec  {}  {model}  {}ms",
                        task.id, result.duration_ms
                    ));
                    results.push(evaluate_result(task, result));
                    candidates.inc(1);
                }
            }
            candidates.finish_and_clear();

            let comparisons = compare_all(&results);

            let mut judgments = Vec::new();
            if let Some(judge_model) = judge {
                let judge_model = ModelId::new(judge_model);
                let pair_total = pair_count(tasks.len(), models.len());
                let judges = progress_bar(pair_total);
                for task in &tasks {
                    let task_results: Vec<_> = results
                        .iter()
                        .filter(|result| result.task_id == task.id)
                        .collect();

                    for i in 0..task_results.len() {
                        for j in (i + 1)..task_results.len() {
                            let model_a = &task_results[i].model;
                            let model_b = &task_results[j].model;
                            judges.set_message(format!(
                                "task {}  judge  {model_a} vs {model_b}",
                                task.id
                            ));
                            let judgment = judge_pair(
                                &provider,
                                judge_model.clone(),
                                task,
                                task_results[i],
                                task_results[j],
                            )
                            .await?;
                            judges.println(format!(
                                "judge  {}  {model_a} vs {model_b}  {}ms",
                                task.id, judgment.duration_ms
                            ));
                            judgments.push(judgment);
                            judges.inc(1);
                        }
                    }
                }
                judges.finish_and_clear();
            }

            let statistics = stats::aggregate(&judgments);
            let output_data = Output {
                results,
                comparisons,
                judgments,
                statistics,
            };

            if let Some(path) = output {
                persist::write(path, &output_data)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&output_data)?);
            }
        }
    }

    Ok(())
}

fn pair_count(tasks: usize, models: usize) -> u64 {
    let n = models as u64;
    tasks as u64 * n.saturating_sub(1) * n / 2
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
