use std::time::Duration;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use arena::cli::{Cli, Command};
use arena::error::Result;
use arena::exec::{self, ExecConfig};
use arena::provider::{
    CompletionRequest, DEFAULT_MAX_TOKENS, ModelId, ModelProvider, OpenAICompatibleProvider,
};
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
                    temperature: None,
                    max_tokens: Some(DEFAULT_MAX_TOKENS),
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
            let models = exec::unique_models(models)?;
            let judge = judge.map(ModelId::new);
            exec::validate_judge(&models, judge.as_ref())?;
            let started_at = arena::persist::utc_timestamp();
            let provider = OpenAICompatibleProvider::from_env()?;
            let tasks = task::load(&tasks_path)?;

            let candidate_total = (tasks.len() * models.len()) as u64;
            let candidates = progress_bar(candidate_total);
            let pair_total = pair_count(tasks.len(), models.len());
            let judges = judge.as_ref().map(|_| progress_bar(pair_total));

            let config = ExecConfig {
                tasks,
                models,
                judge,
                seed,
                tasks_path: Some(tasks_path),
                started_at,
                base_url: provider.base_url().to_string(),
            };
            let (output_data, failed_pairs) = exec::collect_exec(
                &provider,
                &config,
                |result| {
                    candidates.set_message(format!("task {}  exec", result.task_id));
                    candidates.println(format!(
                        "exec  {}  {}  {}ms",
                        result.task_id, result.model, result.duration_ms
                    ));
                    candidates.inc(1);
                },
                |judgment| {
                    if let Some(judges) = &judges {
                        judges.set_message(format!("task {}  judge", judgment.task_id));
                        judges.println(format!(
                            "judge  {}  {} vs {}  {}ms",
                            judgment.task_id,
                            judgment.model_a,
                            judgment.model_b,
                            judgment.duration_ms
                        ));
                        judges.inc(1);
                    }
                },
                |failure| {
                    if let Some(judges) = &judges {
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
                },
            )
            .await?;
            candidates.finish_and_clear();
            if let Some(judges) = judges {
                judges.finish_and_clear();
            }

            exec::complete_exec(output_data, failed_pairs, output.as_deref())?;
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
