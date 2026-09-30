use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use arena::cli::{Cli, Command, ProviderChoice};
use arena::error::Result;
use arena::exec::{self, ExecConfig};
use arena::html;
use arena::persist;
use arena::provider::{
    AnthropicProvider, CompletionRequest, DEFAULT_MAX_TOKENS, GeminiProvider, ModelId,
    ModelProvider, OpenAICompatibleProvider, OpenRouterProvider,
};
use arena::report;
use arena::task;
use arena::tournament::{self, TournamentFormat};
use arena::web;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            provider,
            model,
            prompt,
        } => run_with_provider(provider, model, prompt).await?,
        Command::Exec {
            provider,
            tasks: tasks_path,
            models,
            output,
            judge,
            tournament,
            best_of,
            seed,
        } => {
            exec_with_provider(
                provider, tasks_path, models, output, judge, tournament, best_of, seed,
            )
            .await?;
        }
        Command::Report { input, output } => {
            let data = persist::read(&input)?;
            let report = report::from_output(&data);
            let html = html::render(&report);
            std::fs::write(&output, html).map_err(persist::PersistError::from)?;
        }
        Command::Web { port } => {
            web::serve(port).await?;
        }
    }

    Ok(())
}

async fn run_with_provider(choice: ProviderChoice, model: String, prompt: String) -> Result<()> {
    match choice {
        ProviderChoice::Openai => {
            run_prompt(OpenAICompatibleProvider::from_env()?, model, prompt).await
        }
        ProviderChoice::Openrouter => {
            run_prompt(OpenRouterProvider::from_env()?, model, prompt).await
        }
        ProviderChoice::Anthropic => {
            run_prompt(AnthropicProvider::from_env()?, model, prompt).await
        }
        ProviderChoice::Gemini => run_prompt(GeminiProvider::from_env()?, model, prompt).await,
    }
}

async fn run_prompt(provider: impl ModelProvider, model: String, prompt: String) -> Result<()> {
    let response = provider
        .complete(CompletionRequest {
            model: ModelId::new(model),
            prompt,
            temperature: None,
            max_tokens: Some(DEFAULT_MAX_TOKENS),
        })
        .await?;
    println!("{}", response.text);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn exec_with_provider(
    choice: ProviderChoice,
    tasks_path: PathBuf,
    models: Vec<String>,
    output: Option<PathBuf>,
    judge: Option<String>,
    tournament: TournamentFormat,
    best_of: u32,
    seed: u64,
) -> Result<()> {
    match choice {
        ProviderChoice::Openai => {
            let provider = OpenAICompatibleProvider::from_env()?;
            let base_url = provider.base_url().to_string();
            run_exec(
                provider, base_url, tasks_path, models, output, judge, tournament, best_of, seed,
            )
            .await
        }
        ProviderChoice::Openrouter => {
            let provider = OpenRouterProvider::from_env()?;
            let base_url = provider.base_url().to_string();
            run_exec(
                provider, base_url, tasks_path, models, output, judge, tournament, best_of, seed,
            )
            .await
        }
        ProviderChoice::Anthropic => {
            let provider = AnthropicProvider::from_env()?;
            let base_url = provider.base_url().to_string();
            run_exec(
                provider, base_url, tasks_path, models, output, judge, tournament, best_of, seed,
            )
            .await
        }
        ProviderChoice::Gemini => {
            let provider = GeminiProvider::from_env()?;
            let base_url = provider.base_url().to_string();
            run_exec(
                provider, base_url, tasks_path, models, output, judge, tournament, best_of, seed,
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_exec<P>(
    provider: P,
    base_url: String,
    tasks_path: PathBuf,
    models: Vec<String>,
    output: Option<PathBuf>,
    judge: Option<String>,
    tournament: TournamentFormat,
    best_of: u32,
    seed: u64,
) -> Result<()>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let models = exec::unique_models(models)?;
    let judge = judge.map(ModelId::new);
    exec::validate_judge(&models, judge.as_ref())?;
    let started_at = persist::utc_timestamp();
    let tasks = task::load(&tasks_path)?;
    tournament::validate(tournament, models.len())?;
    tournament::validate_best_of(best_of)?;

    let candidate_total = (tasks.len() * models.len()) as u64;
    let candidates = progress_bar(candidate_total);
    let pair_total = judge_progress_len(tournament, tasks.len(), models.len(), best_of);
    let judges = judge.as_ref().map(|_| progress_bar(pair_total));

    let config = ExecConfig {
        tasks,
        models,
        judge,
        tournament,
        best_of,
        opening_matchups: None,
        seed,
        tasks_path: Some(tasks_path),
        started_at,
        base_url,
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
                judges.set_message(format!("task {}  judge game", judgment.task_id));
                judges.println(format!(
                    "judge  {}  {} vs {}  {}ms",
                    judgment.task_id, judgment.model_a, judgment.model_b, judgment.duration_ms
                ));
                judges.inc(1);
            }
        },
        |failure| {
            if let Some(judges) = &judges {
                let detail = failure
                    .orientations
                    .iter()
                    .map(|item| format!("{} {}", item.orientation.as_str(), item.kind.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ");
                judges.println(format!(
                    "judge  {}  {} vs {}  failed  {}",
                    failure.task_id, failure.model_a, failure.model_b, detail
                ));
                judges.inc(1);
            }
        },
    )
    .await?;
    candidates.finish_and_clear();
    if let Some(judges) = judges {
        judges.finish_and_clear();
    }

    exec::complete_exec(output_data, failed_pairs, output.as_deref())
}

fn judge_progress_len(format: TournamentFormat, tasks: usize, models: usize, best_of: u32) -> u64 {
    tournament::planned_match_count(format, models)
        .saturating_mul(tasks)
        .saturating_mul(best_of as usize) as u64
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

#[cfg(test)]
mod tests {
    use super::judge_progress_len;
    use arena::tournament::TournamentFormat;

    #[test]
    fn judge_progress_length_matches_expected_unordered_pairs() {
        assert_eq!(judge_progress_len(TournamentFormat::RoundRobin, 1, 2, 1), 1);
        assert_eq!(judge_progress_len(TournamentFormat::RoundRobin, 2, 3, 1), 6);
        assert_eq!(judge_progress_len(TournamentFormat::RoundRobin, 0, 4, 1), 0);
        assert_eq!(
            judge_progress_len(TournamentFormat::RoundRobin, 3, 4, 1),
            arena::exec::expected_pairs(3, 4) as u64
        );
        assert_eq!(
            judge_progress_len(TournamentFormat::SingleElimination, 2, 4, 1),
            6
        );
        assert_eq!(
            judge_progress_len(TournamentFormat::KingOfTheHill, 2, 3, 1),
            4
        );
        assert_eq!(judge_progress_len(TournamentFormat::RoundRobin, 1, 2, 3), 3);
    }
}
