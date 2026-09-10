use clap::Parser;
use tracing_subscriber::EnvFilter;

use arena::cli::{Cli, Command};
use arena::config::Config;
use arena::error::{Error, Result};
use arena::execute::execute_models;
use arena::persist;
use arena::provider::{CompletionRequest, DeepInfraProvider, ModelId, ModelProvider};
use arena::task;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::from_cli(&cli);
    init_tracing(&config)?;

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
        } => {
            let provider = DeepInfraProvider::from_env()?;
            let tasks = task::load(tasks)?;
            let models: Vec<_> = models.into_iter().map(ModelId::new).collect();

            let mut results = Vec::new();
            for task in &tasks {
                results.extend(execute_models(&provider, task, &models).await?);
            }

            if let Some(path) = output {
                persist::write(path, &results)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&results)?);
            }
        }
    }

    Ok(())
}

fn init_tracing(config: &Config) -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(config.log_level.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|_| Error::LoggingInit)?;

    Ok(())
}
