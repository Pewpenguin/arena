use clap::Parser;

use arena::cli::{Cli, Command};
use arena::error::Result;
use arena::evaluate::evaluate_result;
use arena::execute::execute_models;
use arena::persist;
use arena::provider::{CompletionRequest, DeepInfraProvider, ModelId, ModelProvider};
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
        } => {
            let provider = DeepInfraProvider::from_env()?;
            let tasks = task::load(tasks)?;
            let models: Vec<_> = models.into_iter().map(ModelId::new).collect();

            let mut results = Vec::new();
            for task in &tasks {
                for result in execute_models(&provider, task, &models).await? {
                    results.push(evaluate_result(task, result));
                }
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
