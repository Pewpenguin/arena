use crate::provider::{
    CompletionRequest, CompletionResponse, ModelId, ModelProvider, ProviderError,
};
use crate::task::Task;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionResult {
    pub task_id: String,
    pub model: ModelId,
    pub response: CompletionResponse,
}

pub async fn execute(
    provider: &impl ModelProvider,
    model: ModelId,
    task: &Task,
) -> Result<CompletionResponse, ProviderError> {
    provider
        .complete(CompletionRequest {
            model,
            prompt: task.prompt.clone(),
        })
        .await
}

pub async fn execute_models(
    provider: &impl ModelProvider,
    task: &Task,
    models: &[ModelId],
) -> Result<Vec<ExecutionResult>, ProviderError> {
    let mut results = Vec::with_capacity(models.len());

    for model in models {
        let response = execute(provider, model.clone(), task).await?;
        results.push(ExecutionResult {
            task_id: task.id.clone(),
            model: model.clone(),
            response,
        });
    }

    Ok(results)
}
