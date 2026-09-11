use std::time::Instant;

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
    pub duration_ms: u64,
}

pub async fn execute(
    provider: &impl ModelProvider,
    model: ModelId,
    task: &Task,
) -> Result<ExecutionResult, ProviderError> {
    let request = CompletionRequest {
        model: model.clone(),
        prompt: task.prompt.clone(),
    };
    let started = Instant::now();
    let response = provider.complete(request).await?;
    Ok(ExecutionResult {
        task_id: task.id.clone(),
        model,
        response,
        duration_ms: started.elapsed().as_millis() as u64,
    })
}
