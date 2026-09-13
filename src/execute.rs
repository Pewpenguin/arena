use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelId, ModelProvider, PROVIDER_CONCURRENCY,
    ProviderError,
};
use crate::task::Task;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionResult {
    pub task_id: String,
    pub model: ModelId,
    pub response: CompletionResponse,
    pub duration_ms: u64,
}

#[derive(Debug, Error)]
#[error("execution failed for task {task_id} with model {model}: {source}")]
pub struct ExecutionError {
    pub task_id: String,
    pub model: ModelId,
    #[source]
    pub source: ProviderError,
}

pub async fn execute(
    provider: &impl ModelProvider,
    model: ModelId,
    task: &Task,
) -> Result<ExecutionResult, ExecutionError> {
    let request = CompletionRequest {
        model: model.clone(),
        prompt: task.prompt.clone(),
    };
    let started = Instant::now();
    let response = provider
        .complete(request)
        .await
        .map_err(|source| ExecutionError {
            task_id: task.id.clone(),
            model: model.clone(),
            source,
        })?;
    Ok(ExecutionResult {
        task_id: task.id.clone(),
        model,
        response,
        duration_ms: started.elapsed().as_millis() as u64,
    })
}

pub async fn execute_models<P>(
    provider: &P,
    task: &Task,
    models: &[ModelId],
    mut on_complete: impl FnMut(&ExecutionResult),
) -> Result<Vec<ExecutionResult>, ExecutionError>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let semaphore = Arc::new(Semaphore::new(PROVIDER_CONCURRENCY));
    let mut set = JoinSet::new();

    for (index, model) in models.iter().cloned().enumerate() {
        let provider = provider.clone();
        let task = task.clone();
        let semaphore = semaphore.clone();
        set.spawn(async move {
            let _permit = semaphore
                .acquire()
                .await
                .expect("provider semaphore is not closed");
            (index, execute(&provider, model, &task).await)
        });
    }

    let mut ordered = vec![None; models.len()];
    while let Some(joined) = set.join_next().await {
        let (index, result) = joined.expect("candidate execution panicked");
        let result = result?;
        on_complete(&result);
        ordered[index] = Some(result);
    }

    Ok(ordered
        .into_iter()
        .map(|result| result.expect("missing candidate result"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use tokio::sync::oneshot;

    #[derive(Clone)]
    struct GateProvider {
        tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
        rx: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    }

    impl ModelProvider for GateProvider {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            match request.model.to_string().as_str() {
                "slow" => {
                    let rx = self.rx.lock().expect("slow gate").take().expect("slow rx");
                    rx.await.expect("gate");
                }
                "fast" => {
                    let tx = self.tx.lock().expect("fast gate").take().expect("fast tx");
                    let _ = tx.send(());
                }
                _ => {}
            }

            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[tokio::test]
    async fn results_follow_model_order_when_completion_order_differs() {
        let (tx, rx) = oneshot::channel();
        let provider = GateProvider {
            tx: Arc::new(Mutex::new(Some(tx))),
            rx: Arc::new(Mutex::new(Some(rx))),
        };
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let models = vec![ModelId::new("slow"), ModelId::new("fast")];

        let results = execute_models(&provider, &task, &models, |_| {})
            .await
            .unwrap();

        assert_eq!(results[0].model, ModelId::new("slow"));
        assert_eq!(results[1].model, ModelId::new("fast"));
    }

    #[derive(Clone)]
    struct FailingProvider;

    impl ModelProvider for FailingProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            Err(ProviderError::RequestFailed("upstream down".into()))
        }
    }

    #[tokio::test]
    async fn execution_error_includes_task_and_model() {
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let error = execute_models(&FailingProvider, &task, &[ModelId::new("m1")], |_| {})
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("t1"), "{message}");
        assert!(message.contains("m1"), "{message}");
        assert!(message.contains("upstream down"), "{message}");
    }
}
