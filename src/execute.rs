use std::time::Instant;

use tokio::task::JoinSet;

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

pub async fn execute_models<P>(
    provider: &P,
    task: &Task,
    models: &[ModelId],
    mut on_complete: impl FnMut(&ExecutionResult),
) -> Result<Vec<ExecutionResult>, ProviderError>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let mut set = JoinSet::new();

    for (index, model) in models.iter().cloned().enumerate() {
        let provider = provider.clone();
        let task = task.clone();
        set.spawn(async move { (index, execute(&provider, model, &task).await) });
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
}
