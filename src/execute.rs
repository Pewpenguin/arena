use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelId, ModelProvider, PROVIDER_CONCURRENCY,
    ProviderError,
};
use crate::task::Task;

const CANDIDATE_ATTEMPTS: u32 = 3;

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

fn is_retryable(error: &ProviderError) -> bool {
    matches!(
        error,
        ProviderError::RequestFailed(_) | ProviderError::InvalidResponse(_)
    )
}

fn retry_backoff(failed_attempts: u32) -> Option<Duration> {
    let secs = match failed_attempts {
        1 => 1,
        2 => 2,
        _ => return None,
    };
    Some(if cfg!(test) {
        Duration::ZERO
    } else {
        Duration::from_secs(secs)
    })
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
    let mut attempts = 0;
    loop {
        attempts += 1;
        match provider.complete(request.clone()).await {
            Ok(response) => {
                return Ok(ExecutionResult {
                    task_id: task.id.clone(),
                    model,
                    response,
                    duration_ms: started.elapsed().as_millis() as u64,
                });
            }
            Err(source) if attempts < CANDIDATE_ATTEMPTS && is_retryable(&source) => {
                if let Some(delay) = retry_backoff(attempts) {
                    tokio::time::sleep(delay).await;
                }
            }
            Err(source) => {
                return Err(ExecutionError {
                    task_id: task.id.clone(),
                    model,
                    source,
                });
            }
        }
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use tokio::sync::oneshot;

    fn sample_task() -> Task {
        Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        }
    }

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
        let models = vec![ModelId::new("slow"), ModelId::new("fast")];

        let results = execute_models(&provider, &sample_task(), &models, |_| {})
            .await
            .unwrap();

        assert_eq!(results[0].model, ModelId::new("slow"));
        assert_eq!(results[1].model, ModelId::new("fast"));
    }

    #[derive(Clone)]
    struct FailThenSucceed {
        calls: Arc<AtomicUsize>,
    }

    impl ModelProvider for FailThenSucceed {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(ProviderError::RequestFailed("transient".into()));
            }
            Ok(CompletionResponse {
                text: "recovered".into(),
            })
        }
    }

    #[tokio::test]
    async fn transient_provider_error_retries_and_uses_the_successful_response() {
        let provider = FailThenSucceed {
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let results = execute_models(&provider, &sample_task(), &[ModelId::new("m1")], |_| {})
            .await
            .unwrap();

        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_id, "t1");
        assert_eq!(results[0].model, ModelId::new("m1"));
        assert_eq!(results[0].response.text, "recovered");
    }

    #[derive(Clone)]
    struct AlwaysFail {
        calls: Arc<AtomicUsize>,
    }

    impl ModelProvider for AlwaysFail {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(ProviderError::RequestFailed("upstream down".into()))
        }
    }

    #[tokio::test]
    async fn permanent_provider_error_exhausts_three_attempts_and_keeps_context() {
        let provider = AlwaysFail {
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let error = execute_models(&provider, &sample_task(), &[ModelId::new("m1")], |_| {})
            .await
            .unwrap_err();

        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            CANDIDATE_ATTEMPTS as usize
        );
        assert_eq!(error.task_id, "t1");
        assert_eq!(error.model, ModelId::new("m1"));
        assert_eq!(
            error.source,
            ProviderError::RequestFailed("upstream down".into())
        );
        let message = error.to_string();
        assert!(message.contains("t1"), "{message}");
        assert!(message.contains("m1"), "{message}");
        assert!(message.contains("upstream down"), "{message}");
    }
}
