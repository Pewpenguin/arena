use std::future::Future;
use std::time::Duration;

use crate::provider::ProviderError;

pub(crate) const ATTEMPTS: u32 = 3;

pub(crate) fn backoff(failed_attempts: u32) -> Option<Duration> {
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

pub(crate) fn is_retryable_provider(error: &ProviderError) -> bool {
    matches!(
        error,
        ProviderError::RequestFailed(_) | ProviderError::InvalidResponse(_)
    )
}

pub(crate) async fn with_retries<T, E, F, Fut, R>(mut op: F, is_retryable: R) -> (u32, Result<T, E>)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    R: Fn(&E) -> bool,
{
    let mut attempts = 0;
    loop {
        attempts += 1;
        match op().await {
            Ok(value) => return (attempts, Ok(value)),
            Err(error) if attempts < ATTEMPTS && is_retryable(&error) => {
                if let Some(delay) = backoff(attempts) {
                    tokio::time::sleep(delay).await;
                }
            }
            Err(error) => return (attempts, Err(error)),
        }
    }
}
