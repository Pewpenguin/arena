use reqwest::{Client, RequestBuilder};

use super::{CONNECT_TIMEOUT, ProviderError, REQUEST_TIMEOUT};

pub(super) fn client() -> Client {
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .expect("failed to create HTTP client")
}

pub(super) async fn execute_json(
    request: RequestBuilder,
    secret: &str,
) -> Result<Vec<u8>, ProviderError> {
    let response = request
        .send()
        .await
        .map_err(|error| ProviderError::RequestFailed(redact(&error.to_string(), secret)))?;

    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| ProviderError::RequestFailed(redact(&error.to_string(), secret)))?;

    if !status.is_success() {
        let body = String::from_utf8_lossy(&bytes);
        let message = format!("HTTP {status}: {body}");
        return Err(ProviderError::RequestFailed(redact(&message, secret)));
    }

    Ok(bytes.to_vec())
}

pub(super) fn invalid_json(error: &serde_json::Error, secret: &str) -> ProviderError {
    ProviderError::InvalidResponse(redact(&error.to_string(), secret))
}

fn redact(message: &str, secret: &str) -> String {
    if secret.is_empty() {
        message.to_string()
    } else {
        message.replace(secret, "[redacted]")
    }
}
