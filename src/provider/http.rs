use reqwest::redirect::Policy;
use reqwest::{Client, RequestBuilder, Url};

use super::{CONNECT_TIMEOUT, ProviderError, REQUEST_TIMEOUT};

const MAX_REDIRECTS: usize = 10;

pub(super) fn client() -> Client {
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .redirect(redirect_policy())
        .build()
        .expect("failed to create HTTP client")
}

fn redirect_policy() -> Policy {
    Policy::custom(|attempt| {
        if attempt.previous().len() > MAX_REDIRECTS {
            return attempt.error("too many redirects");
        }
        match attempt.previous().last() {
            Some(previous) if same_origin(previous, attempt.url()) => attempt.follow(),
            Some(_) => attempt.stop(),
            None => attempt.follow(),
        }
    })
}

fn same_origin(left: &Url, right: &Url) -> bool {
    left.scheme() == right.scheme()
        && left.host() == right.host()
        && left.port_or_known_default() == right.port_or_known_default()
}

pub(super) fn reject_length_limit(
    reason: Option<&str>,
    length_reason: &str,
) -> Result<(), ProviderError> {
    if reason == Some(length_reason) {
        Err(ProviderError::InvalidResponse(
            "completion truncated by output length limit".into(),
        ))
    } else {
        Ok(())
    }
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
