use std::fmt;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{
    CONNECT_TIMEOUT, CompletionRequest, CompletionResponse, ModelProvider, ProviderError,
    REQUEST_TIMEOUT,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const API_KEY_ENV: &str = "ARENA_API_KEY";
const BASE_URL_ENV: &str = "ARENA_BASE_URL";

#[derive(Clone)]
pub struct OpenAICompatibleProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl fmt::Debug for OpenAICompatibleProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAICompatibleProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenAICompatibleProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::with_timeouts(api_key, base_url, REQUEST_TIMEOUT, CONNECT_TIMEOUT)
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        dotenvy::dotenv().ok();

        let api_key = std::env::var(API_KEY_ENV).map_err(|_| {
            ProviderError::RequestFailed(format!("{API_KEY_ENV} environment variable is not set"))
        })?;
        let base_url = std::env::var(BASE_URL_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

        Ok(Self::new(api_key, base_url))
    }

    fn with_timeouts(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        request_timeout: Duration,
        connect_timeout: Duration,
    ) -> Self {
        Self {
            client: Client::builder()
                .timeout(request_timeout)
                .connect_timeout(connect_timeout)
                .build()
                .expect("failed to create HTTP client"),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl ModelProvider for OpenAICompatibleProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = ChatCompletionRequest {
            model: request.model.to_string(),
            messages: vec![ChatMessage {
                role: "user",
                content: request.prompt,
            }],
        };

        let response = self
            .client
            .post(chat_completions_url(&self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderError::RequestFailed(error.to_string()))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| ProviderError::RequestFailed(error.to_string()))?;

        if !status.is_success() {
            let body = String::from_utf8_lossy(&bytes);
            return Err(ProviderError::RequestFailed(format!(
                "HTTP {status}: {body}"
            )));
        }

        let parsed: ChatCompletionResponse = serde_json::from_slice(&bytes)
            .map_err(|error| ProviderError::InvalidResponse(error.to_string()))?;

        let text = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| ProviderError::InvalidResponse("missing completion content".into()))?;

        Ok(CompletionResponse { text })
    }
}

fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::oneshot;

    use super::*;
    use crate::provider::ModelId;

    struct MockServer {
        base_url: String,
        request: oneshot::Receiver<Vec<u8>>,
        handle: tokio::task::JoinHandle<()>,
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: ModelId::new("test-model"),
            prompt: "hello".into(),
        }
    }

    async fn start_mock(status: u16, reason: &str, body: &str) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (tx, rx) = oneshot::channel();
        let reason = reason.to_string();
        let body = body.to_string();

        let handle = tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };

            let request = read_http_request(&mut stream).await;
            let _ = tx.send(request);
            let content_length = body.len();
            let headers = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(headers.as_bytes()).await;
            let _ = stream.write_all(body.as_bytes()).await;
            let _ = stream.shutdown().await;
        });

        MockServer {
            base_url: format!("http://{addr}/v1/"),
            request: rx,
            handle,
        }
    }

    async fn start_stalling_mock() -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (_tx, rx) = oneshot::channel();

        let handle = tokio::spawn(async move {
            let Ok((_stream, _)) = listener.accept().await else {
                return;
            };
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        MockServer {
            base_url: format!("http://{addr}/v1/"),
            request: rx,
            handle,
        }
    }

    async fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut data = Vec::new();
        loop {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            data.extend_from_slice(&buf[..n]);
            if let Some(headers_end) = find_headers_end(&data) {
                let content_length = content_length(&data[..headers_end]).unwrap_or(0);
                if data.len() >= headers_end + content_length {
                    data.truncate(headers_end + content_length);
                    break;
                }
            }
        }
        data
    }

    fn find_headers_end(data: &[u8]) -> Option<usize> {
        data.windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
    }

    fn content_length(headers: &[u8]) -> Option<usize> {
        let text = std::str::from_utf8(headers).ok()?;
        for line in text.split("\r\n") {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            if name.eq_ignore_ascii_case("content-length") {
                return value.trim().parse().ok();
            }
        }
        None
    }

    fn header_value(request: &str, name: &str) -> Option<String> {
        request.lines().find_map(|line| {
            let (header_name, value) = line.split_once(':')?;
            header_name
                .eq_ignore_ascii_case(name)
                .then_some(value.trim().to_string())
        })
    }

    fn request_path(request: &str) -> &str {
        request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .expect("request path")
    }

    fn request_body(raw: &[u8]) -> serde_json::Value {
        let headers_end = find_headers_end(raw).expect("headers");
        serde_json::from_slice(&raw[headers_end..]).expect("json body")
    }

    #[test]
    fn chat_completions_url_joins_base_without_duplicate_slash() {
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn debug_omits_api_key() {
        let provider = OpenAICompatibleProvider::new("super-secret-key", "https://example.test/v1");
        let debug = format!("{provider:?}");
        assert!(!debug.contains("super-secret-key"), "{debug}");
        assert_eq!(provider.base_url(), "https://example.test/v1");
    }

    #[tokio::test]
    async fn complete_sends_chat_completion_and_returns_assistant_text() {
        let mock = start_mock(
            200,
            "OK",
            r#"{"choices":[{"message":{"content":"<think>reason</think>hello"}}]}"#,
        )
        .await;
        let provider = OpenAICompatibleProvider::new("test-key", mock.base_url.as_str());

        let response = provider
            .complete(sample_request())
            .await
            .expect("completion");
        let raw = mock.request.await.expect("captured request");
        mock.handle.abort();

        let request = String::from_utf8_lossy(&raw);
        assert_eq!(request_path(&request), "/v1/chat/completions");
        assert_eq!(
            header_value(&request, "authorization").as_deref(),
            Some("Bearer test-key")
        );
        assert_eq!(
            request_body(&raw),
            serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}]
            })
        );
        assert_eq!(response.text, "<think>reason</think>hello");
    }

    #[tokio::test]
    async fn complete_rejects_malformed_response() {
        let mock = start_mock(200, "OK", "not-json").await;
        let provider = OpenAICompatibleProvider::new("test-key", mock.base_url.as_str());

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("malformed response");
        mock.handle.abort();

        assert!(
            matches!(error, ProviderError::InvalidResponse(_)),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn complete_rejects_missing_completion_content() {
        let mock = start_mock(200, "OK", r#"{"choices":[{"message":{}}]}"#).await;
        let provider = OpenAICompatibleProvider::new("test-key", mock.base_url.as_str());

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("missing content");
        mock.handle.abort();

        assert_eq!(
            error,
            ProviderError::InvalidResponse("missing completion content".into())
        );
    }

    #[tokio::test]
    async fn complete_preserves_http_error_status_and_body() {
        let mock = start_mock(500, "Internal Server Error", "upstream exploded").await;
        let provider = OpenAICompatibleProvider::new("test-key", mock.base_url.as_str());

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("http error");
        mock.handle.abort();

        match error {
            ProviderError::RequestFailed(message) => {
                assert!(message.contains("HTTP 500"), "{message}");
                assert!(message.contains("upstream exploded"), "{message}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn complete_propagates_request_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(listener);

        let provider = OpenAICompatibleProvider::new("test-key", format!("http://{addr}/v1"));
        let error = provider
            .complete(sample_request())
            .await
            .expect_err("connection failure");

        assert!(
            matches!(error, ProviderError::RequestFailed(_)),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn complete_respects_request_timeout() {
        let mock = start_stalling_mock().await;
        let provider = OpenAICompatibleProvider::with_timeouts(
            "test-key",
            mock.base_url.as_str(),
            Duration::from_millis(50),
            Duration::from_millis(50),
        );

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("timeout");
        mock.handle.abort();

        assert!(
            matches!(error, ProviderError::RequestFailed(_)),
            "{error:?}"
        );
    }
}
