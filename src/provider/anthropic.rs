use std::fmt;

use serde::{Deserialize, Serialize};

use super::http;
use super::{CompletionRequest, CompletionResponse, ModelProvider, ProviderError};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_KEY_ENV: &str = "ARENA_ANTHROPIC_API_KEY";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: http::client(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        dotenvy::dotenv().ok();

        let api_key = std::env::var(API_KEY_ENV).map_err(|_| {
            ProviderError::RequestFailed(format!("{API_KEY_ENV} environment variable is not set"))
        })?;

        Ok(Self::new(api_key, DEFAULT_BASE_URL))
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl ModelProvider for AnthropicProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = MessagesRequest {
            model: request.model.to_string(),
            messages: vec![Message {
                role: "user",
                content: request.prompt,
            }],
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        };

        let bytes = http::execute_json(
            self.client
                .post(messages_url(&self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body),
            &self.api_key,
        )
        .await?;

        let parsed: MessagesResponse = serde_json::from_slice(&bytes)
            .map_err(|error| http::invalid_json(&error, &self.api_key))?;

        Ok(CompletionResponse {
            text: message_text(parsed.content)?,
        })
    }
}

fn messages_url(base_url: &str) -> String {
    format!("{}/messages", base_url.trim_end_matches('/'))
}

fn message_text(blocks: Vec<ContentBlock>) -> Result<String, ProviderError> {
    let mut text = String::new();
    let mut saw_text = false;
    for block in blocks {
        if block.kind == "text"
            && let Some(piece) = block.text
        {
            saw_text = true;
            text.push_str(&piece);
        }
    }
    if saw_text {
        Ok(text)
    } else {
        Err(ProviderError::InvalidResponse(
            "missing completion content".into(),
        ))
    }
}

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::{header_value, request_body, request_path, start_mock};
    use crate::provider::{DEFAULT_MAX_TOKENS, ModelId};

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: ModelId::new("claude-test"),
            prompt: "hello".into(),
            temperature: Some(0.0),
            max_tokens: Some(DEFAULT_MAX_TOKENS),
        }
    }

    #[test]
    fn messages_url_joins_base_without_duplicate_slash() {
        assert_eq!(
            messages_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            messages_url("https://api.anthropic.com/v1/"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[tokio::test]
    async fn complete_posts_native_messages_request() {
        let mock = start_mock(
            200,
            "OK",
            r#"{"content":[{"type":"text","text":"native-ok"}],"role":"assistant"}"#,
        )
        .await;
        let base_url = format!("{}/v1/", mock.origin);
        let provider = AnthropicProvider::new("test-key", &base_url);

        let response = provider
            .complete(sample_request())
            .await
            .expect("completion");
        let raw = mock.request.await.expect("captured request");
        mock.handle.abort();

        let request = String::from_utf8_lossy(&raw);
        assert_eq!(request_path(&request), "/v1/messages");
        assert!(!request_path(&request).contains("test-key"));
        assert_eq!(
            header_value(&request, "x-api-key").as_deref(),
            Some("test-key")
        );
        assert_eq!(
            header_value(&request, "anthropic-version").as_deref(),
            Some("2023-06-01")
        );
        assert_eq!(
            request_body(&raw),
            serde_json::json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.0,
                "max_tokens": 4096
            })
        );
        assert_eq!(response.text, "native-ok");
    }

    #[tokio::test]
    async fn complete_rejects_malformed_response() {
        let mock = start_mock(200, "OK", "not-json").await;
        let provider = AnthropicProvider::new("test-key", format!("{}/v1", mock.origin));

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("malformed response");
        mock.handle.abort();

        assert!(
            matches!(error, ProviderError::InvalidResponse(_)),
            "{error:?}"
        );
        assert!(!error.to_string().contains("test-key"), "{error}");
    }

    #[tokio::test]
    async fn api_key_is_not_exposed() {
        let mock = start_mock(
            401,
            "Unauthorized",
            r#"{"error":"rejected super-secret-key"}"#,
        )
        .await;
        let provider = AnthropicProvider::new("super-secret-key", format!("{}/v1", mock.origin));

        let debug = format!("{provider:?}");
        assert!(debug.contains(provider.base_url()), "{debug}");
        assert!(!debug.contains("super-secret-key"), "{debug}");

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("http error");
        mock.handle.abort();

        let message = error.to_string();
        let rendered = format!("{error:?}");
        assert!(message.contains("HTTP 401"), "{message}");
        assert!(message.contains("[redacted]"), "{message}");
        assert!(!message.contains("super-secret-key"), "{message}");
        assert!(!rendered.contains("super-secret-key"), "{rendered}");
    }
}
