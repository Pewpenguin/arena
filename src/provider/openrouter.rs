use std::fmt;

use serde::{Deserialize, Serialize};

use super::http;
use super::{CompletionRequest, CompletionResponse, ModelProvider, ProviderError};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const API_KEY_ENV: &str = "ARENA_OPENROUTER_API_KEY";

#[derive(Clone)]
pub struct OpenRouterProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl fmt::Debug for OpenRouterProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenRouterProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenRouterProvider {
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

impl ModelProvider for OpenRouterProvider {
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
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        };

        let bytes = http::execute_json(
            self.client
                .post(chat_completions_url(&self.base_url))
                .bearer_auth(&self.api_key)
                .json(&body),
            &self.api_key,
        )
        .await?;

        let parsed: ChatCompletionResponse = serde_json::from_slice(&bytes)
            .map_err(|error| http::invalid_json(&error, &self.api_key))?;

        let choice =
            parsed.choices.into_iter().next().ok_or_else(|| {
                ProviderError::InvalidResponse("missing completion content".into())
            })?;
        http::reject_length_limit(choice.finish_reason.as_deref(), "length")?;
        let text = choice
            .message
            .content
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
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
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
    #[serde(default)]
    finish_reason: Option<String>,
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::{header_value, request_body, request_path, start_mock};
    use crate::provider::{DEFAULT_MAX_TOKENS, ModelId};

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: ModelId::new("vendor/model"),
            prompt: "hello".into(),
            temperature: Some(0.0),
            max_tokens: Some(DEFAULT_MAX_TOKENS),
        }
    }

    #[test]
    fn chat_completions_url_joins_base_without_duplicate_slash() {
        assert_eq!(
            chat_completions_url("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn complete_posts_openrouter_chat_completion() {
        let mock = start_mock(
            200,
            "OK",
            r#"{"choices":[{"message":{"role":"assistant","content":"native-ok"}}]}"#,
        )
        .await;
        let base_url = format!("{}/api/v1/", mock.origin);
        let provider = OpenRouterProvider::new("test-key", &base_url);

        let response = provider
            .complete(sample_request())
            .await
            .expect("completion");
        let raw = mock.request.await.expect("captured request");
        mock.handle.abort();

        let request = String::from_utf8_lossy(&raw);
        assert_eq!(request_path(&request), "/api/v1/chat/completions");
        assert!(!request_path(&request).contains("test-key"));
        assert_eq!(
            header_value(&request, "authorization").as_deref(),
            Some("Bearer test-key")
        );
        assert_eq!(
            request_body(&raw),
            serde_json::json!({
                "model": "vendor/model",
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
        let provider = OpenRouterProvider::new("test-key", format!("{}/api/v1", mock.origin));

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
        let provider =
            OpenRouterProvider::new("super-secret-key", format!("{}/api/v1", mock.origin));

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
        assert!(!message.contains("Bearer"), "{message}");
        assert!(!rendered.contains("super-secret-key"), "{rendered}");
    }

    #[tokio::test]
    async fn complete_accepts_stop_finish_reason() {
        let mock = start_mock(
            200,
            "OK",
            r#"{"choices":[{"finish_reason":"stop","message":{"content":"full answer"}}]}"#,
        )
        .await;
        let provider = OpenRouterProvider::new("test-key", format!("{}/api/v1", mock.origin));

        let response = provider
            .complete(sample_request())
            .await
            .expect("completion");
        mock.handle.abort();

        assert_eq!(response.text, "full answer");
    }

    #[tokio::test]
    async fn complete_rejects_length_finish_reason() {
        let mock = start_mock(
            200,
            "OK",
            r#"{"choices":[{"finish_reason":"length","message":{"content":"partial answer"}}]}"#,
        )
        .await;
        let provider = OpenRouterProvider::new("test-key", format!("{}/api/v1", mock.origin));

        let error = provider
            .complete(sample_request())
            .await
            .expect_err("truncated");
        mock.handle.abort();

        assert_eq!(
            error,
            ProviderError::InvalidResponse("completion truncated by output length limit".into())
        );
        assert!(crate::retry::is_retryable_provider(&error));
        assert!(!error.to_string().contains("partial answer"));
    }
}
