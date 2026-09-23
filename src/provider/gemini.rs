use std::fmt;

use serde::{Deserialize, Serialize};

use super::http;
use super::{CompletionRequest, CompletionResponse, ModelProvider, ProviderError};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const API_KEY_ENV: &str = "ARENA_GEMINI_API_KEY";

#[derive(Clone)]
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl GeminiProvider {
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

impl ModelProvider for GeminiProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let generation_config = match (request.temperature, request.max_tokens) {
            (None, None) => None,
            (temperature, max_tokens) => Some(GenerationConfig {
                temperature,
                max_output_tokens: max_tokens,
            }),
        };
        let body = GenerateRequest {
            contents: vec![InputContent {
                role: "user",
                parts: vec![InputPart {
                    text: request.prompt,
                }],
            }],
            generation_config,
        };

        let bytes = http::execute_json(
            self.client
                .post(generate_content_url(
                    &self.base_url,
                    &request.model.to_string(),
                ))
                .header("x-goog-api-key", &self.api_key)
                .json(&body),
            &self.api_key,
        )
        .await?;

        let parsed: GenerateResponse = serde_json::from_slice(&bytes)
            .map_err(|error| http::invalid_json(&error, &self.api_key))?;

        Ok(CompletionResponse {
            text: candidate_text(parsed)?,
        })
    }
}

fn generate_content_url(base_url: &str, model: &str) -> String {
    format!(
        "{}/models/{model}:generateContent",
        base_url.trim_end_matches('/')
    )
}

fn candidate_text(response: GenerateResponse) -> Result<String, ProviderError> {
    let Some(candidate) = response.candidates.into_iter().next() else {
        return Err(ProviderError::InvalidResponse(
            "missing completion content".into(),
        ));
    };
    let Some(parts) = candidate.content.and_then(|content| content.parts) else {
        return Err(ProviderError::InvalidResponse(
            "missing completion content".into(),
        ));
    };

    let mut text = String::new();
    let mut saw_text = false;
    for part in parts {
        if let Some(piece) = part.text {
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
struct GenerateRequest {
    contents: Vec<InputContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct InputContent {
    role: &'static str,
    parts: Vec<InputPart>,
}

#[derive(Serialize)]
struct InputPart {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<ResponseContent>,
}

#[derive(Deserialize)]
struct ResponseContent {
    parts: Option<Vec<ResponsePart>>,
}

#[derive(Deserialize)]
struct ResponsePart {
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::{header_value, request_body, request_path, start_mock};
    use crate::provider::{DEFAULT_MAX_TOKENS, ModelId};

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: ModelId::new("gemini-test"),
            prompt: "hello".into(),
            temperature: Some(0.0),
            max_tokens: Some(DEFAULT_MAX_TOKENS),
        }
    }

    #[test]
    fn generate_content_url_places_model_and_trims_slash() {
        assert_eq!(
            generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta",
                "gemini-test"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
        );
        assert_eq!(
            generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta/",
                "gemini-test"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
        );
    }

    #[tokio::test]
    async fn complete_posts_native_generate_content_request() {
        let mock = start_mock(
            200,
            "OK",
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"native-ok"}]}}]}"#,
        )
        .await;
        let base_url = format!("{}/v1beta/", mock.origin);
        let provider = GeminiProvider::new("test-key", &base_url);

        let response = provider
            .complete(sample_request())
            .await
            .expect("completion");
        let raw = mock.request.await.expect("captured request");
        mock.handle.abort();

        let request = String::from_utf8_lossy(&raw);
        assert_eq!(
            request_path(&request),
            "/v1beta/models/gemini-test:generateContent"
        );
        assert!(!request_path(&request).contains("test-key"));
        assert!(request_path(&request).starts_with('/'));
        assert!(!request.contains("?key="));
        assert_eq!(
            header_value(&request, "x-goog-api-key").as_deref(),
            Some("test-key")
        );
        assert_eq!(
            request_body(&raw),
            serde_json::json!({
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "hello"}]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 4096
                }
            })
        );
        assert_eq!(response.text, "native-ok");
    }

    #[tokio::test]
    async fn complete_rejects_malformed_response() {
        let mock = start_mock(200, "OK", "not-json").await;
        let provider = GeminiProvider::new("test-key", format!("{}/v1beta", mock.origin));

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
        let provider = GeminiProvider::new("super-secret-key", format!("{}/v1beta", mock.origin));

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
