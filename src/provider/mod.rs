mod anthropic;
mod gemini;
mod http;
mod openai_compatible;
mod openrouter;

#[cfg(test)]
mod test_support;

pub use anthropic::AnthropicProvider;
pub use gemini::GeminiProvider;
pub use openai_compatible::OpenAICompatibleProvider;
pub use openrouter::OpenRouterProvider;

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) const PROVIDER_CONCURRENCY: usize = 8;
pub(crate) const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
pub(crate) const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Requested output-token budget for candidate and judge completions.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionRequest {
    pub model: ModelId,
    pub prompt: String,
    /// Sampling temperature. `None` omits the field so the provider default applies.
    pub temperature: Option<f64>,
    /// Maximum number of output tokens. `None` omits the field so the provider default applies.
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProviderError {
    #[error("provider request failed: {0}")]
    RequestFailed(String),
    #[error("invalid provider response: {0}")]
    InvalidResponse(String),
}

pub trait ModelProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, ProviderError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_carries_max_tokens() {
        let request = CompletionRequest {
            model: ModelId::new("m"),
            prompt: "p".into(),
            temperature: None,
            max_tokens: Some(DEFAULT_MAX_TOKENS),
        };
        assert_eq!(request.max_tokens, Some(4096));
        assert_eq!(request.temperature, None);
    }
}
