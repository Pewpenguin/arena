mod openai_compatible;

pub use openai_compatible::OpenAICompatibleProvider;

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) const PROVIDER_CONCURRENCY: usize = 8;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionRequest {
    pub model: ModelId,
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
