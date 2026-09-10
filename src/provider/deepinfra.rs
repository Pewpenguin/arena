use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{CompletionRequest, CompletionResponse, ModelProvider, ProviderError};

const CHAT_COMPLETIONS_URL: &str = "https://api.deepinfra.com/v1/openai/chat/completions";

#[derive(Debug, Clone)]
pub struct DeepInfraProvider {
    client: Client,
    api_key: String,
}

impl DeepInfraProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
        }
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("DEEPINFRA_TOKEN").map_err(|_| {
            ProviderError::RequestFailed("DEEPINFRA_TOKEN environment variable is not set".into())
        })?;

        Ok(Self::new(api_key))
    }
}

impl ModelProvider for DeepInfraProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = DeepInfraChatRequest {
            model: request.model.to_string(),
            messages: vec![DeepInfraMessage {
                role: "user",
                content: request.prompt,
            }],
        };

        let response = self
            .client
            .post(CHAT_COMPLETIONS_URL)
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

        let parsed: DeepInfraChatResponse = serde_json::from_slice(&bytes)
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

#[derive(Serialize)]
struct DeepInfraChatRequest {
    model: String,
    messages: Vec<DeepInfraMessage>,
}

#[derive(Serialize)]
struct DeepInfraMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct DeepInfraChatResponse {
    choices: Vec<DeepInfraChoice>,
}

#[derive(Deserialize)]
struct DeepInfraChoice {
    message: DeepInfraResponseMessage,
}

#[derive(Deserialize)]
struct DeepInfraResponseMessage {
    content: Option<String>,
}
