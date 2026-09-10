use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evaluate::EvaluatedResult;
use crate::provider::{CompletionRequest, ModelId, ModelProvider, ProviderError};
use crate::task::Task;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JudgeDecision {
    A,
    B,
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Judgment {
    pub task_id: String,
    pub model_a: ModelId,
    pub model_b: ModelId,
    pub judge_model: ModelId,
    pub winner: JudgeDecision,
}

#[derive(Debug, Error)]
pub enum JudgeError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("invalid judge response JSON: {0}")]
    InvalidJson(String),
    #[error("cannot judge results from different tasks")]
    DifferentTasks,
}

pub fn build_judge_prompt(task: &Task, response_a: &str, response_b: &str) -> String {
    format!(
        "You are comparing two model responses to the same task.\n\
         \n\
         Task:\n\
         {prompt}\n\
         \n\
         Response A:\n\
         {response_a}\n\
         \n\
         Response B:\n\
         {response_b}\n\
         \n\
         Which response better satisfies the user's task?\n\
         \n\
         Reply with only valid JSON in exactly one of these forms:\n\
         {{\"winner\":\"a\"}}\n\
         {{\"winner\":\"b\"}}\n\
         {{\"winner\":\"draw\"}}",
        prompt = task.prompt,
        response_a = response_a,
        response_b = response_b,
    )
}

pub fn parse_decision(text: &str) -> Result<JudgeDecision, JudgeError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Payload {
        winner: JudgeDecision,
    }

    let trimmed = text.trim();
    let mut search_from = 0;

    while let Some(offset) = trimmed[search_from..].find('{') {
        let start = search_from + offset;
        let mut deserializer = serde_json::Deserializer::from_str(&trimmed[start..]);
        match Payload::deserialize(&mut deserializer) {
            Ok(payload) => return Ok(payload.winner),
            Err(_) => search_from = start + 1,
        }
    }

    let preview: String = trimmed.chars().take(500).collect();
    Err(JudgeError::InvalidJson(format!(
        "no valid judgment JSON found; response: {preview}"
    )))
}

pub async fn judge_pair(
    provider: &impl ModelProvider,
    judge_model: ModelId,
    task: &Task,
    result_a: &EvaluatedResult,
    result_b: &EvaluatedResult,
) -> Result<Judgment, JudgeError> {
    if result_a.task_id != result_b.task_id {
        return Err(JudgeError::DifferentTasks);
    }

    let prompt = build_judge_prompt(task, &result_a.response.text, &result_b.response.text);
    let response = provider
        .complete(CompletionRequest {
            model: judge_model.clone(),
            prompt,
        })
        .await?;

    let winner = parse_decision(&response.text)?;

    Ok(Judgment {
        task_id: result_a.task_id.clone(),
        model_a: result_a.model.clone(),
        model_b: result_b.model.clone(),
        judge_model,
        winner,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_winner_a() {
        assert_eq!(
            parse_decision(r#"{"winner":"a"}"#).unwrap(),
            JudgeDecision::A
        );
    }

    #[test]
    fn parses_winner_b() {
        assert_eq!(
            parse_decision(r#"{"winner":"b"}"#).unwrap(),
            JudgeDecision::B
        );
    }

    #[test]
    fn parses_draw() {
        assert_eq!(
            parse_decision(r#"{"winner":"draw"}"#).unwrap(),
            JudgeDecision::Draw
        );
    }

    #[test]
    fn parses_json_embedded_in_surrounding_text() {
        let text = "<think>\nreasoning about the answers\n</think>\n{\"winner\":\"b\"}\n";
        assert_eq!(parse_decision(text).unwrap(), JudgeDecision::B);
    }

    #[test]
    fn rejects_invalid_winner() {
        let error = parse_decision(r#"{"winner":"c"}"#).unwrap_err();
        assert!(matches!(error, JudgeError::InvalidJson(_)));
    }

    #[test]
    fn rejects_missing_winner() {
        let error = parse_decision(r#"{"result":"a"}"#).unwrap_err();
        assert!(matches!(error, JudgeError::InvalidJson(_)));
    }

    #[test]
    fn rejects_response_without_judgment_json() {
        let error = parse_decision("<think>no judgment object here</think>").unwrap_err();
        match error {
            JudgeError::InvalidJson(message) => {
                assert!(message.contains("no valid judgment JSON found"));
                assert!(message.contains("response:"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
