use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<TaskEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TaskEvaluation {
    #[serde(rename = "exact")]
    Exact { expected: String },
}

#[derive(Debug, Error)]
pub enum TaskError {
    #[error("failed to read tasks file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse tasks file: {0}")]
    Parse(#[from] serde_json::Error),
}

pub fn load(path: impl AsRef<Path>) -> Result<Vec<Task>, TaskError> {
    let contents = fs::read_to_string(path)?;
    parse(&contents)
}

fn parse(contents: &str) -> Result<Vec<Task>, TaskError> {
    Ok(serde_json::from_str(contents)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_task_without_evaluation() {
        let tasks = parse(r#"[{"id": "t1", "prompt": "Say hello"}]"#).unwrap();

        assert_eq!(
            tasks,
            vec![Task {
                id: "t1".into(),
                prompt: "Say hello".into(),
                evaluation: None,
            }]
        );
    }

    #[test]
    fn parses_task_with_exact_evaluation() {
        let tasks = parse(
            r#"[
                {
                    "id": "t2",
                    "prompt": "What is the capital of France? Answer with only the city name.",
                    "evaluation": {
                        "type": "exact",
                        "expected": "Paris"
                    }
                }
            ]"#,
        )
        .unwrap();

        assert_eq!(
            tasks,
            vec![Task {
                id: "t2".into(),
                prompt: "What is the capital of France? Answer with only the city name.".into(),
                evaluation: Some(TaskEvaluation::Exact {
                    expected: "Paris".into(),
                }),
            }]
        );
    }

    #[test]
    fn rejects_malformed_json() {
        let error = parse(r#"{"id": "t1"}"#).unwrap_err();
        assert!(matches!(error, TaskError::Parse(_)));
    }
}
