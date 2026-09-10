use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub prompt: String,
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
    fn parses_task_list() {
        let tasks = parse(
            r#"[
                {"id": "t1", "prompt": "Say hello"},
                {"id": "t2", "prompt": "Say goodbye"}
            ]"#,
        )
        .unwrap();

        assert_eq!(
            tasks,
            vec![
                Task {
                    id: "t1".into(),
                    prompt: "Say hello".into(),
                },
                Task {
                    id: "t2".into(),
                    prompt: "Say goodbye".into(),
                },
            ]
        );
    }

    #[test]
    fn rejects_malformed_json() {
        let error = parse(r#"{"id": "t1"}"#).unwrap_err();
        assert!(matches!(error, TaskError::Parse(_)));
    }
}
