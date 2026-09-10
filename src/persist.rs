use std::fs;
use std::path::Path;

use thiserror::Error;

use crate::evaluate::EvaluatedResult;

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("failed to write results file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to serialize results: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn write(path: impl AsRef<Path>, results: &[EvaluatedResult]) -> Result<(), PersistError> {
    let contents = serde_json::to_string_pretty(results)?;
    fs::write(path, contents)?;
    Ok(())
}
