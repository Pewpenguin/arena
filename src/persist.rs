use std::fs;
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use crate::compare::Comparison;
use crate::evaluate::EvaluatedResult;

#[derive(Debug, Serialize)]
pub struct Output {
    pub results: Vec<EvaluatedResult>,
    pub comparisons: Vec<Comparison>,
}

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("failed to write results file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to serialize results: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn write(path: impl AsRef<Path>, output: &Output) -> Result<(), PersistError> {
    let contents = serde_json::to_string_pretty(output)?;
    fs::write(path, contents)?;
    Ok(())
}
