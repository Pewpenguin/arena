use serde::Serialize;
use std::fs;
use std::path::Path;

use thiserror::Error;

use crate::compare::Comparison;
use crate::elo::ModelRating;
use crate::evaluate::EvaluatedResult;
use crate::judge::Judgment;
use crate::stats::ModelStats;

#[derive(Debug, Serialize)]
pub struct Output {
    pub results: Vec<EvaluatedResult>,
    pub comparisons: Vec<Comparison>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub judgments: Vec<Judgment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub statistics: Vec<ModelStats>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ratings: Vec<ModelRating>,
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
