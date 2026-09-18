use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::compare::Comparison;
use crate::evaluate::EvaluatedResult;
use crate::judge::Judgment;
use crate::provider::ModelId;
use crate::rating::ModelRating;
use crate::stats::ModelStats;

#[derive(Debug, Serialize)]
pub struct RunMetadata {
    version: &'static str,
    models: Vec<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    judge: Option<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tasks: Option<PathBuf>,
    started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bootstrap_seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bootstrap_replicates: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bootstrap_valid: Option<u32>,
}

impl RunMetadata {
    pub fn new(
        models: Vec<ModelId>,
        judge: Option<ModelId>,
        tasks: Option<PathBuf>,
        started_at: String,
    ) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            models,
            judge,
            tasks,
            started_at,
            bootstrap_seed: None,
            bootstrap_replicates: None,
            bootstrap_valid: None,
        }
    }

    pub fn with_bootstrap(mut self, seed: u64, replicates: u32, valid: u32) -> Self {
        self.bootstrap_seed = Some(seed);
        self.bootstrap_replicates = Some(replicates);
        self.bootstrap_valid = Some(valid);
        self
    }
}

pub fn utc_timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub run: RunMetadata,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_metadata_serializes_provenance_and_preserves_output_collections() {
        let run = RunMetadata::new(
            vec![ModelId::new("a"), ModelId::new("b")],
            Some(ModelId::new("judge")),
            Some(PathBuf::from("tasks.json")),
            "2026-01-02T03:04:05Z".into(),
        );
        assert_eq!(
            serde_json::to_value(&run).unwrap(),
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a", "b"],
                "judge": "judge",
                "tasks": "tasks.json",
                "started_at": "2026-01-02T03:04:05Z",
            })
        );

        let run = RunMetadata::new(
            vec![ModelId::new("a")],
            None,
            None,
            "2026-01-02T03:04:05Z".into(),
        );
        let value = serde_json::to_value(&run).unwrap();
        assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(value["models"], serde_json::json!(["a"]));
        assert!(value.get("judge").is_none());
        assert!(value.get("tasks").is_none());
        assert_eq!(value["started_at"], "2026-01-02T03:04:05Z");
        assert!(value.get("bootstrap_seed").is_none());
        assert!(value.get("bootstrap_replicates").is_none());
        assert!(value.get("bootstrap_valid").is_none());

        let output = Output {
            run,
            results: vec![],
            comparisons: vec![],
            judgments: vec![],
            statistics: vec![],
            ratings: vec![],
        };
        let value = serde_json::to_value(&output).unwrap();
        assert!(value.get("run").is_some());
        assert_eq!(value["results"], serde_json::json!([]));
        assert_eq!(value["comparisons"], serde_json::json!([]));
        assert!(value.get("judgments").is_none());
        assert!(value.get("statistics").is_none());
        assert!(value.get("ratings").is_none());
    }

    #[test]
    fn bootstrap_metadata_and_rating_bounds_serialize_only_when_present() {
        let run = RunMetadata::new(
            vec![ModelId::new("a")],
            None,
            None,
            "2026-01-02T03:04:05Z".into(),
        )
        .with_bootstrap(0, 1000, 1000);
        assert_eq!(
            serde_json::to_value(&run).unwrap(),
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a"],
                "started_at": "2026-01-02T03:04:05Z",
                "bootstrap_seed": 0,
                "bootstrap_replicates": 1000,
                "bootstrap_valid": 1000,
            })
        );

        let with_bounds = ModelRating {
            model: ModelId::new("a"),
            rating: Some(1500.0),
            rating_lower: Some(1400.0),
            rating_upper: Some(1600.0),
        };
        assert_eq!(
            serde_json::to_value(&with_bounds).unwrap(),
            serde_json::json!({
                "model": "a",
                "rating": 1500.0,
                "rating_lower": 1400.0,
                "rating_upper": 1600.0,
            })
        );

        let without_bounds = ModelRating {
            model: ModelId::new("a"),
            rating: Some(1500.0),
            rating_lower: None,
            rating_upper: None,
        };
        let value = serde_json::to_value(&without_bounds).unwrap();
        assert_eq!(value["model"], "a");
        assert_eq!(value["rating"], 1500.0);
        assert!(value.get("rating_lower").is_none());
        assert!(value.get("rating_upper").is_none());
    }
}
