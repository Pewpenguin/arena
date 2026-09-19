use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::compare::Comparison;
use crate::evaluate::EvaluatedResult;
use crate::judge::{Judgment, JudgmentFailure};
use crate::provider::ModelId;
use crate::rating::ModelRating;
use crate::stats::ModelStats;
use crate::task::Task;

#[derive(Debug, Serialize)]
pub struct RunMetadata {
    version: &'static str,
    models: Vec<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    judge: Option<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tasks: Option<PathBuf>,
    base_url: String,
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
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            models,
            judge,
            tasks,
            base_url: base_url.into(),
            started_at,
            bootstrap_seed: None,
            bootstrap_replicates: None,
            bootstrap_valid: None,
        }
    }

    pub fn with_bootstrap(mut self, seed: u64, replicates: u32, valid: Option<u32>) -> Self {
        self.bootstrap_seed = Some(seed);
        self.bootstrap_replicates = Some(replicates);
        self.bootstrap_valid = valid;
        self
    }
}

pub fn utc_timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub run: RunMetadata,
    pub tasks: Vec<Task>,
    pub results: Vec<EvaluatedResult>,
    pub comparisons: Vec<Comparison>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub judgments: Vec<Judgment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub judgment_failures: Vec<JudgmentFailure>,
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
    use crate::judge::{JudgeOrientation, JudgmentFailureKind, OrientationFailure};
    use crate::rating::UnavailableReason;
    use crate::task::{Task, TaskEvaluation};

    fn run_meta(
        models: Vec<ModelId>,
        judge: Option<ModelId>,
        tasks: Option<PathBuf>,
    ) -> RunMetadata {
        RunMetadata::new(
            models,
            judge,
            tasks,
            "2026-01-02T03:04:05Z".into(),
            "https://example.test/v1",
        )
    }

    #[test]
    fn run_metadata_serializes_provenance_and_preserves_output_collections() {
        let run = run_meta(
            vec![ModelId::new("a"), ModelId::new("b")],
            Some(ModelId::new("judge")),
            Some(PathBuf::from("tasks.json")),
        );
        let value = serde_json::to_value(&run).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a", "b"],
                "judge": "judge",
                "tasks": "tasks.json",
                "base_url": "https://example.test/v1",
                "started_at": "2026-01-02T03:04:05Z",
            })
        );
        assert!(value.get("api_key").is_none());
        assert!(value.get("authorization").is_none());

        let run = run_meta(vec![ModelId::new("a")], None, None);
        let value = serde_json::to_value(&run).unwrap();
        assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(value["models"], serde_json::json!(["a"]));
        assert_eq!(value["base_url"], "https://example.test/v1");
        assert!(value.get("judge").is_none());
        assert!(value.get("tasks").is_none());
        assert_eq!(value["started_at"], "2026-01-02T03:04:05Z");
        assert!(value.get("bootstrap_seed").is_none());
        assert!(value.get("bootstrap_replicates").is_none());
        assert!(value.get("bootstrap_valid").is_none());

        let output = Output {
            run: run_meta(
                vec![ModelId::new("a")],
                None,
                Some(PathBuf::from("tasks.json")),
            ),
            tasks: vec![
                Task {
                    id: "t1".into(),
                    prompt: "Say hello".into(),
                    evaluation: None,
                },
                Task {
                    id: "t2".into(),
                    prompt: "Capital?".into(),
                    evaluation: Some(TaskEvaluation::Exact {
                        expected: "Paris".into(),
                    }),
                },
            ],
            results: vec![],
            comparisons: vec![],
            judgments: vec![],
            judgment_failures: vec![],
            statistics: vec![],
            ratings: vec![],
        };
        let value = serde_json::to_value(&output).unwrap();
        assert!(value.get("run").is_some());
        assert_eq!(value["run"]["tasks"], "tasks.json");
        assert_eq!(
            value["tasks"],
            serde_json::json!([
                {"id": "t1", "prompt": "Say hello"},
                {
                    "id": "t2",
                    "prompt": "Capital?",
                    "evaluation": {"type": "exact", "expected": "Paris"}
                }
            ])
        );
        assert_eq!(value["results"], serde_json::json!([]));
        assert_eq!(value["comparisons"], serde_json::json!([]));
        assert!(value.get("judgments").is_none());
        assert!(value.get("judgment_failures").is_none());
        assert!(value.get("statistics").is_none());
        assert!(value.get("ratings").is_none());
    }

    #[test]
    fn bootstrap_metadata_and_rating_bounds_serialize_only_when_present() {
        let configured =
            run_meta(vec![ModelId::new("a")], None, None).with_bootstrap(0, 1000, None);
        assert_eq!(
            serde_json::to_value(&configured).unwrap(),
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a"],
                "base_url": "https://example.test/v1",
                "started_at": "2026-01-02T03:04:05Z",
                "bootstrap_seed": 0,
                "bootstrap_replicates": 1000,
            })
        );
        assert!(
            serde_json::to_value(&configured)
                .unwrap()
                .get("bootstrap_valid")
                .is_none()
        );

        let run = run_meta(vec![ModelId::new("a")], None, None).with_bootstrap(0, 1000, Some(1000));
        assert_eq!(
            serde_json::to_value(&run).unwrap(),
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a"],
                "base_url": "https://example.test/v1",
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
            unavailable: None,
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
            unavailable: None,
        };
        let value = serde_json::to_value(&without_bounds).unwrap();
        assert_eq!(value["model"], "a");
        assert_eq!(value["rating"], 1500.0);
        assert!(value.get("rating_lower").is_none());
        assert!(value.get("rating_upper").is_none());
        assert!(value.get("unavailable").is_none());

        for reason in [
            UnavailableReason::NoComparisons,
            UnavailableReason::Disconnected,
            UnavailableReason::Separated,
            UnavailableReason::Nonfinite,
        ] {
            let value = serde_json::to_value(&ModelRating {
                model: ModelId::new("a"),
                rating: None,
                rating_lower: None,
                rating_upper: None,
                unavailable: Some(reason),
            })
            .unwrap();
            assert_eq!(value["rating"], serde_json::Value::Null);
            assert_eq!(
                value["unavailable"],
                match reason {
                    UnavailableReason::NoComparisons => "no_comparisons",
                    UnavailableReason::Disconnected => "disconnected",
                    UnavailableReason::Separated => "separated",
                    UnavailableReason::Nonfinite => "nonfinite",
                }
            );
        }
    }

    #[test]
    fn judgment_failures_serialize_and_are_omitted_when_empty() {
        let failure = JudgmentFailure {
            task_id: "t1".into(),
            model_a: ModelId::new("a"),
            model_b: ModelId::new("b"),
            judge_model: ModelId::new("judge"),
            orientations: vec![
                OrientationFailure {
                    orientation: JudgeOrientation::Ab,
                    kind: JudgmentFailureKind::InvalidJson,
                    error: "no valid judgment JSON found".into(),
                    attempts: 3,
                },
                OrientationFailure {
                    orientation: JudgeOrientation::Ba,
                    kind: JudgmentFailureKind::Provider,
                    error: "HTTP 500: upstream".into(),
                    attempts: 3,
                },
            ],
        };
        assert_eq!(
            serde_json::to_value(&failure).unwrap(),
            serde_json::json!({
                "task_id": "t1",
                "model_a": "a",
                "model_b": "b",
                "judge_model": "judge",
                "orientations": [
                    {
                        "orientation": "ab",
                        "kind": "invalid_json",
                        "error": "no valid judgment JSON found",
                        "attempts": 3
                    },
                    {
                        "orientation": "ba",
                        "kind": "provider",
                        "error": "HTTP 500: upstream",
                        "attempts": 3
                    }
                ]
            })
        );

        let output = Output {
            run: run_meta(vec![ModelId::new("a")], None, None),
            tasks: vec![],
            results: vec![],
            comparisons: vec![],
            judgments: vec![],
            judgment_failures: vec![],
            statistics: vec![],
            ratings: vec![],
        };
        let value = serde_json::to_value(&output).unwrap();
        assert!(value.get("judgment_failures").is_none());
    }
}
