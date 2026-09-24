use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::bootstrap::{BootstrapMeta, BootstrapUnavailable};
use crate::compare::Comparison;
use crate::evaluate::EvaluatedResult;
use crate::judge::{Judgment, JudgmentFailure};
use crate::provider::ModelId;
use crate::rating::ModelRating;
use crate::stats::{ModelStats, PairAgreement};
use crate::task::Task;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeDecoding {
    pub temperature: f64,
    pub max_tokens: u32,
}

impl JudgeDecoding {
    pub fn arena_default() -> Self {
        Self {
            temperature: crate::judge::JUDGE_TEMPERATURE,
            max_tokens: crate::provider::DEFAULT_MAX_TOKENS,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub(crate) version: String,
    pub(crate) models: Vec<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) judge: Option<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tasks: Option<PathBuf>,
    pub(crate) base_url: String,
    pub(crate) started_at: String,
    pub(crate) provider_concurrency: usize,
    pub(crate) request_timeout_secs: u64,
    pub(crate) connect_timeout_secs: u64,
    pub(crate) candidate_attempts: u32,
    pub(crate) judge_attempts: u32,
    pub(crate) candidate_max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) complete: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expected_pairs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) resolved_pairs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) failed_pairs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) orientation_agreement: Option<PairAgreement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) judge_decoding: Option<JudgeDecoding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bootstrap_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bootstrap_replicates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bootstrap_valid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bootstrap_clusters: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bootstrap_ran: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bootstrap_unavailable: Option<BootstrapUnavailable>,
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
            version: env!("CARGO_PKG_VERSION").to_string(),
            models,
            judge,
            tasks,
            base_url: base_url.into(),
            started_at,
            provider_concurrency: crate::provider::PROVIDER_CONCURRENCY,
            request_timeout_secs: crate::provider::REQUEST_TIMEOUT.as_secs(),
            connect_timeout_secs: crate::provider::CONNECT_TIMEOUT.as_secs(),
            candidate_attempts: crate::retry::ATTEMPTS,
            judge_attempts: crate::retry::ATTEMPTS,
            candidate_max_tokens: crate::provider::DEFAULT_MAX_TOKENS,
            complete: None,
            expected_pairs: None,
            resolved_pairs: None,
            failed_pairs: None,
            orientation_agreement: None,
            judge_decoding: None,
            bootstrap_seed: None,
            bootstrap_replicates: None,
            bootstrap_valid: None,
            bootstrap_clusters: None,
            bootstrap_ran: None,
            bootstrap_unavailable: None,
        }
    }

    pub fn with_bootstrap(mut self, meta: &BootstrapMeta) -> Self {
        self.bootstrap_seed = Some(meta.seed);
        self.bootstrap_replicates = Some(meta.replicates);
        self.bootstrap_valid = meta.valid;
        self.bootstrap_clusters = Some(meta.clusters);
        self.bootstrap_ran = Some(meta.ran);
        self.bootstrap_unavailable = meta.unavailable;
        self
    }

    pub fn with_judge_coverage(
        mut self,
        expected_pairs: usize,
        resolved_pairs: usize,
        failed_pairs: usize,
    ) -> Self {
        self.complete = Some(resolved_pairs == expected_pairs && failed_pairs == 0);
        self.expected_pairs = Some(expected_pairs);
        self.resolved_pairs = Some(resolved_pairs);
        self.failed_pairs = Some(failed_pairs);
        self
    }

    pub fn with_orientation_agreement(mut self, agreement: PairAgreement) -> Self {
        self.orientation_agreement = Some(agreement);
        self
    }

    pub fn with_judge_decoding(mut self, decoding: JudgeDecoding) -> Self {
        self.judge_decoding = Some(decoding);
        self
    }
}

pub fn utc_timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    pub run: RunMetadata,
    pub tasks: Vec<Task>,
    pub results: Vec<EvaluatedResult>,
    pub comparisons: Vec<Comparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judgments: Option<Vec<Judgment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judgment_failures: Option<Vec<JudgmentFailure>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub statistics: Option<Vec<ModelStats>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratings: Option<Vec<ModelRating>>,
}

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("failed to write results file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to serialize results: {0}")]
    Serialize(serde_json::Error),
    #[error("failed to parse results file: {0}")]
    Parse(serde_json::Error),
}

pub fn to_pretty_json(output: &Output) -> Result<String, PersistError> {
    serde_json::to_string_pretty(output).map_err(PersistError::Serialize)
}

pub fn write(path: impl AsRef<Path>, output: &Output) -> Result<(), PersistError> {
    fs::write(path, to_pretty_json(output)?)?;
    Ok(())
}

pub fn read(path: impl AsRef<Path>) -> Result<Output, PersistError> {
    let contents = fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(PersistError::Parse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::{BootstrapMeta, BootstrapUnavailable};
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
                "provider_concurrency": 8,
                "request_timeout_secs": 120,
                "connect_timeout_secs": 10,
                "candidate_attempts": 3,
                "judge_attempts": 3,
                "candidate_max_tokens": 4096,
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
        assert_eq!(value["provider_concurrency"], 8);
        assert_eq!(value["request_timeout_secs"], 120);
        assert_eq!(value["connect_timeout_secs"], 10);
        assert_eq!(value["candidate_attempts"], 3);
        assert_eq!(value["judge_attempts"], 3);
        assert_eq!(value["candidate_max_tokens"], 4096);
        assert!(value.get("bootstrap_seed").is_none());
        assert!(value.get("bootstrap_replicates").is_none());
        assert!(value.get("bootstrap_valid").is_none());
        assert!(value.get("bootstrap_clusters").is_none());
        assert!(value.get("bootstrap_ran").is_none());
        assert!(value.get("bootstrap_unavailable").is_none());
        assert!(value.get("complete").is_none());
        assert!(value.get("expected_pairs").is_none());
        assert!(value.get("resolved_pairs").is_none());
        assert!(value.get("failed_pairs").is_none());
        assert!(value.get("orientation_agreement").is_none());
        assert!(value.get("judge_decoding").is_none());

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
            judgments: None,
            judgment_failures: None,
            statistics: None,
            ratings: None,
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
        let parsed: Output = serde_json::from_str(&to_pretty_json(&output).unwrap()).unwrap();
        assert!(parsed.judgments.is_none());
        assert_eq!(parsed.tasks.len(), 2);
        assert_eq!(parsed.run.models, vec![ModelId::new("a")]);
    }

    #[test]
    fn bootstrap_metadata_and_rating_bounds_serialize_only_when_present() {
        let skipped =
            run_meta(vec![ModelId::new("a")], None, None).with_bootstrap(&BootstrapMeta {
                seed: 0,
                replicates: 1000,
                valid: None,
                clusters: 1,
                ran: false,
                unavailable: Some(BootstrapUnavailable::TooFewTasks),
            });
        assert_eq!(
            serde_json::to_value(&skipped).unwrap(),
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a"],
                "base_url": "https://example.test/v1",
                "started_at": "2026-01-02T03:04:05Z",
                "provider_concurrency": 8,
                "request_timeout_secs": 120,
                "connect_timeout_secs": 10,
                "candidate_attempts": 3,
                "judge_attempts": 3,
                "candidate_max_tokens": 4096,
                "bootstrap_seed": 0,
                "bootstrap_replicates": 1000,
                "bootstrap_clusters": 1,
                "bootstrap_ran": false,
                "bootstrap_unavailable": "too_few_tasks",
            })
        );
        assert!(
            serde_json::to_value(&skipped)
                .unwrap()
                .get("bootstrap_valid")
                .is_none()
        );

        let run = run_meta(vec![ModelId::new("a")], None, None).with_bootstrap(&BootstrapMeta {
            seed: 0,
            replicates: 1000,
            valid: Some(1000),
            clusters: 2,
            ran: true,
            unavailable: None,
        });
        assert_eq!(
            serde_json::to_value(&run).unwrap(),
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "models": ["a"],
                "base_url": "https://example.test/v1",
                "started_at": "2026-01-02T03:04:05Z",
                "provider_concurrency": 8,
                "request_timeout_secs": 120,
                "connect_timeout_secs": 10,
                "candidate_attempts": 3,
                "judge_attempts": 3,
                "candidate_max_tokens": 4096,
                "bootstrap_seed": 0,
                "bootstrap_replicates": 1000,
                "bootstrap_valid": 1000,
                "bootstrap_clusters": 2,
                "bootstrap_ran": true,
            })
        );
        assert!(
            serde_json::to_value(&run)
                .unwrap()
                .get("bootstrap_unavailable")
                .is_none()
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
            UnavailableReason::Nonconvergence,
            UnavailableReason::NonfiniteResult,
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
                    UnavailableReason::Nonconvergence => "nonconvergence",
                    UnavailableReason::NonfiniteResult => "nonfinite_result",
                }
            );
        }
    }

    #[test]
    fn judge_coverage_and_decoding_serialize_only_when_set() {
        let incomplete = run_meta(
            vec![ModelId::new("a"), ModelId::new("b")],
            Some(ModelId::new("judge")),
            Some(PathBuf::from("tasks.json")),
        )
        .with_judge_coverage(1, 0, 1)
        .with_orientation_agreement(crate::stats::pair_agreement(&[]))
        .with_judge_decoding(JudgeDecoding::arena_default());
        let value = serde_json::to_value(&incomplete).unwrap();
        assert_eq!(value["complete"], false);
        assert_eq!(value["expected_pairs"], 1);
        assert_eq!(value["resolved_pairs"], 0);
        assert_eq!(value["failed_pairs"], 1);
        assert_eq!(
            value["orientation_agreement"],
            serde_json::json!({
                "resolved_pairs": 0,
                "orientation_agreeing_pairs": 0,
                "orientation_disagreeing_pairs": 0,
                "agreement_rate": 0.0
            })
        );
        assert_eq!(
            value["judge_decoding"],
            serde_json::json!({ "temperature": 0.0, "max_tokens": 4096 })
        );
        assert_eq!(value["candidate_max_tokens"], 4096);

        let complete = run_meta(vec![ModelId::new("a"), ModelId::new("b")], None, None)
            .with_judge_coverage(3, 3, 0);
        let value = serde_json::to_value(&complete).unwrap();
        assert_eq!(value["complete"], true);
        assert_eq!(value["expected_pairs"], 3);
        assert_eq!(value["resolved_pairs"], 3);
        assert_eq!(value["failed_pairs"], 0);
        assert!(value.get("judge_decoding").is_none());
    }

    #[test]
    fn judgment_failures_serialize_and_are_omitted_without_a_judge() {
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
            raw_ab: None,
            reason_ab: None,
            raw_ba: None,
            reason_ba: None,
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
            judgments: None,
            judgment_failures: None,
            statistics: None,
            ratings: None,
        };
        let value = serde_json::to_value(&output).unwrap();
        assert!(value.get("judgments").is_none());
        assert!(value.get("judgment_failures").is_none());
        assert!(value.get("statistics").is_none());
        assert!(value.get("ratings").is_none());
        assert!(value["run"].get("complete").is_none());
        assert!(value["run"].get("orientation_agreement").is_none());
    }

    #[test]
    fn judgment_failure_round_trips_kept_orientation_and_loads_older_files() {
        let legacy = r#"{"task_id":"t1","model_a":"a","model_b":"b","judge_model":"judge","orientations":[]}"#;
        let loaded: JudgmentFailure = serde_json::from_str(legacy).unwrap();
        assert!(loaded.raw_ab.is_none());
        assert!(loaded.reason_ab.is_none());
        assert!(loaded.raw_ba.is_none());
        assert!(loaded.reason_ba.is_none());

        let mut kept = loaded;
        kept.raw_ab = Some("ab raw".into());
        kept.reason_ab = Some("ab reason".into());
        let value = serde_json::to_value(&kept).unwrap();
        assert_eq!(value["raw_ab"], "ab raw");
        assert_eq!(value["reason_ab"], "ab reason");
        assert!(value.get("raw_ba").is_none());
        assert!(value.get("reason_ba").is_none());
        let restored: JudgmentFailure = serde_json::from_value(value).unwrap();
        assert_eq!(restored, kept);
    }

    #[test]
    fn judge_run_empty_collections_serialize_as_empty_arrays() {
        let run = run_meta(vec![ModelId::new("a")], Some(ModelId::new("judge")), None)
            .with_judge_coverage(0, 0, 0)
            .with_orientation_agreement(crate::stats::pair_agreement(&[]));
        let output = Output {
            run,
            tasks: vec![],
            results: vec![],
            comparisons: vec![],
            judgments: Some(vec![]),
            judgment_failures: Some(vec![]),
            statistics: Some(vec![]),
            ratings: Some(vec![]),
        };
        let value = serde_json::to_value(&output).unwrap();
        assert_eq!(value["judgments"], serde_json::json!([]));
        assert_eq!(value["judgment_failures"], serde_json::json!([]));
        assert_eq!(value["statistics"], serde_json::json!([]));
        assert_eq!(value["ratings"], serde_json::json!([]));
        assert_eq!(value["run"]["complete"], true);
        assert_eq!(value["run"]["expected_pairs"], 0);
        assert_eq!(value["run"]["resolved_pairs"], 0);
        assert_eq!(value["run"]["failed_pairs"], 0);
        assert_eq!(
            value["run"]["orientation_agreement"],
            serde_json::json!({
                "resolved_pairs": 0,
                "orientation_agreeing_pairs": 0,
                "orientation_disagreeing_pairs": 0,
                "agreement_rate": 0.0
            })
        );
    }

    #[test]
    fn current_experiment_fixture_loads_and_reports() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/experiment.json");
        let output = read(&path).expect("fixture");
        let report = crate::report::from_output(&output);

        assert_eq!(output.run.version, "1.2.0");
        assert_eq!(output.run.complete, Some(false));
        assert_eq!(output.run.expected_pairs, Some(2));
        assert_eq!(output.run.resolved_pairs, Some(1));
        assert_eq!(output.run.failed_pairs, Some(1));
        assert_eq!(output.tasks.len(), 2);
        assert_eq!(output.results.len(), 4);
        assert_eq!(output.comparisons.len(), 1);
        assert_eq!(output.judgments.as_ref().map(Vec::len), Some(1));
        assert_eq!(output.judgment_failures.as_ref().map(Vec::len), Some(1));
        assert_eq!(output.statistics.as_ref().map(Vec::len), Some(2));
        assert_eq!(output.ratings.as_ref().map(Vec::len), Some(2));
        assert_eq!(
            output.run.bootstrap_unavailable,
            Some(crate::bootstrap::BootstrapUnavailable::OriginalUnrated)
        );
        assert!(output.run.bootstrap_valid.is_none());

        assert!(report.summary.judge_used);
        assert_eq!(report.summary.complete, Some(false));
        assert_eq!(report.summary.task_count, 2);
        assert_eq!(report.summary.candidate_count, 2);
        assert_eq!(report.pairs.len(), 1);
        assert_eq!(report.failed_pairs.len(), 1);
        assert_eq!(report.results.len(), 4);
        assert_eq!(report.models.len(), 2);
        assert!(report.models.iter().all(|model| model.rating.is_none()));
        let bootstrap = report.bootstrap.as_ref().expect("bootstrap");
        assert_eq!(bootstrap.seed, 0);
        assert_eq!(bootstrap.replicates, 1000);
        assert!(!bootstrap.ran);
        assert!(bootstrap.valid.is_none());
        assert_eq!(
            bootstrap.unavailable,
            Some(crate::bootstrap::BootstrapUnavailable::OriginalUnrated)
        );

        let html = crate::html::render(&report);
        assert!(html.contains("INCOMPLETE"));
        assert!(html.contains("model-a"));
        assert!(html.contains("separated"));
    }
}
