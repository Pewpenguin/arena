use std::path::Path;

use crate::bootstrap::BootstrapUnavailable;
use crate::evaluate::EvaluatedResult;
use crate::judge::{JudgeDecision, Judgment, JudgmentFailure, OrientationFailure};
use crate::persist::{JudgeDecoding, Output};
use crate::provider::ModelId;
use crate::rating::{ModelRating, UnavailableReason};
use crate::stats::ModelStats;

#[derive(Debug, Clone, PartialEq)]
pub struct Report<'a> {
    pub summary: ReportSummary,
    pub config: RunConfig<'a>,
    pub models: Vec<ModelReport<'a>>,
    pub results: Vec<CandidateRow<'a>>,
    pub pairs: Vec<PairRow<'a>>,
    pub failed_pairs: Vec<FailedPairRow<'a>>,
    pub bootstrap: Option<BootstrapReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReportSummary {
    pub judge_used: bool,
    pub complete: Option<bool>,
    pub task_count: usize,
    pub candidate_count: usize,
    pub expected_pairs: Option<usize>,
    pub resolved_pairs: Option<usize>,
    pub failed_pairs: Option<usize>,
    pub orientation_agreeing_pairs: Option<usize>,
    pub orientation_disagreeing_pairs: Option<usize>,
    pub agreement_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunConfig<'a> {
    pub models: &'a [ModelId],
    pub judge: Option<&'a ModelId>,
    pub tasks: Option<&'a Path>,
    pub base_url: &'a str,
    pub started_at: &'a str,
    pub provider_concurrency: usize,
    pub request_timeout_secs: u64,
    pub connect_timeout_secs: u64,
    pub candidate_attempts: u32,
    pub judge_attempts: u32,
    pub candidate_max_tokens: u32,
    pub judge_decoding: Option<&'a JudgeDecoding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelReport<'a> {
    pub model: &'a ModelId,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub total: u32,
    pub rating: Option<f64>,
    pub rating_lower: Option<f64>,
    pub rating_upper: Option<f64>,
    pub unavailable: Option<UnavailableReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRow<'a> {
    pub task_id: &'a str,
    pub model: &'a ModelId,
    pub response: &'a str,
    pub score: Option<f64>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairRow<'a> {
    pub task_id: &'a str,
    pub model_a: &'a ModelId,
    pub model_b: &'a ModelId,
    pub winner: JudgeDecision,
    pub agreement: bool,
    pub orientation_ab: Option<JudgeDecision>,
    pub orientation_ba: Option<JudgeDecision>,
    pub reason_ab: Option<&'a str>,
    pub reason_ba: Option<&'a str>,
    pub duration_ms: u64,
    pub raw_ab: Option<&'a str>,
    pub raw_ba: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FailedPairRow<'a> {
    pub task_id: &'a str,
    pub model_a: &'a ModelId,
    pub model_b: &'a ModelId,
    pub orientations: &'a [OrientationFailure],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapReport {
    pub seed: u64,
    pub replicates: u32,
    pub clusters: u32,
    pub ran: bool,
    pub valid: Option<u32>,
    pub unavailable: Option<BootstrapUnavailable>,
    pub bounds_present: bool,
}

pub fn from_output(output: &Output) -> Report<'_> {
    let judge_used = output.run.judge.is_some();
    let agreement = output.run.orientation_agreement.as_ref();
    Report {
        summary: ReportSummary {
            judge_used,
            complete: output.run.complete,
            task_count: output.tasks.len(),
            candidate_count: output.run.models.len(),
            expected_pairs: output.run.expected_pairs,
            resolved_pairs: output.run.resolved_pairs,
            failed_pairs: output.run.failed_pairs,
            orientation_agreeing_pairs: agreement.map(|value| value.orientation_agreeing_pairs),
            orientation_disagreeing_pairs: agreement
                .map(|value| value.orientation_disagreeing_pairs),
            agreement_rate: agreement.map(|value| value.agreement_rate),
        },
        config: RunConfig {
            models: &output.run.models,
            judge: output.run.judge.as_ref(),
            tasks: output.run.tasks.as_deref(),
            base_url: &output.run.base_url,
            started_at: &output.run.started_at,
            provider_concurrency: output.run.provider_concurrency,
            request_timeout_secs: output.run.request_timeout_secs,
            connect_timeout_secs: output.run.connect_timeout_secs,
            candidate_attempts: output.run.candidate_attempts,
            judge_attempts: output.run.judge_attempts,
            candidate_max_tokens: output.run.candidate_max_tokens,
            judge_decoding: output.run.judge_decoding.as_ref(),
        },
        models: output
            .run
            .models
            .iter()
            .map(|model| {
                model_report(
                    model,
                    output.statistics.as_deref(),
                    output.ratings.as_deref(),
                )
            })
            .collect(),
        results: output.results.iter().map(candidate_row).collect(),
        pairs: output
            .judgments
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(pair_row)
            .collect(),
        failed_pairs: output
            .judgment_failures
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(failed_pair_row)
            .collect(),
        bootstrap: bootstrap_report(output),
    }
}

fn model_report<'a>(
    model: &'a ModelId,
    statistics: Option<&'a [ModelStats]>,
    ratings: Option<&'a [ModelRating]>,
) -> ModelReport<'a> {
    let stats = statistics.and_then(|stats| stats.iter().find(|item| item.model == *model));
    let rating = ratings.and_then(|ratings| ratings.iter().find(|item| item.model == *model));
    ModelReport {
        model,
        wins: stats.map(|item| item.wins).unwrap_or(0),
        losses: stats.map(|item| item.losses).unwrap_or(0),
        draws: stats.map(|item| item.draws).unwrap_or(0),
        total: stats.map(|item| item.total).unwrap_or(0),
        rating: rating.and_then(|item| item.rating),
        rating_lower: rating.and_then(|item| item.rating_lower),
        rating_upper: rating.and_then(|item| item.rating_upper),
        unavailable: rating.and_then(|item| item.unavailable),
    }
}

fn candidate_row(result: &EvaluatedResult) -> CandidateRow<'_> {
    CandidateRow {
        task_id: &result.task_id,
        model: &result.model,
        response: &result.response.text,
        score: result.evaluation.as_ref().map(|item| item.score),
        duration_ms: result.duration_ms,
    }
}

fn pair_row(judgment: &Judgment) -> PairRow<'_> {
    PairRow {
        task_id: &judgment.task_id,
        model_a: &judgment.model_a,
        model_b: &judgment.model_b,
        winner: judgment.winner.clone(),
        agreement: judgment.agreement,
        orientation_ab: judgment.orientation_ab.clone(),
        orientation_ba: judgment.orientation_ba.clone(),
        reason_ab: judgment.reason_ab.as_deref(),
        reason_ba: judgment.reason_ba.as_deref(),
        duration_ms: judgment.duration_ms,
        raw_ab: judgment.raw_ab.as_deref(),
        raw_ba: judgment.raw_ba.as_deref(),
    }
}

fn failed_pair_row(failure: &JudgmentFailure) -> FailedPairRow<'_> {
    FailedPairRow {
        task_id: &failure.task_id,
        model_a: &failure.model_a,
        model_b: &failure.model_b,
        orientations: &failure.orientations,
    }
}

fn bootstrap_report(output: &Output) -> Option<BootstrapReport> {
    Some(BootstrapReport {
        seed: output.run.bootstrap_seed?,
        replicates: output.run.bootstrap_replicates?,
        clusters: output.run.bootstrap_clusters?,
        ran: output.run.bootstrap_ran?,
        valid: output.run.bootstrap_valid,
        unavailable: output.run.bootstrap_unavailable,
        bounds_present: output.ratings.as_deref().is_some_and(|ratings| {
            ratings
                .iter()
                .any(|rating| rating.rating_lower.is_some() && rating.rating_upper.is_some())
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::bootstrap::{BootstrapMeta, BootstrapUnavailable};
    use crate::judge::{JudgeOrientation, JudgmentFailureKind};
    use crate::persist::RunMetadata;
    use crate::stats::{self, PairAgreement};
    use crate::task::Task;

    fn run_meta(models: &[&str], judge: Option<&str>) -> RunMetadata {
        RunMetadata::new(
            models.iter().map(|id| ModelId::new(*id)).collect(),
            judge.map(ModelId::new),
            Some(PathBuf::from("tasks.json")),
            "2026-01-02T03:04:05Z".into(),
            "https://example.test/v1",
        )
    }

    fn task(id: &str) -> Task {
        Task {
            id: id.into(),
            prompt: "p".into(),
            evaluation: None,
        }
    }

    fn judgment(
        task_id: &str,
        model_a: &str,
        model_b: &str,
        winner: JudgeDecision,
        agreement: bool,
        orientation_ab: JudgeDecision,
        orientation_ba: JudgeDecision,
    ) -> Judgment {
        Judgment {
            task_id: task_id.into(),
            model_a: ModelId::new(model_a),
            model_b: ModelId::new(model_b),
            judge_model: ModelId::new("judge"),
            winner,
            reason: String::new(),
            duration_ms: 12,
            agreement,
            orientation_ab: Some(orientation_ab),
            orientation_ba: Some(orientation_ba),
            reason_ab: Some("ab".into()),
            reason_ba: Some("ba".into()),
            raw_ab: Some(r#"{"winner":"a","reason":"ab"}"#.into()),
            raw_ba: Some(r#"{"winner":"b","reason":"ba"}"#.into()),
            raw: None,
        }
    }

    fn output(
        run: RunMetadata,
        tasks: Vec<Task>,
        judgments: Option<Vec<Judgment>>,
        judgment_failures: Option<Vec<JudgmentFailure>>,
        statistics: Option<Vec<ModelStats>>,
        ratings: Option<Vec<ModelRating>>,
    ) -> Output {
        Output {
            run,
            tasks,
            results: vec![],
            comparisons: vec![],
            judgments,
            judgment_failures,
            statistics,
            ratings,
        }
    }

    #[test]
    fn no_judge_output_produces_a_no_judge_report() {
        let output = output(
            run_meta(&["m0", "m1"], None),
            vec![task("t1")],
            None,
            None,
            None,
            None,
        );
        let report = from_output(&output);

        assert!(!report.summary.judge_used);
        assert_eq!(report.summary.complete, None);
        assert_eq!(report.summary.task_count, 1);
        assert_eq!(report.summary.candidate_count, 2);
        assert_eq!(report.summary.expected_pairs, None);
        assert_eq!(report.summary.resolved_pairs, None);
        assert_eq!(report.summary.failed_pairs, None);
        assert_eq!(report.summary.orientation_agreeing_pairs, None);
        assert_eq!(report.summary.orientation_disagreeing_pairs, None);
        assert_eq!(report.summary.agreement_rate, None);
        assert!(report.config.judge.is_none());
        assert!(report.config.judge_decoding.is_none());
        assert!(report.bootstrap.is_none());
        assert!(report.pairs.is_empty());
        assert!(report.failed_pairs.is_empty());
        assert!(report.results.is_empty());
        assert_eq!(report.models[0].model, &ModelId::new("m0"));
        assert_eq!(report.models[1].model, &ModelId::new("m1"));
        assert!(report.models.iter().all(|model| {
            model.wins == 0
                && model.rating.is_none()
                && model.unavailable.is_none()
                && model.rating_lower.is_none()
        }));
    }

    #[test]
    fn complete_judge_output_maps_coverage() {
        let judgments = vec![judgment(
            "t1",
            "m0",
            "m1",
            JudgeDecision::A,
            true,
            JudgeDecision::A,
            JudgeDecision::A,
        )];
        let models = vec![ModelId::new("m0"), ModelId::new("m1")];
        let statistics = stats::aggregate(&judgments, &models);
        let ratings = vec![
            ModelRating {
                model: ModelId::new("m0"),
                rating: Some(1600.0),
                rating_lower: Some(1500.0),
                rating_upper: Some(1700.0),
                unavailable: None,
            },
            ModelRating {
                model: ModelId::new("m1"),
                rating: Some(1400.0),
                rating_lower: Some(1300.0),
                rating_upper: Some(1500.0),
                unavailable: None,
            },
        ];
        let run = run_meta(&["m0", "m1"], Some("judge"))
            .with_judge_coverage(1, 1, 0)
            .with_orientation_agreement(stats::pair_agreement(&judgments))
            .with_judge_decoding(JudgeDecoding::arena_default())
            .with_bootstrap(&BootstrapMeta {
                seed: 0,
                replicates: 1000,
                valid: Some(1000),
                clusters: 2,
                ran: true,
                unavailable: None,
            });
        let output = output(
            run,
            vec![task("t1")],
            Some(judgments),
            Some(vec![]),
            Some(statistics),
            Some(ratings),
        );
        let report = from_output(&output);

        assert!(report.summary.judge_used);
        assert_eq!(report.summary.complete, Some(true));
        assert_eq!(report.summary.expected_pairs, Some(1));
        assert_eq!(report.summary.resolved_pairs, Some(1));
        assert_eq!(report.summary.failed_pairs, Some(0));
        assert_eq!(report.summary.orientation_agreeing_pairs, Some(1));
        assert_eq!(report.summary.orientation_disagreeing_pairs, Some(0));
        assert_eq!(report.summary.agreement_rate, Some(1.0));
        assert_eq!(report.pairs.len(), 1);
        assert!(report.failed_pairs.is_empty());
        assert_eq!(report.config.judge, Some(&ModelId::new("judge")));
        assert_eq!(report.config.candidate_max_tokens, 4096);
        assert_eq!(
            report.config.judge_decoding,
            Some(&JudgeDecoding::arena_default())
        );
        assert_eq!(report.bootstrap.as_ref().map(|item| item.ran), Some(true));
        assert_eq!(
            report.bootstrap.as_ref().map(|item| item.bounds_present),
            Some(true)
        );
    }

    #[test]
    fn incomplete_judge_output_preserves_failed_pairs() {
        let judgments = vec![judgment(
            "t1",
            "m0",
            "m1",
            JudgeDecision::A,
            true,
            JudgeDecision::A,
            JudgeDecision::A,
        )];
        let failure = JudgmentFailure {
            task_id: "t2".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            judge_model: ModelId::new("judge"),
            orientations: vec![OrientationFailure {
                orientation: JudgeOrientation::Ab,
                kind: JudgmentFailureKind::InvalidJson,
                error: "no valid judgment JSON found".into(),
                attempts: 3,
            }],
        };
        let models = vec![ModelId::new("m0"), ModelId::new("m1")];
        let statistics = stats::aggregate(&judgments, &models);
        let run = run_meta(&["m0", "m1"], Some("judge"))
            .with_judge_coverage(2, 1, 1)
            .with_orientation_agreement(stats::pair_agreement(&judgments))
            .with_judge_decoding(JudgeDecoding::arena_default());
        let output = output(
            run,
            vec![task("t1"), task("t2")],
            Some(judgments),
            Some(vec![failure]),
            Some(statistics),
            None,
        );
        let report = from_output(&output);

        assert_eq!(report.summary.complete, Some(false));
        assert_eq!(report.summary.expected_pairs, Some(2));
        assert_eq!(report.summary.resolved_pairs, Some(1));
        assert_eq!(report.summary.failed_pairs, Some(1));
        assert_eq!(report.pairs.len(), 1);
        assert_eq!(report.failed_pairs.len(), 1);
        assert_eq!(report.failed_pairs[0].task_id, "t2");
        assert_eq!(report.failed_pairs[0].model_a, &ModelId::new("m0"));
        assert_eq!(report.failed_pairs[0].model_b, &ModelId::new("m1"));
        assert_eq!(report.failed_pairs[0].orientations.len(), 1);
        assert_eq!(
            report.failed_pairs[0].orientations[0].kind,
            JudgmentFailureKind::InvalidJson
        );
    }

    #[test]
    fn pairwise_rows_preserve_orientation_and_final_winner() {
        let judgments = vec![judgment(
            "t1",
            "m0",
            "m1",
            JudgeDecision::Draw,
            false,
            JudgeDecision::A,
            JudgeDecision::B,
        )];
        let output = output(
            run_meta(&["m0", "m1"], Some("judge"))
                .with_judge_coverage(1, 1, 0)
                .with_orientation_agreement(stats::pair_agreement(&judgments)),
            vec![task("t1")],
            Some(judgments),
            Some(vec![]),
            None,
            None,
        );
        let report = from_output(&output);
        let row = &report.pairs[0];

        assert_eq!(row.task_id, "t1");
        assert_eq!(row.model_a, &ModelId::new("m0"));
        assert_eq!(row.model_b, &ModelId::new("m1"));
        assert_eq!(row.winner, JudgeDecision::Draw);
        assert!(!row.agreement);
        assert_eq!(row.orientation_ab, Some(JudgeDecision::A));
        assert_eq!(row.orientation_ba, Some(JudgeDecision::B));
        assert_eq!(row.reason_ab, Some("ab"));
        assert_eq!(row.reason_ba, Some("ba"));
        assert_eq!(row.duration_ms, 12);
        assert_eq!(row.raw_ab, Some(r#"{"winner":"a","reason":"ab"}"#));
        assert_eq!(row.raw_ba, Some(r#"{"winner":"b","reason":"ba"}"#));
    }

    #[test]
    fn unavailable_ratings_remain_unavailable_and_requested_models_remain_present() {
        let ratings = vec![
            ModelRating {
                model: ModelId::new("m0"),
                rating: Some(1500.0),
                rating_lower: None,
                rating_upper: None,
                unavailable: None,
            },
            ModelRating {
                model: ModelId::new("m1"),
                rating: None,
                rating_lower: None,
                rating_upper: None,
                unavailable: Some(UnavailableReason::NoComparisons),
            },
            ModelRating {
                model: ModelId::new("m2"),
                rating: None,
                rating_lower: None,
                rating_upper: None,
                unavailable: Some(UnavailableReason::NoComparisons),
            },
        ];
        let statistics = stats::aggregate(
            &[judgment(
                "t1",
                "m0",
                "m1",
                JudgeDecision::A,
                true,
                JudgeDecision::A,
                JudgeDecision::A,
            )],
            &[ModelId::new("m0"), ModelId::new("m1"), ModelId::new("m2")],
        );
        let output = output(
            run_meta(&["m0", "m1", "m2"], Some("judge")).with_judge_coverage(3, 1, 2),
            vec![task("t1")],
            Some(vec![]),
            Some(vec![]),
            Some(statistics),
            Some(ratings),
        );
        let report = from_output(&output);

        assert_eq!(report.models.len(), 3);
        assert_eq!(report.models[0].model, &ModelId::new("m0"));
        assert_eq!(report.models[0].rating, Some(1500.0));
        assert_eq!(report.models[0].unavailable, None);
        assert_eq!(report.models[1].model, &ModelId::new("m1"));
        assert_eq!(report.models[1].rating, None);
        assert_eq!(
            report.models[1].unavailable,
            Some(UnavailableReason::NoComparisons)
        );
        assert!(report.models[1].rating_lower.is_none());
        assert!(report.models[1].rating_upper.is_none());
        assert_eq!(report.models[2].model, &ModelId::new("m2"));
        assert_eq!(report.models[2].wins, 0);
        assert_eq!(report.models[2].total, 0);
        assert_eq!(
            report.models[2].unavailable,
            Some(UnavailableReason::NoComparisons)
        );
    }

    #[test]
    fn bootstrap_invalid_replicates_state_is_preserved() {
        let ratings = vec![ModelRating {
            model: ModelId::new("m0"),
            rating: Some(1500.0),
            rating_lower: None,
            rating_upper: None,
            unavailable: None,
        }];
        let run = run_meta(&["m0"], Some("judge"))
            .with_judge_coverage(0, 0, 0)
            .with_orientation_agreement(PairAgreement {
                resolved_pairs: 0,
                orientation_agreeing_pairs: 0,
                orientation_disagreeing_pairs: 0,
                agreement_rate: 0.0,
            })
            .with_bootstrap(&BootstrapMeta {
                seed: 7,
                replicates: 1000,
                valid: Some(744),
                clusters: 2,
                ran: true,
                unavailable: Some(BootstrapUnavailable::InvalidReplicates),
            });
        let output = output(
            run,
            vec![task("t1"), task("t2")],
            Some(vec![]),
            Some(vec![]),
            Some(stats::aggregate(&[], &[ModelId::new("m0")])),
            Some(ratings),
        );
        let report = from_output(&output);
        let bootstrap = report.bootstrap.expect("judge run records bootstrap");

        assert_eq!(bootstrap.seed, 7);
        assert_eq!(bootstrap.replicates, 1000);
        assert_eq!(bootstrap.valid, Some(744));
        assert_eq!(bootstrap.clusters, 2);
        assert!(bootstrap.ran);
        assert_eq!(
            bootstrap.unavailable,
            Some(BootstrapUnavailable::InvalidReplicates)
        );
        assert!(!bootstrap.bounds_present);
    }
}
