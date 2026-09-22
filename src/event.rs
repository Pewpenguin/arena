use crate::judge::{Judgment, JudgmentFailure};
use crate::provider::ModelId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentEvent {
    CandidateFinished {
        task_id: String,
        model: ModelId,
        duration_ms: u64,
    },
    PairResolved {
        task_id: String,
        model_a: ModelId,
        model_b: ModelId,
        judgment: Judgment,
    },
    PairFailed {
        task_id: String,
        model_a: ModelId,
        model_b: ModelId,
        failure: JudgmentFailure,
    },
    RunComplete {
        expected_pairs: usize,
        resolved_pairs: usize,
        failed_pairs: usize,
    },
}
