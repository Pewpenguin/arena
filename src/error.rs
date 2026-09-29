use thiserror::Error;

use crate::execute::ExecutionError;
use crate::judge::JudgeError;
use crate::persist::PersistError;
use crate::provider::{ModelId, ProviderError};
use crate::task::TaskError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Execute(#[from] ExecutionError),
    #[error(transparent)]
    Task(#[from] TaskError),
    #[error(transparent)]
    Persist(#[from] PersistError),
    #[error(transparent)]
    Judge(#[from] JudgeError),
    #[error("duplicate model id: {0}")]
    DuplicateModel(ModelId),
    #[error("judge model is also a candidate: {0}")]
    JudgeIsCandidate(ModelId),
    #[error("incomplete run: {0} judgment pair(s) failed")]
    IncompleteJudgments(usize),
    #[error(
        "inconsistent pair coverage: expected {expected}, resolved {resolved}, failed {failed}"
    )]
    InconsistentPairCoverage {
        expected: usize,
        resolved: usize,
        failed: usize,
    },
    #[error("at least one candidate model is required")]
    NoCandidates,
    #[error(
        "{format} requires a power-of-two number of candidates (2, 4, 8, ...), got {candidates}"
    )]
    UnsupportedTournament {
        format: &'static str,
        candidates: usize,
    },
    #[error("missing execution result for candidate {model} on task {task_id}")]
    MissingCandidateResult { task_id: String, model: ModelId },
    #[error("single-elimination cannot advance {winner} from {model_a} vs {model_b}")]
    InvalidTournamentWinner {
        model_a: ModelId,
        model_b: ModelId,
        winner: ModelId,
    },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
