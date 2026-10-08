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
    #[error("incomplete run: {0} judgment(s) failed")]
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
    #[error("model universe is empty; load or discover models before random selection")]
    EmptyModelUniverse,
    #[error(
        "model universe is too small for random selection: need at least {needed} models, got {available}"
    )]
    UniverseTooSmall { needed: usize, available: usize },
    #[error("candidate count must be between 1 and {available}, got {count}")]
    InvalidCandidateCount { count: usize, available: usize },
    #[error("use either manual model selection or random selection, not both")]
    ConflictingSelectionMode,
    #[error(
        "random selection requires a provider that can list models (openai or openrouter); {provider} cannot"
    )]
    RandomRequiresDiscovery { provider: &'static str },
    #[error("provided candidates do not match Arena's random selection for this seed")]
    SelectionMismatch,
    #[error(
        "{format} requires a power-of-two number of candidates (2, 4, 8, ...), got {candidates}"
    )]
    UnsupportedTournament {
        format: &'static str,
        candidates: usize,
    },
    #[error("missing execution result for candidate {model} on task {task_id}")]
    MissingCandidateResult { task_id: String, model: ModelId },
    #[error("best-of must be an odd number of games (1, 3, 5, ...), got {best_of}")]
    InvalidBestOf { best_of: u32 },
    #[error("single-elimination cannot advance {winner} from {model_a} vs {model_b}")]
    InvalidTournamentWinner {
        model_a: ModelId,
        model_b: ModelId,
        winner: ModelId,
    },
    #[error("opening matchups are only available for single-elimination")]
    OpeningMatchupsRequireElimination,
    #[error("opening matchups must assign every candidate to exactly one slot")]
    IncompleteOpeningMatchups,
    #[error("opening matchups repeat candidate {0}")]
    DuplicateOpeningMatchup(ModelId),
    #[error("opening matchups include {0}, which is not a selected candidate")]
    InvalidOpeningMatchup(ModelId),
    #[error("opening matchups omit candidate {0}")]
    MissingOpeningMatchup(ModelId),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
