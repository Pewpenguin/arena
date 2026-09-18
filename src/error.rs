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
    #[error("incomplete run: {0} judgment pair(s) failed")]
    IncompleteJudgments(usize),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
