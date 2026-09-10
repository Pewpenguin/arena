use thiserror::Error;

use crate::persist::PersistError;
use crate::provider::ProviderError;
use crate::task::TaskError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to initialize logging")]
    LoggingInit,
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Task(#[from] TaskError),
    #[error(transparent)]
    Persist(#[from] PersistError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
