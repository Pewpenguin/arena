use crate::execute::ExecutionResult;
use crate::task::Task;

#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    pub score: f64,
}

pub trait Evaluator {
    fn evaluate(&self, task: &Task, result: &ExecutionResult) -> Evaluation;
}
