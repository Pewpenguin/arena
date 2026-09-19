use crate::execute::ExecutionResult;
use crate::provider::{CompletionResponse, ModelId};
use crate::task::{Task, TaskEvaluation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatedResult {
    pub task_id: String,
    pub model: ModelId,
    pub response: CompletionResponse,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<Evaluation>,
    pub duration_ms: u64,
}

pub fn evaluate_exact(task: &Task, result: &ExecutionResult) -> Option<Evaluation> {
    let Some(TaskEvaluation::Exact { expected }) = &task.evaluation else {
        return None;
    };

    let score = if result.response.text.trim() == expected {
        1.0
    } else {
        0.0
    };

    Some(Evaluation { score })
}

pub fn evaluate_result(task: &Task, result: ExecutionResult) -> EvaluatedResult {
    EvaluatedResult {
        evaluation: evaluate_exact(task, &result),
        task_id: result.task_id,
        model: result.model,
        response: result.response,
        duration_ms: result.duration_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(text: &str) -> ExecutionResult {
        ExecutionResult {
            task_id: "t1".into(),
            model: ModelId::new("model"),
            response: CompletionResponse { text: text.into() },
            duration_ms: 0,
        }
    }

    fn exact_task(expected: &str) -> Task {
        Task {
            id: "t1".into(),
            prompt: "prompt".into(),
            evaluation: Some(TaskEvaluation::Exact {
                expected: expected.into(),
            }),
        }
    }

    #[test]
    fn scores_matching_response() {
        let evaluation = evaluate_exact(&exact_task("Paris"), &result("Paris")).unwrap();
        assert_eq!(evaluation.score, 1.0);
    }

    #[test]
    fn scores_non_matching_response() {
        let evaluation = evaluate_exact(&exact_task("Paris"), &result("London")).unwrap();
        assert_eq!(evaluation.score, 0.0);
    }

    #[test]
    fn ignores_surrounding_whitespace() {
        let evaluation = evaluate_exact(&exact_task("Paris"), &result("  Paris\n")).unwrap();
        assert_eq!(evaluation.score, 1.0);
    }

    #[test]
    fn skips_tasks_without_evaluation() {
        let task = Task {
            id: "t1".into(),
            prompt: "prompt".into(),
            evaluation: None,
        };

        assert!(evaluate_exact(&task, &result("Paris")).is_none());
    }

    #[test]
    fn evaluate_result_preserves_duration_ms() {
        let mut execution = result("Paris");
        execution.duration_ms = 42;
        let evaluated = evaluate_result(&exact_task("Paris"), execution);

        assert_eq!(evaluated.duration_ms, 42);
        assert_eq!(serde_json::to_value(&evaluated).unwrap()["duration_ms"], 42);
    }
}
