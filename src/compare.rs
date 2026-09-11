use crate::evaluate::EvaluatedResult;
use crate::provider::ModelId;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Comparison {
    pub task_id: String,
    pub model_a: ModelId,
    pub model_b: ModelId,
    pub winner: Option<ModelId>,
}

pub fn compare(a: &EvaluatedResult, b: &EvaluatedResult) -> Option<Comparison> {
    if a.task_id != b.task_id {
        return None;
    }

    let score_a = a.evaluation.as_ref()?.score;
    let score_b = b.evaluation.as_ref()?.score;

    let winner = if score_a > score_b {
        Some(a.model.clone())
    } else if score_b > score_a {
        Some(b.model.clone())
    } else {
        None
    };

    Some(Comparison {
        task_id: a.task_id.clone(),
        model_a: a.model.clone(),
        model_b: b.model.clone(),
        winner,
    })
}

pub fn compare_all(results: &[EvaluatedResult]) -> Vec<Comparison> {
    let mut comparisons = Vec::new();

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            if let Some(comparison) = compare(&results[i], &results[j]) {
                comparisons.push(comparison);
            }
        }
    }

    comparisons
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate::Evaluation;
    use crate::provider::CompletionResponse;

    fn evaluated(task_id: &str, model: &str, score: Option<f64>) -> EvaluatedResult {
        EvaluatedResult {
            task_id: task_id.into(),
            model: ModelId::new(model),
            response: CompletionResponse {
                text: String::new(),
            },
            evaluation: score.map(|score| Evaluation { score }),
            duration_ms: 0,
        }
    }

    #[test]
    fn higher_score_wins() {
        let a = evaluated("t1", "model-a", Some(1.0));
        let b = evaluated("t1", "model-b", Some(0.0));

        let comparison = compare(&a, &b).unwrap();
        assert_eq!(comparison.winner, Some(ModelId::new("model-a")));
    }

    #[test]
    fn equal_scores_draw() {
        let a = evaluated("t1", "model-a", Some(1.0));
        let b = evaluated("t1", "model-b", Some(1.0));

        let comparison = compare(&a, &b).unwrap();
        assert_eq!(comparison.winner, None);
    }

    #[test]
    fn missing_evaluation_produces_no_comparison() {
        let a = evaluated("t1", "model-a", Some(1.0));
        let b = evaluated("t1", "model-b", None);

        assert!(compare(&a, &b).is_none());
    }

    #[test]
    fn different_tasks_are_not_compared() {
        let a = evaluated("t1", "model-a", Some(1.0));
        let b = evaluated("t2", "model-b", Some(0.0));

        assert!(compare(&a, &b).is_none());
    }
}
