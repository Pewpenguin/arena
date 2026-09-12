use std::collections::BTreeMap;

use serde::Serialize;

use crate::judge::{JudgeDecision, Judgment};
use crate::provider::ModelId;

const INITIAL: f64 = 1500.0;
const K: f64 = 32.0;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelRating {
    pub model: ModelId,
    pub rating: f64,
}

pub fn rate(judgments: &[Judgment]) -> Vec<ModelRating> {
    let mut ratings: BTreeMap<ModelId, f64> = BTreeMap::new();

    for judgment in judgments {
        let rating_a = *ratings.get(&judgment.model_a).unwrap_or(&INITIAL);
        let rating_b = *ratings.get(&judgment.model_b).unwrap_or(&INITIAL);

        let (score_a, score_b) = match judgment.winner {
            JudgeDecision::A => (1.0, 0.0),
            JudgeDecision::B => (0.0, 1.0),
            JudgeDecision::Draw => (0.5, 0.5),
        };

        let expected_a = expected(rating_a, rating_b);
        let expected_b = 1.0 - expected_a;

        ratings.insert(
            judgment.model_a.clone(),
            rating_a + K * (score_a - expected_a),
        );
        ratings.insert(
            judgment.model_b.clone(),
            rating_b + K * (score_b - expected_b),
        );
    }

    ratings
        .into_iter()
        .map(|(model, rating)| ModelRating { model, rating })
        .collect()
}

fn expected(rating: f64, opponent: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((opponent - rating) / 400.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judgment(model_a: &str, model_b: &str, winner: JudgeDecision) -> Judgment {
        Judgment {
            task_id: "t1".into(),
            model_a: ModelId::new(model_a),
            model_b: ModelId::new(model_b),
            judge_model: ModelId::new("judge"),
            winner,
            reason: String::new(),
            duration_ms: 0,
        }
    }

    fn rating(ratings: &[ModelRating], model: &str) -> f64 {
        ratings
            .iter()
            .find(|r| r.model == ModelId::new(model))
            .unwrap()
            .rating
    }

    #[test]
    fn rates_initial_a_win() {
        let ratings = rate(&[judgment("a", "b", JudgeDecision::A)]);

        assert_eq!(rating(&ratings, "a"), 1516.0);
        assert_eq!(rating(&ratings, "b"), 1484.0);
    }

    #[test]
    fn rates_initial_draw() {
        let ratings = rate(&[judgment("a", "b", JudgeDecision::Draw)]);

        assert_eq!(rating(&ratings, "a"), 1500.0);
        assert_eq!(rating(&ratings, "b"), 1500.0);
    }

    #[test]
    fn applies_judgments_sequentially() {
        // First A win from 1500/1500 → 1516 / 1484.
        // Second A win from those ratings:
        //   E_A = 1 / (1 + 10^((1484 - 1516) / 400))
        //   A → 1516 + 32 * (1 - E_A)
        //   B → 1484 + 32 * (0 - (1 - E_A))
        let e_a = 1.0 / (1.0 + 10f64.powf((1484.0 - 1516.0) / 400.0));
        let expected_a = 1516.0 + 32.0 * (1.0 - e_a);
        let expected_b = 1484.0 + 32.0 * (0.0 - (1.0 - e_a));

        let ratings = rate(&[
            judgment("a", "b", JudgeDecision::A),
            judgment("a", "b", JudgeDecision::A),
        ]);

        assert_eq!(rating(&ratings, "a"), expected_a);
        assert_eq!(rating(&ratings, "b"), expected_b);
        // Independent (wrong) double application from 1500 would yield 1532 / 1468.
        assert_ne!(expected_a, 1532.0);
        assert_ne!(expected_b, 1468.0);
    }
}
