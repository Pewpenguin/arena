use std::collections::BTreeMap;

use serde::Serialize;

use crate::judge::{JudgeDecision, Judgment};
use crate::provider::ModelId;

const CENTER: f64 = 1500.0;
const SCALE: f64 = 400.0;
const MAX_ITERS: u32 = 10_000;
const TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelRating {
    pub model: ModelId,
    pub rating: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rating_lower: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rating_upper: Option<f64>,
}

pub fn rate(judgments: &[Judgment]) -> Vec<ModelRating> {
    let mut models: BTreeMap<ModelId, usize> = BTreeMap::new();
    for judgment in judgments {
        models.insert(judgment.model_a.clone(), 0);
        models.insert(judgment.model_b.clone(), 0);
    }
    for (index, slot) in models.values_mut().enumerate() {
        *slot = index;
    }

    let n = models.len();
    if n == 0 {
        return Vec::new();
    }

    let mut parent: Vec<usize> = (0..n).collect();
    for judgment in judgments {
        let a = models[&judgment.model_a];
        let b = models[&judgment.model_b];
        let pa = find(&mut parent, a);
        let pb = find(&mut parent, b);
        if pa != pb {
            parent[pa] = pb;
        }
    }

    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        members.entry(find(&mut parent, i)).or_default().push(i);
    }

    let mut ratings = vec![None; n];
    let comparable_groups: Vec<_> = members.values().filter(|group| group.len() >= 2).collect();
    if comparable_groups.len() == 1 {
        let group = comparable_groups[0];
        if let Some(fitted) = fit_component(group, &models, judgments) {
            for (index, rating) in group.iter().zip(fitted) {
                ratings[*index] = Some(rating);
            }
        }
    }

    models
        .into_iter()
        .map(|(model, index)| ModelRating {
            model,
            rating: ratings[index],
            rating_lower: None,
            rating_upper: None,
        })
        .collect()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn fit_component(
    group: &[usize],
    models: &BTreeMap<ModelId, usize>,
    judgments: &[Judgment],
) -> Option<Vec<f64>> {
    let local: BTreeMap<usize, usize> = group
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect();
    let m = group.len();
    let mut wins = vec![0.0; m];
    let mut games = vec![vec![0u32; m]; m];

    for judgment in judgments {
        let Some(&a) = models.get(&judgment.model_a).and_then(|i| local.get(i)) else {
            continue;
        };
        let Some(&b) = models.get(&judgment.model_b).and_then(|i| local.get(i)) else {
            continue;
        };

        games[a][b] += 1;
        games[b][a] += 1;
        match judgment.winner {
            JudgeDecision::A => wins[a] += 1.0,
            JudgeDecision::B => wins[b] += 1.0,
            JudgeDecision::Draw => {
                wins[a] += 0.5;
                wins[b] += 0.5;
            }
        }
    }

    let mut strength = vec![1.0; m];
    for _ in 0..MAX_ITERS {
        let mut next = vec![0.0; m];
        let mut delta = 0.0_f64;
        for i in 0..m {
            let mut denom = 0.0;
            for (j, n_ij) in games[i].iter().enumerate() {
                if i == j || *n_ij == 0 {
                    continue;
                }
                denom += f64::from(*n_ij) / (strength[i] + strength[j]);
            }
            next[i] = if denom > 0.0 { wins[i] / denom } else { 0.0 };
            delta = delta.max((next[i] - strength[i]).abs());
        }
        let total: f64 = next.iter().sum();
        if total > 0.0 {
            for value in &mut next {
                *value /= total;
            }
        }
        strength = next;
        if delta < TOLERANCE {
            break;
        }
    }

    if strength
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return None;
    }

    let logs: Vec<f64> = strength.iter().map(|value| value.ln()).collect();
    if logs.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mean = logs.iter().sum::<f64>() / m as f64;
    let scale = SCALE / std::f64::consts::LN_10;
    let ratings: Vec<f64> = logs
        .into_iter()
        .map(|value| CENTER + scale * (value - mean))
        .collect();
    if ratings.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(ratings)
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
            agreement: true,
            orientation_ab: None,
            orientation_ba: None,
        }
    }

    fn rating(ratings: &[ModelRating], model: &str) -> Option<f64> {
        ratings
            .iter()
            .find(|r| r.model == ModelId::new(model))
            .and_then(|r| r.rating)
    }

    #[test]
    fn single_a_win_ranks_a_above_b() {
        let ratings = rate(&[judgment("a", "b", JudgeDecision::A)]);
        assert_eq!(rating(&ratings, "a"), None);
        assert_eq!(rating(&ratings, "b"), None);
    }

    #[test]
    fn single_b_win_ranks_b_above_a() {
        let ratings = rate(&[judgment("a", "b", JudgeDecision::B)]);
        assert_eq!(rating(&ratings, "a"), None);
        assert_eq!(rating(&ratings, "b"), None);
    }

    #[test]
    fn all_draws_produce_equal_ratings() {
        let ratings = rate(&[
            judgment("a", "b", JudgeDecision::Draw),
            judgment("a", "c", JudgeDecision::Draw),
            judgment("b", "c", JudgeDecision::Draw),
        ]);
        assert_eq!(rating(&ratings, "a"), Some(CENTER));
        assert_eq!(rating(&ratings, "b"), Some(CENTER));
        assert_eq!(rating(&ratings, "c"), Some(CENTER));
    }

    #[test]
    fn reordering_judgments_preserves_ratings() {
        let first = rate(&[
            judgment("a", "b", JudgeDecision::A),
            judgment("b", "c", JudgeDecision::B),
            judgment("a", "c", JudgeDecision::Draw),
        ]);
        let second = rate(&[
            judgment("a", "c", JudgeDecision::Draw),
            judgment("b", "c", JudgeDecision::B),
            judgment("a", "b", JudgeDecision::A),
        ]);
        assert_eq!(first, second);
    }

    #[test]
    fn agreed_orientations_do_not_double_count_a_resolved_win() {
        let ab = judgment("a", "b", JudgeDecision::A);
        let bc = judgment("b", "c", JudgeDecision::A);
        let ca = judgment("c", "a", JudgeDecision::A);
        let mut agreed_ab = ab.clone();
        agreed_ab.orientation_ab = Some(JudgeDecision::A);
        agreed_ab.orientation_ba = Some(JudgeDecision::A);

        let one_resolved = rate(&[ab.clone(), bc.clone(), ca.clone()]);
        let one_agreed = rate(&[agreed_ab, bc.clone(), ca.clone()]);
        assert_eq!(one_resolved, one_agreed);

        let two_resolved = rate(&[ab.clone(), ab, bc, ca]);
        assert!(
            rating(&two_resolved, "a").unwrap() - rating(&two_resolved, "b").unwrap()
                > rating(&one_agreed, "a").unwrap() - rating(&one_agreed, "b").unwrap()
        );
    }

    #[test]
    fn orientation_fields_round_trip_json() {
        let judgment = Judgment {
            task_id: "t1".into(),
            model_a: ModelId::new("a"),
            model_b: ModelId::new("b"),
            judge_model: ModelId::new("judge"),
            winner: JudgeDecision::A,
            reason: "A is better".into(),
            duration_ms: 12,
            agreement: true,
            orientation_ab: Some(JudgeDecision::A),
            orientation_ba: Some(JudgeDecision::A),
        };

        let value = serde_json::to_value(&judgment).unwrap();
        assert_eq!(value["orientation_ab"], "a");
        assert_eq!(value["orientation_ba"], "a");
        assert_eq!(serde_json::from_value::<Judgment>(value).unwrap(), judgment);
    }

    #[test]
    fn old_judgment_json_without_orientations_still_rates() {
        let judgment: Judgment = serde_json::from_str(
            r#"{
                "task_id": "t1",
                "model_a": "a",
                "model_b": "b",
                "judge_model": "judge",
                "winner": "a",
                "reason": "legacy",
                "duration_ms": 0,
                "agreement": true
            }"#,
        )
        .unwrap();

        assert_eq!(judgment.orientation_ab, None);
        assert_eq!(judgment.orientation_ba, None);
        assert_eq!(judgment.winner, JudgeDecision::A);

        let ratings = rate(&[judgment]);
        assert_eq!(rating(&ratings, "a"), None);
        assert_eq!(rating(&ratings, "b"), None);
    }

    #[test]
    fn separated_component_has_no_finite_ratings() {
        let ratings = rate(&[
            judgment("a", "b", JudgeDecision::A),
            judgment("a", "b", JudgeDecision::A),
        ]);
        assert_eq!(rating(&ratings, "a"), None);
        assert_eq!(rating(&ratings, "b"), None);
    }

    #[test]
    fn disconnected_models_do_not_share_a_rating_scale() {
        let ratings = rate(&[
            judgment("a", "b", JudgeDecision::A),
            judgment("c", "d", JudgeDecision::A),
        ]);
        assert_eq!(rating(&ratings, "a"), None);
        assert_eq!(rating(&ratings, "b"), None);
        assert_eq!(rating(&ratings, "c"), None);
        assert_eq!(rating(&ratings, "d"), None);
    }
}
