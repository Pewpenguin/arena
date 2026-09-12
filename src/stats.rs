use std::collections::BTreeMap;

use serde::Serialize;

use crate::judge::{JudgeDecision, Judgment};
use crate::provider::ModelId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelStats {
    pub model: ModelId,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub total: u32,
}

pub fn aggregate(judgments: &[Judgment]) -> Vec<ModelStats> {
    let mut stats: BTreeMap<ModelId, ModelStats> = BTreeMap::new();

    for judgment in judgments {
        match judgment.winner {
            JudgeDecision::A => {
                entry(&mut stats, &judgment.model_a).wins += 1;
                entry(&mut stats, &judgment.model_b).losses += 1;
            }
            JudgeDecision::B => {
                entry(&mut stats, &judgment.model_b).wins += 1;
                entry(&mut stats, &judgment.model_a).losses += 1;
            }
            JudgeDecision::Draw => {
                entry(&mut stats, &judgment.model_a).draws += 1;
                entry(&mut stats, &judgment.model_b).draws += 1;
            }
        }
    }

    for stat in stats.values_mut() {
        stat.total = stat.wins + stat.losses + stat.draws;
    }

    stats.into_values().collect()
}

fn entry<'a>(stats: &'a mut BTreeMap<ModelId, ModelStats>, model: &ModelId) -> &'a mut ModelStats {
    stats.entry(model.clone()).or_insert_with(|| ModelStats {
        model: model.clone(),
        wins: 0,
        losses: 0,
        draws: 0,
        total: 0,
    })
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

    #[test]
    fn empty_judgments_produce_no_stats() {
        assert!(aggregate(&[]).is_empty());
    }

    #[test]
    fn counts_wins_losses_and_draws() {
        let judgments = vec![
            judgment("a", "b", JudgeDecision::A),
            judgment("a", "b", JudgeDecision::B),
            judgment("a", "b", JudgeDecision::Draw),
        ];

        let stats = aggregate(&judgments);

        let a = stats.iter().find(|s| s.model == ModelId::new("a")).unwrap();
        assert_eq!((a.wins, a.losses, a.draws, a.total), (1, 1, 1, 3));

        let b = stats.iter().find(|s| s.model == ModelId::new("b")).unwrap();
        assert_eq!((b.wins, b.losses, b.draws, b.total), (1, 1, 1, 3));
    }
}
