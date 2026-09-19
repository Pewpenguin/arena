use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::judge::{JudgeDecision, Judgment};
use crate::provider::ModelId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStats {
    pub model: ModelId,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub total: u32,
    /// Resolved pairs this model appeared in whose orientations agreed.
    /// One increment per pair, not a second independent observation of that pair.
    pub agreement_count: u32,
    pub disagreement_count: u32,
    /// Among this model's resolved pairs only. The run-level pair summary is the
    /// primary orientation-agreement metric.
    pub agreement_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairAgreement {
    pub resolved_pairs: usize,
    pub orientation_agreeing_pairs: usize,
    pub orientation_disagreeing_pairs: usize,
    pub agreement_rate: f64,
}

pub fn pair_agreement(judgments: &[Judgment]) -> PairAgreement {
    let resolved_pairs = judgments.len();
    let orientation_agreeing_pairs = judgments
        .iter()
        .filter(|judgment| judgment.agreement)
        .count();
    let orientation_disagreeing_pairs = resolved_pairs - orientation_agreeing_pairs;
    let agreement_rate = if resolved_pairs == 0 {
        0.0
    } else {
        orientation_agreeing_pairs as f64 / resolved_pairs as f64
    };
    PairAgreement {
        resolved_pairs,
        orientation_agreeing_pairs,
        orientation_disagreeing_pairs,
        agreement_rate,
    }
}

pub fn aggregate(judgments: &[Judgment], models: &[ModelId]) -> Vec<ModelStats> {
    let mut stats: BTreeMap<ModelId, ModelStats> = BTreeMap::new();
    for model in models {
        entry(&mut stats, model);
    }

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

        for model in [&judgment.model_a, &judgment.model_b] {
            let stat = entry(&mut stats, model);
            if judgment.agreement {
                stat.agreement_count += 1;
            } else {
                stat.disagreement_count += 1;
            }
        }
    }

    for stat in stats.values_mut() {
        stat.total = stat.wins + stat.losses + stat.draws;
        stat.agreement_rate = if stat.total == 0 {
            0.0
        } else {
            stat.agreement_count as f64 / stat.total as f64
        };
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
        agreement_count: 0,
        disagreement_count: 0,
        agreement_rate: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judgment(model_a: &str, model_b: &str, winner: JudgeDecision, agreement: bool) -> Judgment {
        Judgment {
            task_id: "t1".into(),
            model_a: ModelId::new(model_a),
            model_b: ModelId::new(model_b),
            judge_model: ModelId::new("judge"),
            winner,
            reason: String::new(),
            duration_ms: 0,
            agreement,
            orientation_ab: None,
            orientation_ba: None,
            reason_ab: None,
            reason_ba: None,
            raw_ab: None,
            raw_ba: None,
            raw: None,
        }
    }

    #[test]
    fn empty_judgments_produce_no_stats() {
        assert!(aggregate(&[], &[]).is_empty());
    }

    #[test]
    fn requested_models_are_present_without_inferring_outcomes() {
        let stats = aggregate(&[], &[ModelId::new("a")]);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].model, ModelId::new("a"));
        assert_eq!(
            (
                stats[0].wins,
                stats[0].losses,
                stats[0].draws,
                stats[0].total
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            (stats[0].agreement_count, stats[0].disagreement_count),
            (0, 0)
        );
        assert_eq!(stats[0].agreement_rate, 0.0);

        let stats = aggregate(
            &[judgment("a", "b", JudgeDecision::A, true)],
            &[ModelId::new("a"), ModelId::new("b"), ModelId::new("c")],
        );
        assert_eq!(stats.len(), 3);
        let a = stats.iter().find(|s| s.model == ModelId::new("a")).unwrap();
        assert_eq!(a.wins, 1);
        let c = stats.iter().find(|s| s.model == ModelId::new("c")).unwrap();
        assert_eq!((c.wins, c.losses, c.draws, c.total), (0, 0, 0, 0));
        assert_eq!(c.agreement_rate, 0.0);
    }

    #[test]
    fn counts_wins_losses_and_draws() {
        let judgments = vec![
            judgment("a", "b", JudgeDecision::A, true),
            judgment("a", "b", JudgeDecision::B, true),
            judgment("a", "b", JudgeDecision::Draw, true),
        ];

        let stats = aggregate(&judgments, &[]);

        let a = stats.iter().find(|s| s.model == ModelId::new("a")).unwrap();
        assert_eq!((a.wins, a.losses, a.draws, a.total), (1, 1, 1, 3));
        assert_eq!((a.agreement_count, a.disagreement_count), (3, 0));
        assert_eq!(a.agreement_rate, 1.0);

        let b = stats.iter().find(|s| s.model == ModelId::new("b")).unwrap();
        assert_eq!((b.wins, b.losses, b.draws, b.total), (1, 1, 1, 3));
        assert_eq!((b.agreement_count, b.disagreement_count), (3, 0));
        assert_eq!(b.agreement_rate, 1.0);
    }

    #[test]
    fn counts_disagreement_without_changing_outcome_tallies() {
        let judgments = vec![
            judgment("a", "b", JudgeDecision::A, true),
            judgment("a", "b", JudgeDecision::Draw, false),
        ];

        let stats = aggregate(&judgments, &[]);

        let a = stats.iter().find(|s| s.model == ModelId::new("a")).unwrap();
        assert_eq!((a.wins, a.losses, a.draws, a.total), (1, 0, 1, 2));
        assert_eq!((a.agreement_count, a.disagreement_count), (1, 1));
        assert_eq!(a.agreement_rate, 0.5);

        let b = stats.iter().find(|s| s.model == ModelId::new("b")).unwrap();
        assert_eq!((b.wins, b.losses, b.draws, b.total), (0, 1, 1, 2));
        assert_eq!((b.agreement_count, b.disagreement_count), (1, 1));
        assert_eq!(b.agreement_rate, 0.5);
    }

    #[test]
    fn pair_agreement_all_orientations_agree() {
        let all_agree = [
            judgment("a", "b", JudgeDecision::A, true),
            judgment("a", "c", JudgeDecision::B, true),
        ];
        let summary = pair_agreement(&all_agree);
        assert_eq!(summary.resolved_pairs, 2);
        assert_eq!(summary.orientation_agreeing_pairs, 2);
        assert_eq!(summary.orientation_disagreeing_pairs, 0);
        assert_eq!(summary.agreement_rate, 1.0);
    }

    #[test]
    fn pair_agreement_mixed_agreeing_and_disagreeing() {
        let mixed = [
            judgment("a", "b", JudgeDecision::A, true),
            judgment("a", "c", JudgeDecision::Draw, false),
            judgment("b", "c", JudgeDecision::B, true),
        ];
        let summary = pair_agreement(&mixed);
        assert_eq!(summary.resolved_pairs, 3);
        assert_eq!(summary.orientation_agreeing_pairs, 2);
        assert_eq!(summary.orientation_disagreeing_pairs, 1);
        assert_eq!(summary.agreement_rate, 2.0 / 3.0);
    }

    #[test]
    fn pair_agreement_zero_resolved_pairs() {
        let empty = pair_agreement(&[]);
        assert_eq!(empty.resolved_pairs, 0);
        assert_eq!(empty.orientation_agreeing_pairs, 0);
        assert_eq!(empty.orientation_disagreeing_pairs, 0);
        assert_eq!(empty.agreement_rate, 0.0);
    }

    #[test]
    fn pair_agreement_does_not_double_count_endpoints() {
        let one = [judgment("a", "b", JudgeDecision::A, true)];
        let summary = pair_agreement(&one);
        assert_eq!(summary.resolved_pairs, 1);
        let stats = aggregate(&one, &[ModelId::new("a"), ModelId::new("b")]);
        let endpoint_sum: u32 = stats.iter().map(|s| s.agreement_count).sum();
        assert_eq!(
            endpoint_sum, 2,
            "per-model counts attribute the pair to both endpoints"
        );
        assert_eq!(
            summary.orientation_agreeing_pairs, 1,
            "pair-level summary must not double-count endpoints"
        );
    }
}
