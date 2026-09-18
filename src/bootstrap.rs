use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::judge::Judgment;
use crate::rating::{self, ModelRating};

pub const BOOTSTRAP_REPLICATES: u32 = 1_000;
const LOWER_P: f64 = 0.025;
const UPPER_P: f64 = 0.975;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapMeta {
    pub seed: u64,
    pub replicates: u32,
    pub valid: u32,
}

pub fn rate_with_uncertainty(
    judgments: &[Judgment],
    seed: u64,
) -> (Vec<ModelRating>, Option<BootstrapMeta>) {
    let mut ratings = rating::rate(judgments);
    let clusters = group_by_task(judgments);
    if clusters.len() < 2 || ratings.is_empty() || ratings.iter().any(|r| r.rating.is_none()) {
        return (ratings, None);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = clusters.len();
    let mut valid_samples = Vec::with_capacity(BOOTSTRAP_REPLICATES as usize);

    for _ in 0..BOOTSTRAP_REPLICATES {
        let draws: Vec<usize> = (0..n_clusters)
            .map(|_| rng.gen_range(0..n_clusters))
            .collect();
        let sampled = build_replicate(&clusters, &draws);
        if let Some(values) = finite_for_all(&ratings, &rating::rate(&sampled)) {
            valid_samples.push(values);
        }
    }

    let valid = valid_samples.len() as u32;
    if valid == BOOTSTRAP_REPLICATES {
        assign_bounds(&mut ratings, &valid_samples);
    }

    (
        ratings,
        Some(BootstrapMeta {
            seed,
            replicates: BOOTSTRAP_REPLICATES,
            valid,
        }),
    )
}

fn group_by_task(judgments: &[Judgment]) -> Vec<Vec<Judgment>> {
    let mut grouped: BTreeMap<&str, Vec<Judgment>> = BTreeMap::new();
    for judgment in judgments {
        grouped
            .entry(judgment.task_id.as_str())
            .or_default()
            .push(judgment.clone());
    }
    grouped.into_values().collect()
}

fn build_replicate(clusters: &[Vec<Judgment>], draws: &[usize]) -> Vec<Judgment> {
    let mut sampled = Vec::new();
    for &index in draws {
        sampled.extend(clusters[index].iter().cloned());
    }
    sampled
}

fn finite_for_all(original: &[ModelRating], replicate: &[ModelRating]) -> Option<Vec<f64>> {
    original
        .iter()
        .map(|orig| {
            replicate
                .iter()
                .find(|rating| rating.model == orig.model)
                .and_then(|rating| rating.rating)
        })
        .collect()
}

fn assign_bounds(ratings: &mut [ModelRating], samples: &[Vec<f64>]) {
    for (index, rating) in ratings.iter_mut().enumerate() {
        let column: Vec<f64> = samples.iter().map(|sample| sample[index]).collect();
        rating.rating_lower = Some(percentile(&column, LOWER_P));
        rating.rating_upper = Some(percentile(&column, UPPER_P));
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let mut xs = values.to_vec();
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    let index = (n - 1) as f64 * p;
    let lo = index.floor() as usize;
    let hi = (index.ceil() as usize).min(n - 1);
    let weight = index - lo as f64;
    xs[lo].mul_add(1.0 - weight, xs[hi] * weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::JudgeDecision;
    use crate::provider::ModelId;

    fn judgment(task: &str, model_a: &str, model_b: &str, winner: JudgeDecision) -> Judgment {
        Judgment {
            task_id: task.into(),
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

    fn cycle(task: &str) -> Vec<Judgment> {
        vec![
            judgment(task, "a", "b", JudgeDecision::A),
            judgment(task, "b", "c", JudgeDecision::A),
            judgment(task, "c", "a", JudgeDecision::A),
        ]
    }

    fn draws(task: &str) -> Vec<Judgment> {
        vec![
            judgment(task, "a", "b", JudgeDecision::Draw),
            judgment(task, "a", "c", JudgeDecision::Draw),
            judgment(task, "b", "c", JudgeDecision::Draw),
        ]
    }

    fn hierarchy(task: &str) -> Vec<Judgment> {
        vec![
            judgment(task, "a", "b", JudgeDecision::A),
            judgment(task, "a", "c", JudgeDecision::A),
            judgment(task, "b", "c", JudgeDecision::A),
        ]
    }

    fn rating_of(ratings: &[ModelRating], model: &str) -> Option<f64> {
        ratings
            .iter()
            .find(|r| r.model == ModelId::new(model))
            .and_then(|r| r.rating)
    }

    fn all_bounds_absent(ratings: &[ModelRating]) -> bool {
        ratings
            .iter()
            .all(|r| r.rating_lower.is_none() && r.rating_upper.is_none())
    }

    #[test]
    fn cluster_multiplicity_repeats_every_judgment_in_a_selected_task() {
        let mut judgments = cycle("t1");
        judgments.extend(draws("t2"));
        let clusters = group_by_task(&judgments);
        assert_eq!(clusters.len(), 2);

        let first = &clusters[0];
        let sampled = build_replicate(&clusters, &[0, 0]);
        assert_eq!(sampled.len(), first.len() * 2);
        assert_eq!(&sampled[..first.len()], first.as_slice());
        assert_eq!(&sampled[first.len()..], first.as_slice());
        assert!(sampled.iter().all(|j| j.task_id == first[0].task_id));
    }

    #[test]
    fn task_integrity_keeps_a_task_cluster_together() {
        let mut judgments = cycle("t1");
        judgments.extend(draws("t2"));
        let clusters = group_by_task(&judgments);
        assert_eq!(build_replicate(&clusters, &[0]), clusters[0]);
        assert_eq!(build_replicate(&clusters, &[1]), clusters[1]);

        let sampled = build_replicate(&clusters, &[0, 1]);
        assert_eq!(&sampled[..clusters[0].len()], clusters[0].as_slice());
        assert_eq!(&sampled[clusters[0].len()..], clusters[1].as_slice());
    }

    #[test]
    fn orientation_fields_do_not_change_bootstrap_input() {
        let mut judgments = cycle("t1");
        judgments.extend(draws("t2"));
        let mut oriented = judgments.clone();
        for judgment in &mut oriented {
            judgment.orientation_ab = Some(JudgeDecision::B);
            judgment.orientation_ba = Some(JudgeDecision::A);
            judgment.agreement = false;
        }

        let clusters = group_by_task(&judgments);
        let oriented_clusters = group_by_task(&oriented);
        assert_eq!(
            build_replicate(&clusters, &[0, 1])
                .iter()
                .map(|j| (&j.task_id, &j.model_a, &j.model_b, &j.winner))
                .collect::<Vec<_>>(),
            build_replicate(&oriented_clusters, &[0, 1])
                .iter()
                .map(|j| (&j.task_id, &j.model_a, &j.model_b, &j.winner))
                .collect::<Vec<_>>(),
        );

        assert_eq!(
            rate_with_uncertainty(&judgments, 0),
            rate_with_uncertainty(&oriented, 0)
        );
    }

    #[test]
    fn one_task_keeps_point_ratings_without_bounds() {
        let judgments = draws("t1");
        let (ratings, meta) = rate_with_uncertainty(&judgments, 0);
        assert_eq!(rating::rate(&judgments), ratings);
        assert_eq!(rating_of(&ratings, "a"), Some(1500.0));
        assert!(all_bounds_absent(&ratings));
        assert_eq!(meta, None);
    }

    #[test]
    fn original_invalid_ratings_skip_bootstrap() {
        let judgments = [
            judgment("t1", "a", "b", JudgeDecision::A),
            judgment("t2", "c", "d", JudgeDecision::A),
        ];
        let (ratings, meta) = rate_with_uncertainty(&judgments, 0);
        assert_eq!(rating::rate(&judgments), ratings);
        assert!(ratings.iter().all(|r| r.rating.is_none()));
        assert!(all_bounds_absent(&ratings));
        assert_eq!(meta, None);
    }

    #[test]
    fn invalid_replicate_omits_all_bounds_and_keeps_point_ratings() {
        let mut judgments = cycle("t1");
        judgments.extend(hierarchy("t2"));
        let point = rating::rate(&judgments);
        assert!(point.iter().all(|r| r.rating.is_some()));

        let (ratings, meta) = rate_with_uncertainty(&judgments, 0);
        let meta = meta.expect("bootstrap should run");
        assert_eq!(meta.seed, 0);
        assert_eq!(meta.replicates, BOOTSTRAP_REPLICATES);
        assert!(meta.valid < BOOTSTRAP_REPLICATES);
        assert_eq!(
            ratings
                .iter()
                .map(|r| (r.model.clone(), r.rating))
                .collect::<Vec<_>>(),
            point
                .iter()
                .map(|r| (r.model.clone(), r.rating))
                .collect::<Vec<_>>()
        );
        assert!(all_bounds_absent(&ratings));
    }

    #[test]
    fn same_seed_is_deterministic() {
        let mut judgments = cycle("t1");
        judgments.extend(draws("t2"));
        let first = rate_with_uncertainty(&judgments, 0);
        let second = rate_with_uncertainty(&judgments, 0);
        assert_eq!(first, second);
        let meta = first.1.expect("bootstrap should run");
        assert_eq!(meta.seed, 0);
        assert_eq!(meta.replicates, BOOTSTRAP_REPLICATES);
        assert_eq!(meta.valid, BOOTSTRAP_REPLICATES);
        assert!(
            first
                .0
                .iter()
                .all(|r| r.rating_lower.is_some() && r.rating_upper.is_some())
        );
    }

    #[test]
    fn percentile_uses_sorted_linear_interpolation() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 4.0);
        assert_eq!(percentile(&values, 0.5), 2.5);
        assert_eq!(percentile(&values, 0.25), 1.75);
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5), 2.5);
    }
}
