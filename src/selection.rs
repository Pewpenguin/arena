use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::error::{Error, Result};
use crate::exec;
use crate::provider::ModelId;

/// Default number of candidates Arena picks in autonomous mode.
pub const DEFAULT_CANDIDATE_COUNT: usize = 4;

/// Provenance when Arena chose participants autonomously.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionProvenance {
    /// Model universe Arena drew from (provider listing or workbench list).
    pub universe: Vec<ModelId>,
    pub selection_seed: u64,
}

/// Resolved candidates and judge after optional autonomous selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    pub models: Vec<ModelId>,
    pub judge: Option<ModelId>,
    pub provenance: Option<SelectionProvenance>,
}

/// Manual selection: user-supplied candidates and optional judge.
pub fn manual(models: Vec<String>, judge: Option<String>) -> Result<ResolvedSelection> {
    let models = exec::unique_models(trim_ids(models))?;
    if models.is_empty() {
        return Err(Error::NoCandidates);
    }
    let judge = judge
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(ModelId::new);
    exec::validate_judge(&models, judge.as_ref())?;
    Ok(ResolvedSelection {
        models,
        judge,
        provenance: None,
    })
}

/// Autonomous selection: Arena picks candidates and a judge from a known universe.
///
/// Selection happens once. When `selection_seed` is omitted, a seed is generated
/// for the run. The universe is trimmed and uniqueness-validated first; candidate
/// count is derived from that concrete universe (up to [`DEFAULT_CANDIDATE_COUNT`]).
/// The RNG stream shuffles the universe, then takes the candidates and the next
/// model as judge, so the judge is never a candidate. Universe order is part of
/// the deterministic input.
pub fn autonomous(universe: Vec<String>, selection_seed: Option<u64>) -> Result<ResolvedSelection> {
    let universe = exec::unique_models(trim_ids(universe))?;
    if universe.is_empty() {
        return Err(Error::EmptyModelUniverse);
    }
    let candidate_count = candidate_count_for(universe.len());
    if candidate_count == 0 {
        return Err(Error::UniverseTooSmall {
            needed: 2,
            available: universe.len(),
        });
    }

    let seed = selection_seed.unwrap_or_else(rand::random);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut order = universe.clone();
    shuffle(&mut order, &mut rng);

    let models: Vec<ModelId> = order.iter().take(candidate_count).cloned().collect();
    let judge = order
        .get(candidate_count)
        .cloned()
        .expect("universe sized for judge");
    exec::validate_judge(&models, Some(&judge))?;

    Ok(ResolvedSelection {
        models,
        judge: Some(judge),
        provenance: Some(SelectionProvenance {
            universe,
            selection_seed: seed,
        }),
    })
}

/// Candidate count Arena uses for a universe of the given size.
pub fn candidate_count_for(universe_len: usize) -> usize {
    if universe_len <= 1 {
        return 0;
    }
    DEFAULT_CANDIDATE_COUNT.min(universe_len - 1)
}

fn trim_ids(ids: Vec<String>) -> Vec<String> {
    ids.into_iter()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect()
}

fn shuffle(items: &mut [ModelId], rng: &mut StdRng) {
    for i in (1..items.len()).rev() {
        let j = rng.random_range(0..=i);
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_has_no_provenance() {
        let resolved = manual(vec!["a".into(), "b".into()], Some("judge".into())).unwrap();
        assert_eq!(resolved.models, vec![ModelId::new("a"), ModelId::new("b")]);
        assert_eq!(resolved.judge, Some(ModelId::new("judge")));
        assert!(resolved.provenance.is_none());
    }

    #[test]
    fn autonomous_is_reproducible_and_keeps_judge_out_of_candidates() {
        let universe = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
        let first = autonomous(universe.clone(), Some(7)).unwrap();
        let second = autonomous(universe, Some(7)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.models.len(), 4);
        let judge = first.judge.as_ref().unwrap();
        assert!(!first.models.contains(judge));
        let provenance = first.provenance.as_ref().unwrap();
        assert_eq!(provenance.selection_seed, 7);
        assert_eq!(provenance.universe.len(), 5);
    }

    #[test]
    fn autonomous_generates_and_persists_seed() {
        let resolved = autonomous(vec!["a".into(), "b".into(), "c".into()], None).unwrap();
        assert!(resolved.provenance.is_some());
        assert_eq!(resolved.models.len(), 2);
        assert!(resolved.judge.is_some());
    }

    #[test]
    fn autonomous_sizes_count_after_trimming_blanks() {
        // Raw length 5 would request 4 candidates; after trim only 3 models remain.
        let resolved = autonomous(
            vec!["a".into(), "b".into(), "c".into(), " ".into(), "".into()],
            Some(1),
        )
        .unwrap();
        assert_eq!(resolved.provenance.as_ref().unwrap().universe.len(), 3);
        assert_eq!(resolved.models.len(), 2);
        assert!(resolved.judge.is_some());
    }

    #[test]
    fn autonomous_rejects_universe_too_small() {
        let err = autonomous(vec!["a".into()], Some(1)).unwrap_err();
        assert!(matches!(
            err,
            Error::UniverseTooSmall {
                needed: 2,
                available: 1
            }
        ));
    }

    #[test]
    fn candidate_count_for_leaves_room_for_judge() {
        assert_eq!(candidate_count_for(0), 0);
        assert_eq!(candidate_count_for(2), 1);
        assert_eq!(candidate_count_for(5), 4);
        assert_eq!(candidate_count_for(20), DEFAULT_CANDIDATE_COUNT);
    }
}
