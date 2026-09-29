use std::fmt;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::evaluate::EvaluatedResult;
use crate::judge::{self, JudgeDecision, Judgment, JudgmentFailure};
use crate::provider::{ModelId, ModelProvider};
use crate::task::Task;

/// Which candidate pairs a run schedules.
///
/// Round-robin is the historical `exec` behavior: every unordered pair once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TournamentFormat {
    #[default]
    RoundRobin,
    SingleElimination,
}

impl TournamentFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round-robin",
            Self::SingleElimination => "single-elimination",
        }
    }
}

impl fmt::Display for TournamentFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TournamentStatus {
    Complete,
    Draw,
    Incomplete,
    NotJudged,
}

impl TournamentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Draw => "draw",
            Self::Incomplete => "incomplete",
            Self::NotJudged => "not judged",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchOutcome {
    Winner,
    Draw,
    JudgmentFailed,
}

/// One scheduled match and how it resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TournamentMatch {
    pub round: u32,
    pub model_a: ModelId,
    pub model_b: ModelId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<ModelId>,
    pub outcome: MatchOutcome,
}

/// Bracket or round-robin result for a single task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskBracket {
    pub task_id: String,
    pub status: TournamentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<ModelId>,
    pub matches: Vec<TournamentMatch>,
}

/// Format, field, and per-task match results for one experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tournament {
    pub format: TournamentFormat,
    pub candidates: Vec<ModelId>,
    pub status: TournamentStatus,
    pub tasks: Vec<TaskBracket>,
}

impl Tournament {
    pub fn judged_match_count(&self) -> usize {
        self.tasks.iter().map(|task| task.matches.len()).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BracketStep {
    Advance(ModelId),
    Draw,
    JudgmentFailed,
}

#[derive(Debug)]
struct RoundFold {
    matches: Vec<TournamentMatch>,
    winners: Vec<ModelId>,
    stop: bool,
}

pub fn validate(format: TournamentFormat, candidates: usize) -> Result<()> {
    if format == TournamentFormat::SingleElimination && !supported_field(candidates) {
        return Err(Error::UnsupportedTournament {
            format: format.as_str(),
            candidates,
        });
    }
    Ok(())
}

fn supported_field(candidates: usize) -> bool {
    candidates >= 2 && candidates.is_power_of_two()
}

pub fn planned_match_count(format: TournamentFormat, candidates: usize) -> usize {
    match format {
        TournamentFormat::RoundRobin => candidates.saturating_sub(1).saturating_mul(candidates) / 2,
        TournamentFormat::SingleElimination => candidates.saturating_sub(1),
    }
}

pub(crate) fn opening_pairs(
    format: TournamentFormat,
    candidates: &[ModelId],
) -> Vec<(ModelId, ModelId)> {
    match format {
        TournamentFormat::RoundRobin => round_robin_pairs(candidates),
        TournamentFormat::SingleElimination => {
            if supported_field(candidates.len()) {
                plan_round(candidates).unwrap_or_default()
            } else {
                Vec::new()
            }
        }
    }
}

pub(crate) fn not_judged(format: TournamentFormat, candidates: &[ModelId]) -> Tournament {
    Tournament {
        format,
        candidates: candidates.to_vec(),
        status: TournamentStatus::NotJudged,
        tasks: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn play<P, F, R>(
    provider: &P,
    judge_model: &ModelId,
    tasks: &[Task],
    results: &[EvaluatedResult],
    candidates: &[ModelId],
    format: TournamentFormat,
    mut on_judgment: F,
    mut on_round: R,
) -> Result<Tournament>
where
    P: ModelProvider + Clone + Send + 'static,
    F: FnMut(&Judgment),
    R: FnMut(&[JudgmentFailure]),
{
    validate(format, candidates.len())?;
    let mut brackets = Vec::with_capacity(tasks.len());
    for task in tasks {
        let bracket = match format {
            TournamentFormat::RoundRobin => {
                play_round_robin(
                    provider,
                    judge_model,
                    task,
                    results,
                    candidates,
                    &mut on_judgment,
                    &mut on_round,
                )
                .await?
            }
            TournamentFormat::SingleElimination => {
                play_single_elimination(
                    provider,
                    judge_model,
                    task,
                    results,
                    candidates,
                    &mut on_judgment,
                    &mut on_round,
                )
                .await?
            }
        };
        brackets.push(bracket);
    }
    Ok(assemble(format, candidates, brackets))
}

async fn play_round_robin<P, F, R>(
    provider: &P,
    judge_model: &ModelId,
    task: &Task,
    results: &[EvaluatedResult],
    candidates: &[ModelId],
    on_judgment: &mut F,
    on_round: &mut R,
) -> Result<TaskBracket>
where
    P: ModelProvider + Clone + Send + 'static,
    F: FnMut(&Judgment),
    R: FnMut(&[JudgmentFailure]),
{
    let pairs = round_robin_pairs(candidates);
    let listed = listed_pairs(&task.id, results, &pairs)?;
    let outcome = judge::judge_listed_pairs(
        provider,
        judge_model.clone(),
        task,
        &listed,
        &mut *on_judgment,
    )
    .await?;
    on_round(&outcome.failures);
    let matches = round_robin_matches(&pairs, &outcome.judgments, &outcome.failures)?;
    let status = if matches
        .iter()
        .any(|row| row.outcome == MatchOutcome::JudgmentFailed)
    {
        TournamentStatus::Incomplete
    } else {
        TournamentStatus::Complete
    };
    Ok(TaskBracket {
        task_id: task.id.clone(),
        status,
        winner: None,
        matches,
    })
}

async fn play_single_elimination<P, F, R>(
    provider: &P,
    judge_model: &ModelId,
    task: &Task,
    results: &[EvaluatedResult],
    candidates: &[ModelId],
    on_judgment: &mut F,
    on_round: &mut R,
) -> Result<TaskBracket>
where
    P: ModelProvider + Clone + Send + 'static,
    F: FnMut(&Judgment),
    R: FnMut(&[JudgmentFailure]),
{
    let mut active = candidates.to_vec();
    let mut matches = Vec::new();
    let mut round = 1u32;
    let mut stopped = false;
    while active.len() >= 2 {
        let pairs = plan_round(&active)?;
        let listed = listed_pairs(&task.id, results, &pairs)?;
        let outcome = judge::judge_listed_pairs(
            provider,
            judge_model.clone(),
            task,
            &listed,
            &mut *on_judgment,
        )
        .await?;
        on_round(&outcome.failures);
        let steps = steps_for_pairs(&pairs, &outcome.judgments, &outcome.failures)?;
        let folded = fold_round(round, &pairs, &steps)?;
        stopped = folded.stop;
        matches.extend(folded.matches);
        if stopped {
            break;
        }
        active = folded.winners;
        round = round.saturating_add(1);
    }
    Ok(bracket_from_parts(&task.id, matches, stopped, &active))
}

fn round_robin_pairs(candidates: &[ModelId]) -> Vec<(ModelId, ModelId)> {
    candidates
        .iter()
        .enumerate()
        .flat_map(|(index, model_a)| {
            candidates
                .iter()
                .skip(index + 1)
                .map(move |model_b| (model_a.clone(), model_b.clone()))
        })
        .collect()
}

fn plan_round(active: &[ModelId]) -> Result<Vec<(ModelId, ModelId)>> {
    if active.len() < 2 || !active.len().is_multiple_of(2) {
        return Err(Error::UnsupportedTournament {
            format: TournamentFormat::SingleElimination.as_str(),
            candidates: active.len(),
        });
    }
    Ok(active
        .chunks(2)
        .filter_map(|pair| match pair {
            [model_a, model_b] => Some((model_a.clone(), model_b.clone())),
            _ => None,
        })
        .collect())
}

fn listed_pairs(
    task_id: &str,
    results: &[EvaluatedResult],
    pairs: &[(ModelId, ModelId)],
) -> Result<Vec<(EvaluatedResult, EvaluatedResult)>> {
    let mut listed = Vec::with_capacity(pairs.len());
    for (model_a, model_b) in pairs {
        listed.push((
            find_result(results, task_id, model_a)?,
            find_result(results, task_id, model_b)?,
        ));
    }
    Ok(listed)
}

fn find_result(
    results: &[EvaluatedResult],
    task_id: &str,
    model: &ModelId,
) -> Result<EvaluatedResult> {
    results
        .iter()
        .find(|result| result.task_id == task_id && result.model == *model)
        .cloned()
        .ok_or_else(|| Error::MissingCandidateResult {
            task_id: task_id.to_string(),
            model: model.clone(),
        })
}

fn round_robin_matches(
    pairs: &[(ModelId, ModelId)],
    judgments: &[Judgment],
    failures: &[JudgmentFailure],
) -> Result<Vec<TournamentMatch>> {
    let mut matches = Vec::with_capacity(pairs.len());
    for (model_a, model_b) in pairs {
        matches.push(match_from_outcome(
            1, model_a, model_b, judgments, failures,
        )?);
    }
    Ok(matches)
}

fn steps_for_pairs(
    pairs: &[(ModelId, ModelId)],
    judgments: &[Judgment],
    failures: &[JudgmentFailure],
) -> Result<Vec<BracketStep>> {
    let mut steps = Vec::with_capacity(pairs.len());
    for (model_a, model_b) in pairs {
        let row = match_from_outcome(1, model_a, model_b, judgments, failures)?;
        steps.push(match row.outcome {
            MatchOutcome::JudgmentFailed => BracketStep::JudgmentFailed,
            MatchOutcome::Draw => BracketStep::Draw,
            MatchOutcome::Winner => {
                BracketStep::Advance(row.winner.ok_or_else(|| Error::InvalidTournamentWinner {
                    model_a: model_a.clone(),
                    model_b: model_b.clone(),
                    winner: model_a.clone(),
                })?)
            }
        });
    }
    Ok(steps)
}

fn match_from_outcome(
    round: u32,
    model_a: &ModelId,
    model_b: &ModelId,
    judgments: &[Judgment],
    failures: &[JudgmentFailure],
) -> Result<TournamentMatch> {
    if failures
        .iter()
        .any(|failure| failure.model_a == *model_a && failure.model_b == *model_b)
    {
        return Ok(TournamentMatch {
            round,
            model_a: model_a.clone(),
            model_b: model_b.clone(),
            winner: None,
            outcome: MatchOutcome::JudgmentFailed,
        });
    }
    let Some(judgment) = judgments
        .iter()
        .find(|judgment| judgment.model_a == *model_a && judgment.model_b == *model_b)
    else {
        return Err(Error::InconsistentPairCoverage {
            expected: 1,
            resolved: judgments.len(),
            failed: failures.len(),
        });
    };
    let (winner, outcome) = match judgment.winner {
        JudgeDecision::A => (Some(model_a.clone()), MatchOutcome::Winner),
        JudgeDecision::B => (Some(model_b.clone()), MatchOutcome::Winner),
        JudgeDecision::Draw => (None, MatchOutcome::Draw),
    };
    Ok(TournamentMatch {
        round,
        model_a: model_a.clone(),
        model_b: model_b.clone(),
        winner,
        outcome,
    })
}

fn fold_round(
    round: u32,
    pairs: &[(ModelId, ModelId)],
    steps: &[BracketStep],
) -> Result<RoundFold> {
    if pairs.len() != steps.len() {
        let failed = steps
            .iter()
            .filter(|step| matches!(step, BracketStep::JudgmentFailed))
            .count();
        return Err(Error::InconsistentPairCoverage {
            expected: pairs.len(),
            resolved: steps.len().saturating_sub(failed),
            failed,
        });
    }
    let mut matches = Vec::with_capacity(pairs.len());
    let mut winners = Vec::new();
    let mut stop = false;
    for ((model_a, model_b), step) in pairs.iter().zip(steps) {
        let (winner, outcome) = match step {
            BracketStep::Advance(winner) => {
                if winner != model_a && winner != model_b {
                    return Err(Error::InvalidTournamentWinner {
                        model_a: model_a.clone(),
                        model_b: model_b.clone(),
                        winner: winner.clone(),
                    });
                }
                (Some(winner.clone()), MatchOutcome::Winner)
            }
            BracketStep::Draw => (None, MatchOutcome::Draw),
            BracketStep::JudgmentFailed => (None, MatchOutcome::JudgmentFailed),
        };
        if outcome == MatchOutcome::Winner {
            if let Some(winner) = &winner {
                winners.push(winner.clone());
            }
        } else {
            stop = true;
        }
        matches.push(TournamentMatch {
            round,
            model_a: model_a.clone(),
            model_b: model_b.clone(),
            winner,
            outcome,
        });
    }
    if stop {
        winners.clear();
    }
    Ok(RoundFold {
        matches,
        winners,
        stop,
    })
}

fn bracket_from_parts(
    task_id: &str,
    matches: Vec<TournamentMatch>,
    stopped: bool,
    active: &[ModelId],
) -> TaskBracket {
    let (status, winner) = conclude(stopped, &matches, active);
    TaskBracket {
        task_id: task_id.to_string(),
        status,
        winner,
        matches,
    }
}

fn conclude(
    stopped: bool,
    matches: &[TournamentMatch],
    active: &[ModelId],
) -> (TournamentStatus, Option<ModelId>) {
    if matches
        .iter()
        .any(|row| row.outcome == MatchOutcome::JudgmentFailed)
    {
        return (TournamentStatus::Incomplete, None);
    }
    if stopped || matches.iter().any(|row| row.outcome == MatchOutcome::Draw) {
        return (TournamentStatus::Draw, None);
    }
    match active {
        [winner] => (TournamentStatus::Complete, Some(winner.clone())),
        _ => (TournamentStatus::Incomplete, None),
    }
}

fn assemble(
    format: TournamentFormat,
    candidates: &[ModelId],
    tasks: Vec<TaskBracket>,
) -> Tournament {
    let status = if tasks
        .iter()
        .any(|task| task.status == TournamentStatus::Incomplete)
    {
        TournamentStatus::Incomplete
    } else if tasks
        .iter()
        .any(|task| task.status == TournamentStatus::Draw)
    {
        TournamentStatus::Draw
    } else {
        TournamentStatus::Complete
    };
    Tournament {
        format,
        candidates: candidates.to_vec(),
        status,
        tasks,
    }
}

#[cfg(test)]
fn replay_single_elimination(
    task_id: &str,
    candidates: &[ModelId],
    mut play_match: impl FnMut(u32, &ModelId, &ModelId) -> BracketStep,
) -> Result<TaskBracket> {
    validate(TournamentFormat::SingleElimination, candidates.len())?;
    let mut active = candidates.to_vec();
    let mut matches = Vec::new();
    let mut round = 1u32;
    let mut stopped = false;
    while active.len() >= 2 {
        let pairs = plan_round(&active)?;
        let steps: Vec<_> = pairs
            .iter()
            .map(|(model_a, model_b)| play_match(round, model_a, model_b))
            .collect();
        let folded = fold_round(round, &pairs, &steps)?;
        stopped = folded.stop;
        matches.extend(folded.matches);
        if stopped {
            break;
        }
        active = folded.winners;
        round = round.saturating_add(1);
    }
    Ok(bracket_from_parts(task_id, matches, stopped, &active))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn ids(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::new(*name)).collect()
    }

    fn id(name: &str) -> ModelId {
        ModelId::new(name)
    }

    #[test]
    fn round_robin_pairs_follow_candidate_order() {
        let pairs = round_robin_pairs(&ids(&["A", "B", "C", "D"]));
        assert_eq!(
            pairs,
            vec![
                (id("A"), id("B")),
                (id("A"), id("C")),
                (id("A"), id("D")),
                (id("B"), id("C")),
                (id("B"), id("D")),
                (id("C"), id("D")),
            ]
        );
    }

    #[test]
    fn round_robin_pairs_are_unique() {
        let pairs = round_robin_pairs(&ids(&["A", "B", "C", "D"]));
        let mut seen = HashSet::new();
        for (model_a, model_b) in &pairs {
            assert_ne!(model_a, model_b);
            assert!(seen.insert((model_a.clone(), model_b.clone())));
            assert!(
                !seen.contains(&(model_b.clone(), model_a.clone())),
                "swapped pair was also scheduled"
            );
        }
        assert_eq!(seen.len(), pairs.len());
    }

    #[test]
    fn round_robin_pair_count_is_the_unordered_combination() {
        assert_eq!(planned_match_count(TournamentFormat::RoundRobin, 0), 0);
        assert_eq!(planned_match_count(TournamentFormat::RoundRobin, 1), 0);
        assert_eq!(planned_match_count(TournamentFormat::RoundRobin, 2), 1);
        assert_eq!(planned_match_count(TournamentFormat::RoundRobin, 4), 6);
        assert_eq!(planned_match_count(TournamentFormat::RoundRobin, 5), 10);
        assert_eq!(round_robin_pairs(&ids(&["A", "B", "C", "D"])).len(), 6);
        assert_eq!(
            planned_match_count(TournamentFormat::SingleElimination, 4),
            3
        );
        assert_eq!(
            planned_match_count(TournamentFormat::SingleElimination, 8),
            7
        );
    }

    #[test]
    fn single_elimination_advances_match_winners_into_the_next_round() {
        let bracket =
            replay_single_elimination("t1", &ids(&["A", "B", "C", "D"]), |round, a, b| {
                if round == 1 && a == &id("A") {
                    BracketStep::Advance(a.clone())
                } else if round == 1 {
                    BracketStep::Advance(b.clone())
                } else {
                    BracketStep::Advance(a.clone())
                }
            })
            .unwrap();

        assert_eq!(bracket.matches[0].winner, Some(id("A")));
        assert_eq!(bracket.matches[1].winner, Some(id("D")));
        assert_eq!(bracket.matches[2].round, 2);
        assert_eq!(bracket.matches[2].model_a, id("A"));
        assert_eq!(bracket.matches[2].model_b, id("D"));
    }

    #[test]
    fn single_elimination_propagates_the_champion() {
        let bracket =
            replay_single_elimination("t1", &ids(&["A", "B", "C", "D"]), |round, a, b| {
                if round == 1 && a == &id("A") {
                    BracketStep::Advance(a.clone())
                } else if round == 1 {
                    BracketStep::Advance(b.clone())
                } else {
                    BracketStep::Advance(a.clone())
                }
            })
            .unwrap();

        assert_eq!(bracket.status, TournamentStatus::Complete);
        assert_eq!(bracket.winner, Some(id("A")));
        assert_eq!(bracket.matches.len(), 3);
        assert_eq!(bracket.matches[2].winner, Some(id("A")));
        assert_eq!(bracket.matches[2].outcome, MatchOutcome::Winner);
    }

    #[test]
    fn draw_does_not_advance_or_invent_a_winner() {
        let bracket =
            replay_single_elimination("t1", &ids(&["A", "B", "C", "D"]), |round, a, b| {
                assert_eq!(round, 1, "a draw must not schedule the next round");
                if a == &id("A") {
                    BracketStep::Draw
                } else {
                    BracketStep::Advance(b.clone())
                }
            })
            .unwrap();

        assert_eq!(bracket.status, TournamentStatus::Draw);
        assert!(bracket.winner.is_none());
        assert_eq!(bracket.matches.len(), 2);
        assert!(bracket.matches.iter().all(|row| row.round == 1));
        assert_eq!(bracket.matches[0].outcome, MatchOutcome::Draw);
        assert!(bracket.matches[0].winner.is_none());
        assert_eq!(bracket.matches[1].winner, Some(id("D")));
    }

    #[test]
    fn draw_in_the_final_leaves_no_champion() {
        let bracket =
            replay_single_elimination("t1", &ids(&["A", "B", "C", "D"]), |round, a, _| {
                if round == 1 {
                    BracketStep::Advance(a.clone())
                } else {
                    BracketStep::Draw
                }
            })
            .unwrap();

        assert_eq!(bracket.status, TournamentStatus::Draw);
        assert!(bracket.winner.is_none());
        assert_eq!(bracket.matches.len(), 3);
        assert_eq!(bracket.matches[2].round, 2);
        assert_eq!(bracket.matches[2].outcome, MatchOutcome::Draw);
        assert!(bracket.matches[2].winner.is_none());
    }

    #[test]
    fn judgment_failure_stops_the_bracket() {
        let bracket =
            replay_single_elimination("t1", &ids(&["A", "B", "C", "D"]), |round, a, b| {
                assert_eq!(round, 1);
                if a == &id("A") {
                    BracketStep::JudgmentFailed
                } else {
                    BracketStep::Advance(b.clone())
                }
            })
            .unwrap();

        assert_eq!(bracket.status, TournamentStatus::Incomplete);
        assert!(bracket.winner.is_none());
        assert_eq!(bracket.matches.len(), 2);
        assert_eq!(bracket.matches[0].outcome, MatchOutcome::JudgmentFailed);
        assert!(bracket.matches.iter().all(|row| row.round == 1));
    }

    #[test]
    fn winner_outside_the_match_is_rejected() {
        let error =
            fold_round(1, &[(id("A"), id("B"))], &[BracketStep::Advance(id("Z"))]).unwrap_err();
        match error {
            Error::InvalidTournamentWinner {
                model_a,
                model_b,
                winner,
            } => {
                assert_eq!(model_a, id("A"));
                assert_eq!(model_b, id("B"));
                assert_eq!(winner, id("Z"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn single_elimination_rejects_unsupported_field_sizes() {
        for candidates in [0, 1, 3, 5, 6, 7, 9, 12] {
            let error = validate(TournamentFormat::SingleElimination, candidates).unwrap_err();
            match error {
                Error::UnsupportedTournament {
                    format,
                    candidates: got,
                } => {
                    assert_eq!(format, "single-elimination");
                    assert_eq!(got, candidates);
                }
                other => panic!("unexpected error: {other}"),
            }
            assert!(
                replay_single_elimination("t1", &ids(&vec!["m"; candidates]), |_, a, _| {
                    BracketStep::Advance(a.clone())
                })
                .is_err()
            );
        }
        for candidates in [2, 4, 8, 16] {
            assert!(validate(TournamentFormat::SingleElimination, candidates).is_ok());
        }
        for candidates in [0, 1, 3, 6] {
            assert!(validate(TournamentFormat::RoundRobin, candidates).is_ok());
        }
    }

    #[test]
    fn tournament_record_round_trips() {
        let bracket = replay_single_elimination("t1", &ids(&["A", "B", "C", "D"]), |_, a, _| {
            BracketStep::Advance(a.clone())
        })
        .unwrap();
        let tournament = assemble(
            TournamentFormat::SingleElimination,
            &ids(&["A", "B", "C", "D"]),
            vec![bracket],
        );
        let json = serde_json::to_value(&tournament).unwrap();
        assert_eq!(json["format"], "single_elimination");
        assert_eq!(json["status"], "complete");
        assert_eq!(json["candidates"], serde_json::json!(["A", "B", "C", "D"]));
        assert_eq!(json["tasks"][0]["winner"], "A");
        assert_eq!(json["tasks"][0]["matches"][2]["round"], 2);
        let restored: Tournament = serde_json::from_value(json).unwrap();
        assert_eq!(restored, tournament);

        let drawn = replay_single_elimination("t1", &ids(&["A", "B"]), |_, _, _| BracketStep::Draw)
            .unwrap();
        let drawn = assemble(
            TournamentFormat::SingleElimination,
            &ids(&["A", "B"]),
            vec![drawn],
        );
        let value = serde_json::to_value(&drawn).unwrap();
        assert!(value["tasks"][0].get("winner").is_none());
        assert!(value["tasks"][0]["matches"][0].get("winner").is_none());
        assert_eq!(value["tasks"][0]["matches"][0]["outcome"], "draw");
        assert_eq!(value["status"], "draw");
        let restored: Tournament = serde_json::from_value(value).unwrap();
        assert_eq!(restored, drawn);
        assert!(restored.tasks[0].winner.is_none());
    }
}
