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
    Incomplete,
}

/// One game inside a best-of series.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeriesGame {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<ModelId>,
    pub outcome: MatchOutcome,
    /// Played after a drawn elimination series. Absent on regulation games and old JSON.
    #[serde(default, skip_serializing_if = "is_false")]
    pub tiebreak: bool,
}

pub const DEFAULT_BEST_OF: u32 = 1;

pub fn default_best_of() -> u32 {
    DEFAULT_BEST_OF
}

fn is_default_best_of(value: &u32) -> bool {
    *value == DEFAULT_BEST_OF
}

fn is_false(value: &bool) -> bool {
    !*value
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
    /// Individual games. Empty for a single-game match, which is the v1.3.0 shape.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub games: Vec<SeriesGame>,
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
    #[serde(
        default = "default_best_of",
        skip_serializing_if = "is_default_best_of"
    )]
    pub best_of: u32,
    pub tasks: Vec<TaskBracket>,
}

impl TournamentMatch {
    pub fn games_played(&self) -> usize {
        if self.games.is_empty() {
            1
        } else {
            self.games.len()
        }
    }
}

impl Tournament {
    pub fn series_count(&self) -> usize {
        self.tasks.iter().map(|task| task.matches.len()).sum()
    }

    /// Games submitted to the judge. A single-game match counts as one.
    pub fn judged_match_count(&self) -> usize {
        self.tasks
            .iter()
            .flat_map(|task| &task.matches)
            .map(TournamentMatch::games_played)
            .sum()
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum BracketStep {
    Advance(ModelId),
    Draw,
    JudgmentFailed,
}

#[cfg(test)]
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

pub fn validate_best_of(best_of: u32) -> Result<()> {
    if best_of >= 1 && !best_of.is_multiple_of(2) {
        Ok(())
    } else {
        Err(Error::InvalidBestOf { best_of })
    }
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

pub(crate) fn not_judged(
    format: TournamentFormat,
    candidates: &[ModelId],
    best_of: u32,
) -> Tournament {
    Tournament {
        format,
        candidates: candidates.to_vec(),
        status: TournamentStatus::NotJudged,
        best_of,
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
    best_of: u32,
    mut on_judgment: F,
    mut on_round: R,
) -> Result<Tournament>
where
    P: ModelProvider + Clone + Send + 'static,
    F: FnMut(&Judgment),
    R: FnMut(&[JudgmentFailure]),
{
    validate(format, candidates.len())?;
    validate_best_of(best_of)?;
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
                    best_of,
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
                    best_of,
                    &mut on_judgment,
                    &mut on_round,
                )
                .await?
            }
        };
        brackets.push(bracket);
    }
    Ok(assemble(format, candidates, best_of, brackets))
}

#[allow(clippy::too_many_arguments)]
async fn play_round_robin<P, F, R>(
    provider: &P,
    judge_model: &ModelId,
    task: &Task,
    results: &[EvaluatedResult],
    candidates: &[ModelId],
    best_of: u32,
    on_judgment: &mut F,
    on_round: &mut R,
) -> Result<TaskBracket>
where
    P: ModelProvider + Clone + Send + 'static,
    F: FnMut(&Judgment),
    R: FnMut(&[JudgmentFailure]),
{
    let pairs = round_robin_pairs(candidates);
    let matches = play_series(
        provider,
        judge_model,
        task,
        results,
        &pairs,
        1,
        best_of,
        false,
        &mut *on_judgment,
        &mut *on_round,
    )
    .await?;
    let status = if matches.iter().any(|row| {
        matches!(
            row.outcome,
            MatchOutcome::JudgmentFailed | MatchOutcome::Incomplete
        )
    }) {
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

#[allow(clippy::too_many_arguments)]
async fn play_single_elimination<P, F, R>(
    provider: &P,
    judge_model: &ModelId,
    task: &Task,
    results: &[EvaluatedResult],
    candidates: &[ModelId],
    best_of: u32,
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
        let played = play_series(
            provider,
            judge_model,
            task,
            results,
            &pairs,
            round,
            best_of,
            true,
            &mut *on_judgment,
            &mut *on_round,
        )
        .await?;
        let winners = series_winners(&played);
        stopped = winners.is_empty();
        matches.extend(played);
        if stopped {
            break;
        }
        active = winners;
        round = round.saturating_add(1);
    }
    Ok(bracket_from_parts(&task.id, matches, stopped, &active))
}

fn series_winners(matches: &[TournamentMatch]) -> Vec<ModelId> {
    let mut winners = Vec::with_capacity(matches.len());
    for row in matches {
        if row.outcome != MatchOutcome::Winner {
            return Vec::new();
        }
        let Some(winner) = &row.winner else {
            return Vec::new();
        };
        winners.push(winner.clone());
    }
    winners
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

#[allow(clippy::too_many_arguments)]
async fn play_series<P, F, R>(
    provider: &P,
    judge_model: &ModelId,
    task: &Task,
    results: &[EvaluatedResult],
    pairs: &[(ModelId, ModelId)],
    round: u32,
    best_of: u32,
    resolve_draws: bool,
    on_judgment: &mut F,
    on_round: &mut R,
) -> Result<Vec<TournamentMatch>>
where
    P: ModelProvider + Clone + Send + 'static,
    F: FnMut(&Judgment),
    R: FnMut(&[JudgmentFailure]),
{
    let mut open: Vec<OpenSeries> = pairs
        .iter()
        .map(|(model_a, model_b)| OpenSeries::new(round, model_a.clone(), model_b.clone()))
        .collect();
    for _ in 0..best_of {
        let pending: Vec<(ModelId, ModelId)> = open
            .iter()
            .filter(|series| series.outcome.is_none())
            .map(|series| (series.model_a.clone(), series.model_b.clone()))
            .collect();
        if pending.is_empty() {
            break;
        }
        let listed = listed_pairs(&task.id, results, &pending)?;
        let judged = judge::judge_listed_pairs(
            provider,
            judge_model.clone(),
            task,
            &listed,
            &mut *on_judgment,
        )
        .await?;
        on_round(&judged.failures);
        let mut halt = false;
        for series in open.iter_mut().filter(|series| series.outcome.is_none()) {
            apply_game(series, &judged.judgments, &judged.failures, best_of, false)?;
            if series.outcome == Some(MatchOutcome::JudgmentFailed) {
                halt = true;
            }
        }
        if halt {
            for series in &mut open {
                if series.outcome.is_none() {
                    series.outcome = Some(MatchOutcome::Incomplete);
                }
            }
            break;
        }
    }
    let blocked = open.iter().any(|series| {
        matches!(
            series.outcome,
            Some(MatchOutcome::JudgmentFailed | MatchOutcome::Incomplete)
        )
    });
    if resolve_draws && !blocked {
        loop {
            let pending: Vec<(ModelId, ModelId)> = open
                .iter()
                .filter(|series| series.outcome == Some(MatchOutcome::Draw))
                .map(|series| (series.model_a.clone(), series.model_b.clone()))
                .collect();
            if pending.is_empty() {
                break;
            }
            let listed = listed_pairs(&task.id, results, &pending)?;
            let judged = judge::judge_listed_pairs(
                provider,
                judge_model.clone(),
                task,
                &listed,
                &mut *on_judgment,
            )
            .await?;
            on_round(&judged.failures);
            let mut halt = false;
            for series in &mut open {
                if series.outcome != Some(MatchOutcome::Draw) {
                    continue;
                }
                apply_game(series, &judged.judgments, &judged.failures, best_of, true)?;
                if series.outcome == Some(MatchOutcome::Incomplete) {
                    halt = true;
                }
            }
            if halt {
                break;
            }
        }
    }
    Ok(open
        .into_iter()
        .map(|series| series.finish(best_of))
        .collect())
}

struct OpenSeries {
    round: u32,
    model_a: ModelId,
    model_b: ModelId,
    wins_a: u32,
    wins_b: u32,
    games: Vec<SeriesGame>,
    outcome: Option<MatchOutcome>,
    winner: Option<ModelId>,
}

impl OpenSeries {
    fn new(round: u32, model_a: ModelId, model_b: ModelId) -> Self {
        Self {
            round,
            model_a,
            model_b,
            wins_a: 0,
            wins_b: 0,
            games: Vec::new(),
            outcome: None,
            winner: None,
        }
    }

    fn finish(self, best_of: u32) -> TournamentMatch {
        let persist_games =
            best_of != DEFAULT_BEST_OF || self.games.iter().any(|game| game.tiebreak);
        TournamentMatch {
            round: self.round,
            model_a: self.model_a,
            model_b: self.model_b,
            winner: self.winner,
            outcome: self.outcome.unwrap_or(MatchOutcome::Incomplete),
            games: if persist_games {
                self.games
            } else {
                Vec::new()
            },
        }
    }
}

fn apply_game(
    series: &mut OpenSeries,
    judgments: &[Judgment],
    failures: &[JudgmentFailure],
    best_of: u32,
    tiebreak: bool,
) -> Result<()> {
    if failures
        .iter()
        .any(|failure| failure.model_a == series.model_a && failure.model_b == series.model_b)
    {
        series.games.push(SeriesGame {
            winner: None,
            outcome: MatchOutcome::JudgmentFailed,
            tiebreak,
        });
        series.outcome = Some(if tiebreak {
            MatchOutcome::Incomplete
        } else {
            MatchOutcome::JudgmentFailed
        });
        series.winner = None;
        return Ok(());
    }
    let Some(judgment) = judgments
        .iter()
        .find(|judgment| judgment.model_a == series.model_a && judgment.model_b == series.model_b)
    else {
        return Err(Error::InconsistentPairCoverage {
            expected: 1,
            resolved: judgments.len(),
            failed: failures.len(),
        });
    };
    let (game_winner, game_outcome) = match judgment.winner {
        JudgeDecision::A => {
            series.wins_a = series.wins_a.saturating_add(1);
            (Some(series.model_a.clone()), MatchOutcome::Winner)
        }
        JudgeDecision::B => {
            series.wins_b = series.wins_b.saturating_add(1);
            (Some(series.model_b.clone()), MatchOutcome::Winner)
        }
        JudgeDecision::Draw => (None, MatchOutcome::Draw),
    };
    series.games.push(SeriesGame {
        winner: game_winner.clone(),
        outcome: game_outcome,
        tiebreak,
    });
    if tiebreak {
        if game_outcome == MatchOutcome::Winner {
            series.outcome = Some(MatchOutcome::Winner);
            series.winner = game_winner;
        } else {
            series.outcome = Some(MatchOutcome::Draw);
            series.winner = None;
        }
        return Ok(());
    }
    if let Some(end) = series_result(
        series.wins_a,
        series.wins_b,
        series.games.len() as u32,
        best_of,
    ) {
        match end {
            SeriesEnd::WinnerA => {
                series.outcome = Some(MatchOutcome::Winner);
                series.winner = Some(series.model_a.clone());
            }
            SeriesEnd::WinnerB => {
                series.outcome = Some(MatchOutcome::Winner);
                series.winner = Some(series.model_b.clone());
            }
            SeriesEnd::Draw => {
                series.outcome = Some(MatchOutcome::Draw);
                series.winner = None;
            }
        }
    }
    Ok(())
}

enum SeriesEnd {
    WinnerA,
    WinnerB,
    Draw,
}

/// `None` means another game is still required.
fn series_result(wins_a: u32, wins_b: u32, played: u32, best_of: u32) -> Option<SeriesEnd> {
    let need = best_of / 2 + 1;
    let left = best_of.saturating_sub(played);
    if wins_a >= need {
        Some(SeriesEnd::WinnerA)
    } else if wins_b >= need {
        Some(SeriesEnd::WinnerB)
    } else if wins_a.saturating_add(left) < need && wins_b.saturating_add(left) < need {
        Some(SeriesEnd::Draw)
    } else {
        None
    }
}

#[cfg(test)]
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
            games: Vec::new(),
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
    if matches.iter().any(|row| {
        matches!(
            row.outcome,
            MatchOutcome::JudgmentFailed | MatchOutcome::Incomplete
        )
    }) {
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
    best_of: u32,
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
        best_of,
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

    use crate::judge::{JudgeDecision, Judgment};

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
            DEFAULT_BEST_OF,
            vec![bracket],
        );
        let json = serde_json::to_value(&tournament).unwrap();
        assert_eq!(json["format"], "single_elimination");
        assert_eq!(json["status"], "complete");
        assert!(json.get("best_of").is_none());
        assert!(json["tasks"][0]["matches"][0].get("games").is_none());
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
            DEFAULT_BEST_OF,
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
        assert_eq!(restored.best_of, DEFAULT_BEST_OF);
    }

    #[test]
    fn legacy_tournament_without_best_of_metadata_loads_as_a_single_game() {
        let legacy = r#"{"format":"round_robin","candidates":["a","b"],"status":"complete","tasks":[{"task_id":"t1","status":"complete","matches":[{"round":1,"model_a":"a","model_b":"b","winner":"a","outcome":"winner"}]}]}"#;
        let loaded: Tournament = serde_json::from_str(legacy).unwrap();
        assert_eq!(loaded.best_of, DEFAULT_BEST_OF);
        assert!(loaded.tasks[0].matches[0].games.is_empty());
        assert_eq!(loaded.tasks[0].matches[0].games_played(), 1);
    }

    #[test]
    fn best_of_series_stops_when_the_majority_is_decided() {
        let (outcome, winner, played) =
            replay_games(&[JudgeDecision::A, JudgeDecision::A, JudgeDecision::B], 3);
        assert_eq!(outcome, MatchOutcome::Winner);
        assert_eq!(winner, Some(id("A")));
        assert_eq!(played, 2);

        let (outcome, winner, played) = replay_games(
            &[
                JudgeDecision::B,
                JudgeDecision::B,
                JudgeDecision::B,
                JudgeDecision::A,
            ],
            5,
        );
        assert_eq!(outcome, MatchOutcome::Winner);
        assert_eq!(winner, Some(id("B")));
        assert_eq!(played, 3);

        let (outcome, winner, played) = replay_games(&[JudgeDecision::A], 1);
        assert_eq!(outcome, MatchOutcome::Winner);
        assert_eq!(winner, Some(id("A")));
        assert_eq!(played, 1);
    }

    #[test]
    fn draws_do_not_count_as_wins_and_can_end_the_series() {
        let (outcome, winner, played) = replay_games(
            &[JudgeDecision::A, JudgeDecision::Draw, JudgeDecision::A],
            3,
        );
        assert_eq!(outcome, MatchOutcome::Winner);
        assert_eq!(winner, Some(id("A")));
        assert_eq!(played, 3);

        let (outcome, winner, played) = replay_games(
            &[JudgeDecision::Draw, JudgeDecision::Draw, JudgeDecision::A],
            3,
        );
        assert_eq!(outcome, MatchOutcome::Draw);
        assert!(winner.is_none());
        assert_eq!(played, 2);

        let (outcome, winner, played) = replay_games(&[JudgeDecision::Draw], 1);
        assert_eq!(outcome, MatchOutcome::Draw);
        assert!(winner.is_none());
        assert_eq!(played, 1);
    }

    #[test]
    fn elimination_tiebreaks_resolve_a_drawn_series_for_any_best_of() {
        for best_of in [1, 3, 5] {
            let mut series = drawn_regulation_series(best_of);
            let regulation = series.games.len();
            apply_game(
                &mut series,
                &[scripted_judgment(JudgeDecision::Draw)],
                &[],
                best_of,
                true,
            )
            .unwrap();
            assert_eq!(series.outcome, Some(MatchOutcome::Draw));
            apply_game(
                &mut series,
                &[scripted_judgment(JudgeDecision::B)],
                &[],
                best_of,
                true,
            )
            .unwrap();
            assert_eq!(series.outcome, Some(MatchOutcome::Winner));
            assert_eq!(series.winner.as_ref(), Some(&id("B")));
            let finished = series.finish(best_of);
            assert_eq!(finished.round, 1);
            assert!(
                finished
                    .games
                    .iter()
                    .take(regulation)
                    .all(|game| !game.tiebreak)
            );
            assert!(
                finished
                    .games
                    .iter()
                    .skip(regulation)
                    .all(|game| game.tiebreak)
            );
            assert_eq!(finished.games_played(), regulation + 2);
        }
    }

    #[test]
    fn tiebreak_judgment_failure_leaves_the_elimination_match_incomplete() {
        use crate::judge::{JudgmentFailure, JudgmentFailureKind, OrientationFailure};

        let mut series = OpenSeries::new(1, id("A"), id("B"));
        apply_game(
            &mut series,
            &[scripted_judgment(JudgeDecision::Draw)],
            &[],
            1,
            false,
        )
        .unwrap();
        let failure = JudgmentFailure {
            task_id: "t1".into(),
            model_a: id("A"),
            model_b: id("B"),
            judge_model: id("judge"),
            orientations: vec![OrientationFailure {
                orientation: crate::judge::JudgeOrientation::Ab,
                kind: JudgmentFailureKind::Provider,
                error: "stopped".into(),
                attempts: 1,
            }],
            raw_ab: None,
            reason_ab: None,
            raw_ba: None,
            reason_ba: None,
        };
        apply_game(&mut series, &[], &[failure], 1, true).unwrap();
        assert_eq!(series.outcome, Some(MatchOutcome::Incomplete));
        assert!(series.winner.is_none());
        assert_eq!(series.games.len(), 2);
        assert!(!series.games[0].tiebreak);
        assert!(series.games[1].tiebreak);
        assert_eq!(series.games[1].outcome, MatchOutcome::JudgmentFailed);
    }

    #[test]
    fn tiebreak_games_round_trip_and_legacy_games_omit_the_flag() {
        let mut series = OpenSeries::new(1, id("A"), id("B"));
        apply_game(
            &mut series,
            &[scripted_judgment(JudgeDecision::Draw)],
            &[],
            1,
            false,
        )
        .unwrap();
        apply_game(
            &mut series,
            &[scripted_judgment(JudgeDecision::A)],
            &[],
            1,
            true,
        )
        .unwrap();
        let finished = series.finish(1);
        let json = serde_json::to_value(&finished).unwrap();
        assert!(json["games"][0].get("tiebreak").is_none());
        assert_eq!(json["games"][1]["tiebreak"], true);
        assert_eq!(json["round"], 1);
        assert_eq!(json["winner"], "A");
        let restored: TournamentMatch = serde_json::from_value(json).unwrap();
        assert_eq!(restored, finished);

        let legacy: SeriesGame =
            serde_json::from_str(r#"{"winner":"A","outcome":"winner"}"#).unwrap();
        assert!(!legacy.tiebreak);
        assert_eq!(legacy.winner, Some(id("A")));
    }

    fn drawn_regulation_series(best_of: u32) -> OpenSeries {
        let mut series = OpenSeries::new(1, id("A"), id("B"));
        for _ in 0..best_of {
            if series.outcome.is_some() {
                break;
            }
            apply_game(
                &mut series,
                &[scripted_judgment(JudgeDecision::Draw)],
                &[],
                best_of,
                false,
            )
            .unwrap();
        }
        assert_eq!(series.outcome, Some(MatchOutcome::Draw));
        assert!(series.games.iter().all(|game| !game.tiebreak));
        series
    }

    fn scripted_judgment(winner: JudgeDecision) -> Judgment {
        Judgment {
            task_id: "t1".into(),
            model_a: id("A"),
            model_b: id("B"),
            judge_model: id("judge"),
            winner,
            reason: String::new(),
            duration_ms: 1,
            agreement: true,
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
    fn best_of_rejects_even_and_zero_lengths() {
        for best_of in [0, 2, 4, 6] {
            let error = validate_best_of(best_of).unwrap_err();
            match error {
                Error::InvalidBestOf { best_of: got } => assert_eq!(got, best_of),
                other => panic!("unexpected error: {other}"),
            }
        }
        for best_of in [1, 3, 5] {
            assert!(validate_best_of(best_of).is_ok());
        }
    }

    fn replay_games(
        games: &[JudgeDecision],
        best_of: u32,
    ) -> (MatchOutcome, Option<ModelId>, usize) {
        let mut series = OpenSeries::new(1, id("A"), id("B"));
        let mut played = 0;
        for game in games {
            played += 1;
            let judgment = Judgment {
                task_id: "t1".into(),
                model_a: id("A"),
                model_b: id("B"),
                judge_model: id("judge"),
                winner: game.clone(),
                reason: String::new(),
                duration_ms: 1,
                agreement: true,
                orientation_ab: None,
                orientation_ba: None,
                reason_ab: None,
                reason_ba: None,
                raw_ab: None,
                raw_ba: None,
                raw: None,
            };
            apply_game(&mut series, &[judgment], &[], best_of, false).unwrap();
            if series.outcome.is_some() {
                break;
            }
        }
        let outcome = series.outcome.unwrap_or(MatchOutcome::Incomplete);
        let winner = series.winner.clone();
        (outcome, winner, played)
    }
}
