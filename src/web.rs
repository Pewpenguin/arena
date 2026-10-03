use std::convert::Infallible;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path as FilePath, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, broadcast, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::{Error, Result};
use crate::event::ExperimentEvent;
use crate::exec::{self, ExecConfig};
use crate::html;
use crate::judge::{Judgment, JudgmentFailure};
use crate::persist;
use crate::provider::{
    AnthropicProvider, GeminiProvider, ModelId, ModelProvider, OpenAICompatibleProvider,
};
use crate::report;
use crate::task::{self, Task};
use crate::tournament::{self, OpeningMatchup, Tournament, TournamentFormat, TournamentStatus};

const BIND_HOST: [u8; 4] = [127, 0, 0, 1];
const WEB_OUTPUT_DIR: &str = "arena-web";

struct AppState {
    session: Mutex<Option<ProviderSession>>,
    next_run: AtomicU64,
    run: Mutex<Option<Arc<LiveRun>>>,
}

struct ProviderSession {
    kind: WebProviderKind,
    api_key: String,
    base_url: String,
}

struct LiveRun {
    id: String,
    finished: AtomicBool,
    inner: Mutex<RunState>,
    events: broadcast::Sender<ClientEvent>,
}

struct RunState {
    seq: u64,
    status: RunStatus,
    error: Option<String>,
    candidate_count: usize,
    task_count: usize,
    judge: Option<String>,
    started_at: String,
    candidate_completed: usize,
    candidate_total: usize,
    /// Regulation-game ceiling used for live progress (series × tasks × best-of).
    planned_games: usize,
    /// During a run, matches [`planned_games`]. On completion, the actual submitted game count.
    expected_pairs: usize,
    resolved_pairs: usize,
    failed_pairs: usize,
    pairs: Vec<PairRow>,
    output_path: Option<String>,
    tournament_format: String,
    best_of: u32,
    seed: u64,
    tournament: Option<Tournament>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RunStatus {
    Running,
    Complete,
    Incomplete,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PairRow {
    task_id: String,
    model_a: String,
    model_b: String,
    status: PairStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    judgment: Option<Judgment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<JudgmentFailure>,
    /// Resolved judged games in this series (survives SSE reconnect).
    games_resolved: u32,
    wins_a: u32,
    wins_b: u32,
    tiebreak_games: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    series_winner: Option<String>,
    seeded_fallback: bool,
    series_draw: bool,
    series_failed: bool,
    awaiting_tiebreak: bool,
}

impl PairRow {
    fn waiting(task_id: String, model_a: String, model_b: String) -> Self {
        Self {
            task_id,
            model_a,
            model_b,
            status: PairStatus::Waiting,
            judgment: None,
            failure: None,
            games_resolved: 0,
            wins_a: 0,
            wins_b: 0,
            tiebreak_games: 0,
            series_winner: None,
            seeded_fallback: false,
            series_draw: false,
            series_failed: false,
            awaiting_tiebreak: false,
        }
    }

    fn series_complete(&self) -> bool {
        self.series_winner.is_some()
            || self.series_draw
            || self.series_failed
            || self.seeded_fallback
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PairStatus {
    Waiting,
    Judging,
    Resolved,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
struct RunSnapshot {
    seq: u64,
    run_id: String,
    status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    candidate_count: usize,
    task_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    judge: Option<String>,
    started_at: String,
    candidate_completed: usize,
    candidate_total: usize,
    planned_games: usize,
    expected_pairs: usize,
    resolved_pairs: usize,
    failed_pairs: usize,
    pairs: Vec<PairRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_path: Option<String>,
    tournament_format: String,
    best_of: u32,
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tournament: Option<Tournament>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    Snapshot {
        #[serde(flatten)]
        snapshot: Box<RunSnapshot>,
    },
    CandidateFinished {
        seq: u64,
        task_id: String,
        model: String,
        duration_ms: u64,
    },
    PairResolved {
        seq: u64,
        judgment: Judgment,
        games_resolved: u32,
        wins_a: u32,
        wins_b: u32,
        tiebreak_games: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        series_winner: Option<String>,
        seeded_fallback: bool,
        series_draw: bool,
        series_failed: bool,
        awaiting_tiebreak: bool,
    },
    PairFailed {
        seq: u64,
        failure: JudgmentFailure,
        games_resolved: u32,
        wins_a: u32,
        wins_b: u32,
        tiebreak_games: u32,
        series_failed: bool,
        awaiting_tiebreak: bool,
    },
    RunComplete {
        seq: u64,
        planned_games: usize,
        expected_pairs: usize,
        resolved_pairs: usize,
        failed_pairs: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tournament: Option<Tournament>,
    },
    RunFailed {
        seq: u64,
        error: String,
    },
}

#[derive(Debug)]
enum StartRunError {
    Busy,
}

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WebProviderKind {
    Openai,
    Claude,
    Gemini,
    Compatible,
}

impl WebProviderKind {
    fn fixed_base_url(self) -> Option<&'static str> {
        match self {
            Self::Openai => Some(OPENAI_BASE_URL),
            Self::Claude => Some(ANTHROPIC_BASE_URL),
            Self::Gemini => Some(GEMINI_BASE_URL),
            Self::Compatible => None,
        }
    }

    fn supports_model_discovery(self) -> bool {
        matches!(self, Self::Openai | Self::Compatible)
    }

    fn as_persisted(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Claude => "anthropic",
            Self::Gemini => "gemini",
            Self::Compatible => "compatible",
        }
    }
}

struct ResolvedWebProvider {
    kind: WebProviderKind,
    api_key: String,
    base_url: String,
}

impl std::fmt::Debug for ResolvedWebProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedWebProvider")
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

fn resolve_web_provider(
    kind: WebProviderKind,
    api_key: &str,
    base_url: &str,
) -> std::result::Result<ResolvedWebProvider, &'static str> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err("API key is required");
    }
    let base_url = match kind.fixed_base_url() {
        Some(url) => url.to_string(),
        None => {
            let url = base_url.trim();
            if url.is_empty() {
                return Err("base URL is required");
            }
            url.to_string()
        }
    };
    Ok(ResolvedWebProvider {
        kind,
        api_key: api_key.to_string(),
        base_url,
    })
}

#[derive(Debug, Deserialize)]
struct ConnectRequest {
    provider: WebProviderKind,
    #[serde(default)]
    base_url: String,
    api_key: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StartRequest {
    provider: WebProviderKind,
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    api_key: String,
    models: Vec<String>,
    #[serde(default)]
    judge: Option<String>,
    tasks: serde_json::Value,
    #[serde(default)]
    seed: String,
    #[serde(default)]
    tournament: TournamentFormat,
    #[serde(default = "crate::tournament::default_best_of")]
    best_of: u32,
    #[serde(default)]
    opening_matchups: Option<Vec<OpeningMatchup>>,
}

#[derive(Serialize)]
struct StartResponse {
    run_id: String,
}

impl AppState {
    fn new() -> Self {
        Self {
            session: Mutex::new(None),
            next_run: AtomicU64::new(1),
            run: Mutex::new(None),
        }
    }

    fn alloc_id(&self, output_dir: &FilePath) -> String {
        allocate_web_run_id(output_dir, &self.next_run)
    }
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn format_web_run_id(millis: u128, pid: u32, seq: u64) -> String {
    format!("{millis}-{pid}-{seq}")
}

fn allocate_web_run_id(dir: &FilePath, counter: &AtomicU64) -> String {
    allocate_web_run_id_with(dir, counter, unix_millis(), std::process::id())
}

fn allocate_web_run_id_with(dir: &FilePath, counter: &AtomicU64, millis: u128, pid: u32) -> String {
    loop {
        let seq = counter.fetch_add(1, Ordering::Relaxed);
        let id = format_web_run_id(millis, pid, seq);
        if !web_output_path(dir, &id).exists() {
            return id;
        }
    }
}

impl ClientEvent {
    fn seq(&self) -> u64 {
        match self {
            Self::Snapshot { snapshot } => snapshot.seq,
            Self::CandidateFinished { seq, .. }
            | Self::PairResolved { seq, .. }
            | Self::PairFailed { seq, .. }
            | Self::RunComplete { seq, .. }
            | Self::RunFailed { seq, .. } => *seq,
        }
    }

    fn is_terminal(&self) -> bool {
        match self {
            Self::Snapshot { snapshot } => !matches!(snapshot.status, RunStatus::Running),
            Self::RunComplete { .. } | Self::RunFailed { .. } => true,
            _ => false,
        }
    }
}

impl LiveRun {
    fn from_config(id: String, config: &ExecConfig) -> Self {
        let candidate_count = config.models.len();
        let task_count = config.tasks.len();
        let candidate_total = task_count.saturating_mul(candidate_count);
        let planned_games = if config.judge.is_some() {
            tournament::planned_match_count(config.tournament, candidate_count)
                .saturating_mul(task_count)
                .saturating_mul(config.best_of as usize)
        } else {
            0
        };
        let pairs = if config.judge.is_some() {
            waiting_pairs(
                &config.tasks,
                &config.models,
                config.tournament,
                config.opening_matchups.as_deref(),
            )
        } else {
            Vec::new()
        };
        let (events, _) = broadcast::channel(64);
        Self {
            id,
            finished: AtomicBool::new(false),
            inner: Mutex::new(RunState {
                seq: 0,
                status: RunStatus::Running,
                error: None,
                candidate_count,
                task_count,
                judge: config.judge.as_ref().map(ToString::to_string),
                started_at: config.started_at.clone(),
                candidate_completed: 0,
                candidate_total,
                planned_games,
                expected_pairs: planned_games,
                resolved_pairs: 0,
                failed_pairs: 0,
                pairs,
                output_path: None,
                tournament_format: config.tournament.as_str().to_string(),
                best_of: config.best_of,
                seed: config.seed,
                tournament: None,
            }),
            events,
        }
    }

    fn is_running(&self) -> bool {
        !self.finished.load(Ordering::Acquire)
    }

    fn subscribe(&self) -> broadcast::Receiver<ClientEvent> {
        self.events.subscribe()
    }

    async fn snapshot(&self) -> ClientEvent {
        let inner = self.inner.lock().await;
        inner.snapshot(self.id.clone())
    }

    async fn apply(&self, event: ExperimentEvent) -> ClientEvent {
        let client = {
            let mut inner = self.inner.lock().await;
            inner.apply(event)
        };
        if client.is_terminal() {
            self.finished.store(true, Ordering::Release);
        }
        let _ = self.events.send(client.clone());
        client
    }

    async fn fail(&self, error: String) {
        let client = {
            let mut inner = self.inner.lock().await;
            inner.fail(error)
        };
        let Some(client) = client else {
            return;
        };
        self.finished.store(true, Ordering::Release);
        let _ = self.events.send(client);
    }

    async fn record_output_path(&self, path: &FilePath) {
        self.inner.lock().await.output_path = Some(path.display().to_string());
    }

    async fn record_tournament(&self, tournament: Option<Tournament>) {
        self.inner.lock().await.tournament = tournament;
    }
}

impl RunState {
    fn snapshot(&self, run_id: String) -> ClientEvent {
        ClientEvent::Snapshot {
            snapshot: Box::new(RunSnapshot {
                seq: self.seq,
                run_id,
                status: self.status,
                error: self.error.clone(),
                candidate_count: self.candidate_count,
                task_count: self.task_count,
                judge: self.judge.clone(),
                started_at: self.started_at.clone(),
                candidate_completed: self.candidate_completed,
                candidate_total: self.candidate_total,
                planned_games: self.planned_games,
                expected_pairs: self.expected_pairs,
                resolved_pairs: self.resolved_pairs,
                failed_pairs: self.failed_pairs,
                pairs: self.pairs.clone(),
                output_path: self.output_path.clone(),
                tournament_format: self.tournament_format.clone(),
                best_of: self.best_of,
                seed: self.seed,
                tournament: self.tournament.clone(),
            }),
        }
    }

    fn apply(&mut self, event: ExperimentEvent) -> ClientEvent {
        self.seq += 1;
        match event {
            ExperimentEvent::CandidateFinished {
                task_id,
                model,
                duration_ms,
            } => {
                self.candidate_completed += 1;
                self.mark_judging();
                ClientEvent::CandidateFinished {
                    seq: self.seq,
                    task_id,
                    model: model.to_string(),
                    duration_ms,
                }
            }
            ExperimentEvent::PairResolved {
                task_id,
                model_a,
                model_b,
                judgment,
            } => {
                self.resolved_pairs += 1;
                let row = apply_resolved_game(
                    &mut self.pairs,
                    &task_id,
                    &model_a,
                    &model_b,
                    judgment.clone(),
                    self.best_of,
                    &self.tournament_format,
                    self.seed,
                );
                ClientEvent::PairResolved {
                    seq: self.seq,
                    judgment,
                    games_resolved: row.games_resolved,
                    wins_a: row.wins_a,
                    wins_b: row.wins_b,
                    tiebreak_games: row.tiebreak_games,
                    series_winner: row.series_winner.clone(),
                    seeded_fallback: row.seeded_fallback,
                    series_draw: row.series_draw,
                    series_failed: row.series_failed,
                    awaiting_tiebreak: row.awaiting_tiebreak,
                }
            }
            ExperimentEvent::PairFailed {
                task_id,
                model_a,
                model_b,
                failure,
            } => {
                self.failed_pairs += 1;
                let row = apply_failed_game(
                    &mut self.pairs,
                    &task_id,
                    &model_a,
                    &model_b,
                    failure.clone(),
                );
                ClientEvent::PairFailed {
                    seq: self.seq,
                    failure,
                    games_resolved: row.games_resolved,
                    wins_a: row.wins_a,
                    wins_b: row.wins_b,
                    tiebreak_games: row.tiebreak_games,
                    series_failed: row.series_failed,
                    awaiting_tiebreak: row.awaiting_tiebreak,
                }
            }
            ExperimentEvent::RunComplete {
                expected_pairs,
                resolved_pairs,
                failed_pairs,
            } => {
                self.status = if failed_pairs == 0 {
                    RunStatus::Complete
                } else {
                    RunStatus::Incomplete
                };
                self.expected_pairs = expected_pairs;
                self.resolved_pairs = resolved_pairs;
                self.failed_pairs = failed_pairs;
                ClientEvent::RunComplete {
                    seq: self.seq,
                    planned_games: self.planned_games,
                    expected_pairs,
                    resolved_pairs,
                    failed_pairs,
                    output_path: self.output_path.clone(),
                    tournament: self.tournament.clone(),
                }
            }
        }
    }

    fn fail(&mut self, error: String) -> Option<ClientEvent> {
        if !matches!(self.status, RunStatus::Running) {
            return None;
        }
        self.seq += 1;
        self.status = RunStatus::Failed;
        self.error = Some(error.clone());
        Some(ClientEvent::RunFailed {
            seq: self.seq,
            error,
        })
    }

    fn mark_judging(&mut self) {
        // ExperimentEvent has no per-pair start; judging is the post-candidate phase.
        if self.candidate_completed < self.candidate_total {
            return;
        }
        for pair in &mut self.pairs {
            if pair.status == PairStatus::Waiting {
                pair.status = PairStatus::Judging;
            }
        }
    }
}

fn same_unordered_pair(row: &PairRow, task_id: &str, model_a: &str, model_b: &str) -> bool {
    row.task_id == task_id
        && ((row.model_a == model_a && row.model_b == model_b)
            || (row.model_a == model_b && row.model_b == model_a))
}

fn uses_elimination_tiebreaks(tournament_format: &str) -> bool {
    matches!(tournament_format, "single-elimination" | "king-of-the-hill")
}

fn find_or_insert_pair<'a>(
    pairs: &'a mut Vec<PairRow>,
    task_id: &str,
    model_a: &ModelId,
    model_b: &ModelId,
) -> &'a mut PairRow {
    let a = model_a.to_string();
    let b = model_b.to_string();
    if let Some(index) = pairs
        .iter()
        .position(|row| same_unordered_pair(row, task_id, &a, &b))
    {
        return &mut pairs[index];
    }
    pairs.push(PairRow::waiting(task_id.to_string(), a, b));
    pairs.last_mut().expect("pair just inserted")
}

#[allow(clippy::too_many_arguments)]
fn apply_resolved_game(
    pairs: &mut Vec<PairRow>,
    task_id: &str,
    model_a: &ModelId,
    model_b: &ModelId,
    judgment: Judgment,
    best_of: u32,
    tournament_format: &str,
    seed: u64,
) -> PairRow {
    let row = find_or_insert_pair(pairs, task_id, model_a, model_b);
    if row.series_complete() {
        return row.clone();
    }

    row.status = PairStatus::Resolved;
    row.judgment = Some(judgment.clone());
    row.failure = None;
    row.games_resolved = row.games_resolved.saturating_add(1);

    let winner_model = match judgment.winner {
        crate::judge::JudgeDecision::A => Some(judgment.model_a.to_string()),
        crate::judge::JudgeDecision::B => Some(judgment.model_b.to_string()),
        crate::judge::JudgeDecision::Draw => None,
    };
    if let Some(winner) = winner_model.as_ref() {
        if *winner == row.model_a {
            row.wins_a = row.wins_a.saturating_add(1);
        } else if *winner == row.model_b {
            row.wins_b = row.wins_b.saturating_add(1);
        }
    }

    if row.awaiting_tiebreak {
        row.tiebreak_games = row.tiebreak_games.saturating_add(1);
        if let Some(winner) = winner_model {
            row.series_winner = Some(winner);
            row.awaiting_tiebreak = false;
            row.series_draw = false;
            row.seeded_fallback = false;
        } else if row.tiebreak_games >= tournament::MAX_TIEBREAKS {
            let left = ModelId::new(row.model_a.as_str());
            let right = ModelId::new(row.model_b.as_str());
            row.series_winner =
                Some(tournament::seeded_fallback_winner(seed, &left, &right).to_string());
            row.seeded_fallback = true;
            row.awaiting_tiebreak = false;
            row.series_draw = false;
        }
    } else {
        let regulation_played = row.games_resolved;
        match tournament::series_result(row.wins_a, row.wins_b, regulation_played, best_of) {
            Some(tournament::SeriesEnd::WinnerA) => {
                row.series_winner = Some(row.model_a.clone());
                row.series_draw = false;
                row.awaiting_tiebreak = false;
            }
            Some(tournament::SeriesEnd::WinnerB) => {
                row.series_winner = Some(row.model_b.clone());
                row.series_draw = false;
                row.awaiting_tiebreak = false;
            }
            Some(tournament::SeriesEnd::Draw) => {
                if uses_elimination_tiebreaks(tournament_format) {
                    row.awaiting_tiebreak = true;
                    row.series_draw = false;
                } else {
                    row.series_draw = true;
                    row.awaiting_tiebreak = false;
                }
            }
            None => {}
        }
    }

    row.clone()
}

fn apply_failed_game(
    pairs: &mut Vec<PairRow>,
    task_id: &str,
    model_a: &ModelId,
    model_b: &ModelId,
    failure: JudgmentFailure,
) -> PairRow {
    let row = find_or_insert_pair(pairs, task_id, model_a, model_b);
    row.status = PairStatus::Failed;
    row.failure = Some(failure);
    row.judgment = None;
    row.series_failed = true;
    row.awaiting_tiebreak = false;
    row.series_draw = false;
    row.series_winner = None;
    row.seeded_fallback = false;
    row.clone()
}

fn waiting_pairs(
    tasks: &[Task],
    models: &[ModelId],
    format: TournamentFormat,
    opening: Option<&[OpeningMatchup]>,
) -> Vec<PairRow> {
    let mut pairs = Vec::new();
    for task in tasks {
        for (model_a, model_b) in tournament::opening_pairs(format, models, opening) {
            pairs.push(PairRow::waiting(
                task.id.clone(),
                model_a.to_string(),
                model_b.to_string(),
            ));
        }
    }
    pairs
}

fn parse_seed(raw: &str) -> std::result::Result<u64, &'static str> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(0);
    }
    if !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("seed must be a non-negative integer");
    }
    raw.parse::<u64>()
        .map_err(|_| "seed must be a non-negative integer")
}

fn listen_url(listener: &tokio::net::TcpListener) -> std::io::Result<String> {
    let addr = listener.local_addr()?;
    Ok(format!("http://{addr}"))
}

pub async fn serve(port: u16) -> Result<()> {
    let addr = std::net::SocketAddr::from((BIND_HOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Arena web UI: {}", listen_url(&listener)?);
    axum::serve(listener, router()).await?;
    Ok(())
}

fn router() -> Router {
    router_with_state(Arc::new(AppState::new()))
}

fn router_with_state(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(page))
        .route("/api/models", post(load_models))
        .route("/api/runs", get(list_runs).post(start_run))
        .route("/api/runs/{run_id}/events", get(run_events))
        .route("/api/runs/{run_id}/report", get(run_report))
        .with_state(state)
}

async fn page() -> Html<&'static str> {
    Html(PAGE)
}

async fn list_runs() -> Response {
    Json(list_saved_runs(FilePath::new(WEB_OUTPUT_DIR))).into_response()
}

async fn load_models(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ConnectRequest>,
) -> Response {
    if !request.provider.supports_model_discovery() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "model discovery is not available for this provider",
        );
    }
    let resolved = match resolve_web_provider(request.provider, &request.api_key, &request.base_url)
    {
        Ok(resolved) => resolved,
        Err(message) => return json_error(StatusCode::BAD_REQUEST, message),
    };

    let provider =
        OpenAICompatibleProvider::new(resolved.api_key.clone(), resolved.base_url.clone());
    match provider.list_models().await {
        Ok(models) => {
            *state.session.lock().await = Some(ProviderSession {
                kind: resolved.kind,
                api_key: resolved.api_key,
                base_url: resolved.base_url,
            });
            (
                StatusCode::OK,
                Json(ModelsResponse {
                    models: models.iter().map(ToString::to_string).collect(),
                }),
            )
                .into_response()
        }
        Err(error) => json_error(StatusCode::BAD_GATEWAY, error.to_string()),
    }
}

async fn start_run(
    State(state): State<Arc<AppState>>,
    Json(request): Json<StartRequest>,
) -> Response {
    let session = state.session.lock().await;
    let saved = session
        .as_ref()
        .filter(|item| item.kind == request.provider);
    let api_key = if request.api_key.trim().is_empty() {
        saved.map(|item| item.api_key.clone()).unwrap_or_default()
    } else {
        request.api_key.clone()
    };
    let base_url = if request.base_url.trim().is_empty() {
        saved.map(|item| item.base_url.clone()).unwrap_or_default()
    } else {
        request.base_url.clone()
    };
    drop(session);
    let resolved = match resolve_web_provider(request.provider, &api_key, &base_url) {
        Ok(resolved) => resolved,
        Err(message) => return json_error(StatusCode::BAD_REQUEST, message),
    };

    let tasks_json = match serde_json::to_string(&request.tasks) {
        Ok(json) => json,
        Err(error) => return json_error(StatusCode::BAD_REQUEST, error.to_string()),
    };
    let tasks = match task::parse(&tasks_json) {
        Ok(tasks) => tasks,
        Err(error) => return json_error(StatusCode::BAD_REQUEST, error.to_string()),
    };
    let seed = match parse_seed(&request.seed) {
        Ok(seed) => seed,
        Err(message) => return json_error(StatusCode::BAD_REQUEST, message),
    };

    let config = match build_exec_config_with_format(
        request.models,
        request.judge,
        tasks,
        resolved.base_url.clone(),
        Some(resolved.kind.as_persisted().to_string()),
        seed,
        request.tournament,
        request.best_of,
        request.opening_matchups,
    ) {
        Ok(config) => config,
        Err(error) => return json_error(StatusCode::BAD_REQUEST, error.to_string()),
    };

    let output_dir = PathBuf::from(WEB_OUTPUT_DIR);
    let started = match resolved.kind {
        WebProviderKind::Openai | WebProviderKind::Compatible => {
            let provider = OpenAICompatibleProvider::new(resolved.api_key, resolved.base_url);
            start_experiment(state, config, provider, output_dir).await
        }
        WebProviderKind::Claude => {
            let provider = AnthropicProvider::new(resolved.api_key, resolved.base_url);
            start_experiment(state, config, provider, output_dir).await
        }
        WebProviderKind::Gemini => {
            let provider = GeminiProvider::new(resolved.api_key, resolved.base_url);
            start_experiment(state, config, provider, output_dir).await
        }
    };
    match started {
        Ok(run_id) => (StatusCode::ACCEPTED, Json(StartResponse { run_id })).into_response(),
        Err(StartRunError::Busy) => {
            json_error(StatusCode::CONFLICT, "an experiment is already running")
        }
    }
}

fn web_output_path(dir: &FilePath, run_id: &str) -> PathBuf {
    dir.join(format!("{run_id}.json"))
}

fn is_safe_run_id(run_id: &str) -> bool {
    !run_id.is_empty()
        && run_id.len() <= 128
        && !run_id.starts_with('.')
        && run_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
}

fn resolve_saved_run_path(
    dir: &FilePath,
    run_id: &str,
) -> std::result::Result<PathBuf, (StatusCode, String)> {
    if !is_safe_run_id(run_id) {
        return Err((StatusCode::BAD_REQUEST, "invalid run id".into()));
    }
    let path = web_output_path(dir, run_id);
    if !path.is_file() {
        return Err((StatusCode::NOT_FOUND, "run not found".into()));
    }
    Ok(path)
}

/// Lightweight view of a saved experiment file for the history list.
/// Only `run` and optional tournament summary fields are retained.
#[derive(Debug, Deserialize)]
struct HistoryDocument {
    run: HistoryRunMeta,
    #[serde(default)]
    tournament: Option<HistoryTournamentMeta>,
}

#[derive(Debug, Deserialize)]
struct HistoryRunMeta {
    models: Vec<ModelId>,
    #[serde(default)]
    judge: Option<ModelId>,
    started_at: String,
    #[serde(default)]
    complete: Option<bool>,
    #[serde(default)]
    expected_pairs: Option<usize>,
    #[serde(default)]
    resolved_pairs: Option<usize>,
    #[serde(default)]
    failed_pairs: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct HistoryTournamentMeta {
    format: TournamentFormat,
    status: TournamentStatus,
}

#[derive(Debug, Serialize)]
struct HistoryRun {
    run_id: String,
    started_at: String,
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    judge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tournament_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tournament_status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    complete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_pairs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolved_pairs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed_pairs: Option<usize>,
    path: String,
}

#[derive(Debug, Serialize)]
struct HistoryError {
    file: String,
    error: String,
}

#[derive(Debug, Serialize)]
struct HistoryList {
    runs: Vec<HistoryRun>,
    errors: Vec<HistoryError>,
}

fn read_history_document(path: &FilePath) -> std::result::Result<HistoryDocument, String> {
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents).map_err(|error| error.to_string())
}

fn history_run_from_document(run_id: &str, path: &FilePath, doc: HistoryDocument) -> HistoryRun {
    HistoryRun {
        run_id: run_id.to_string(),
        started_at: doc.run.started_at,
        models: doc.run.models.iter().map(ToString::to_string).collect(),
        judge: doc.run.judge.map(|model| model.to_string()),
        tournament_format: doc.tournament.as_ref().map(|item| item.format.as_str()),
        tournament_status: doc.tournament.as_ref().map(|item| item.status.as_str()),
        complete: doc.run.complete,
        expected_pairs: doc.run.expected_pairs,
        resolved_pairs: doc.run.resolved_pairs,
        failed_pairs: doc.run.failed_pairs,
        path: path.display().to_string(),
    }
}

fn list_saved_runs(dir: &FilePath) -> HistoryList {
    let mut runs = Vec::new();
    let mut errors = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return HistoryList { runs, errors };
        }
        Err(error) => {
            errors.push(HistoryError {
                file: dir.display().to_string(),
                error: error.to_string(),
            });
            return HistoryList { runs, errors };
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                errors.push(HistoryError {
                    file: dir.display().to_string(),
                    error: error.to_string(),
                });
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            errors.push(HistoryError {
                file: path.display().to_string(),
                error: "file name is not valid UTF-8".into(),
            });
            continue;
        };
        if !is_safe_run_id(stem) {
            errors.push(HistoryError {
                file: path.display().to_string(),
                error: "run id is not a safe filename stem".into(),
            });
            continue;
        }
        let resolved = match resolve_saved_run_path(dir, stem) {
            Ok(resolved) => resolved,
            Err((_, message)) => {
                errors.push(HistoryError {
                    file: path.display().to_string(),
                    error: message,
                });
                continue;
            }
        };
        match read_history_document(&resolved) {
            Ok(doc) => runs.push(history_run_from_document(stem, &resolved, doc)),
            Err(error) => errors.push(HistoryError {
                file: resolved.display().to_string(),
                error,
            }),
        }
    }

    runs.sort_by(|left, right| {
        right
            .started_at
            .cmp(&left.started_at)
            .then_with(|| right.run_id.cmp(&left.run_id))
    });
    HistoryList { runs, errors }
}

fn render_saved_run_report(
    dir: &FilePath,
    run_id: &str,
) -> std::result::Result<String, (StatusCode, String)> {
    let path = resolve_saved_run_path(dir, run_id)?;
    let output = persist::read(&path)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(html::render_with_nav(
        &report::from_output(&output),
        Some("/"),
    ))
}

fn write_experiment(path: &FilePath, output: &persist::Output) -> std::result::Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let json = persist::to_pretty_json(output).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    file.write_all(json.as_bytes())
        .map_err(|error| error.to_string())?;
    Ok(())
}

async fn start_experiment<P>(
    state: Arc<AppState>,
    config: ExecConfig,
    provider: P,
    output_dir: PathBuf,
) -> std::result::Result<String, StartRunError>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let live = {
        let mut slot = state.run.lock().await;
        if slot.as_ref().is_some_and(|run| run.is_running()) {
            return Err(StartRunError::Busy);
        }
        let id = state.alloc_id(&output_dir);
        let live = Arc::new(LiveRun::from_config(id, &config));
        *slot = Some(live.clone());
        live
    };

    let run_id = live.id.clone();
    let output_path = web_output_path(&output_dir, &run_id);
    tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let collect =
            tokio::spawn(
                async move { exec::collect_exec_with_events(&provider, &config, tx).await },
            );
        let mut completion = None;
        while let Some(event) = rx.recv().await {
            match event {
                ExperimentEvent::RunComplete { .. } => completion = Some(event),
                event => {
                    live.apply(event).await;
                }
            }
        }
        match collect.await {
            Ok(Ok((output, _failed_pairs))) => match write_experiment(&output_path, &output) {
                Ok(()) => {
                    live.record_output_path(&output_path).await;
                    live.record_tournament(output.tournament.clone()).await;
                    if let Some(event) = completion {
                        live.apply(event).await;
                    }
                }
                Err(error) => live.fail(error).await,
            },
            Ok(Err(error)) => live.fail(error.to_string()).await,
            Err(error) => live.fail(error.to_string()).await,
        }
    });

    Ok(run_id)
}

async fn run_events(State(state): State<Arc<AppState>>, Path(run_id): Path<String>) -> Response {
    let live = state.run.lock().await.clone();
    let Some(live) = live.filter(|run| run.id == run_id) else {
        return json_error(StatusCode::NOT_FOUND, "run not found");
    };
    sse_stream(live).into_response()
}

async fn run_report(State(state): State<Arc<AppState>>, Path(run_id): Path<String>) -> Response {
    let live = state.run.lock().await.clone();
    if let Some(live) = live.filter(|run| run.id == run_id) {
        match render_live_run_report(&live).await {
            Ok(body) => return Html(body).into_response(),
            Err((StatusCode::CONFLICT, message)) => {
                return json_error(StatusCode::CONFLICT, message);
            }
            Err(_) => {
                // Fall through to the saved file under arena-web/.
            }
        }
    }
    match render_saved_run_report(FilePath::new(WEB_OUTPUT_DIR), &run_id) {
        Ok(body) => Html(body).into_response(),
        Err((status, message)) => json_error(status, message),
    }
}

async fn render_live_run_report(
    live: &LiveRun,
) -> std::result::Result<String, (StatusCode, String)> {
    let (status, output_path) = {
        let inner = live.inner.lock().await;
        (inner.status, inner.output_path.clone())
    };
    match status {
        RunStatus::Running => {
            return Err((StatusCode::CONFLICT, "run is still in progress".into()));
        }
        RunStatus::Failed => {
            return Err((
                StatusCode::NOT_FOUND,
                "no saved experiment for this run".into(),
            ));
        }
        RunStatus::Complete | RunStatus::Incomplete => {}
    }
    let Some(path) = output_path else {
        return Err((
            StatusCode::NOT_FOUND,
            "no saved experiment for this run".into(),
        ));
    };
    let output = persist::read(&path)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(html::render_with_nav(
        &report::from_output(&output),
        Some("/"),
    ))
}

fn sse_stream(
    live: Arc<LiveRun>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        let mut events = live.subscribe();
        let snapshot = live.snapshot().await;
        let seq = snapshot.seq();
        if tx.send(snapshot).await.is_err() {
            return;
        }
        loop {
            match events.recv().await {
                Ok(event) => {
                    if event.seq() <= seq {
                        continue;
                    }
                    let terminal = event.is_terminal();
                    if tx.send(event).await.is_err() {
                        break;
                    }
                    if terminal {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let snapshot = live.snapshot().await;
                    let terminal = snapshot.is_terminal();
                    if tx.send(snapshot).await.is_err() {
                        break;
                    }
                    if terminal {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Sse::new(ReceiverStream::new(rx).map(|payload| {
        Ok(Event::default()
            .json_data(&payload)
            .expect("client event json"))
    }))
    .keep_alive(KeepAlive::default())
}

fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

#[cfg(test)]
fn build_exec_config(
    models: Vec<String>,
    judge: Option<String>,
    tasks: Vec<Task>,
    base_url: String,
    seed: u64,
) -> Result<ExecConfig> {
    build_exec_config_with_format(
        models,
        judge,
        tasks,
        base_url,
        None,
        seed,
        TournamentFormat::RoundRobin,
        tournament::DEFAULT_BEST_OF,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exec_config_with_format(
    models: Vec<String>,
    judge: Option<String>,
    tasks: Vec<Task>,
    base_url: String,
    provider: Option<String>,
    seed: u64,
    tournament: TournamentFormat,
    best_of: u32,
    opening_matchups: Option<Vec<OpeningMatchup>>,
) -> Result<ExecConfig> {
    let models = exec::unique_models(
        models
            .into_iter()
            .map(|model| model.trim().to_string())
            .filter(|model| !model.is_empty())
            .collect(),
    )?;
    if models.is_empty() {
        return Err(Error::NoCandidates);
    }
    let judge = judge
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(ModelId::new);
    exec::validate_judge(&models, judge.as_ref())?;
    tournament::validate(tournament, models.len())?;
    tournament::validate_best_of(best_of)?;
    tournament::validate_opening_matchups(tournament, &models, opening_matchups.as_deref())?;
    Ok(ExecConfig {
        tasks,
        models,
        judge,
        tournament,
        best_of,
        opening_matchups,
        seed,
        tasks_path: None,
        started_at: persist::utc_timestamp(),
        base_url,
        provider,
    })
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arena</title>
<style>
:root {
  --ink: #1c1915;
  --muted: #5e584f;
  --line: #e0d8cc;
  --bg: #f6f3ec;
  --field: #fffdf8;
  --editor: #f3eee6;
  --select: #f4ebe3;
  --accent: #c04a2c;
  --ok: #2f6b45;
  --warn: #8a5a12;
  --bad: #9c3a32;
  --mono: ui-monospace, "Cascadia Mono", "Segoe UI Mono", monospace;
  --sans: "Segoe UI", system-ui, sans-serif;
}
* { box-sizing: border-box; }
html { color-scheme: light; }
html, body { overflow-x: hidden; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 var(--sans);
}
.wrap { max-width: 76rem; margin: 0 auto; padding: 24px 32px 32px; }
.mast {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--line);
}
.brand h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: .04em;
  line-height: 1;
  text-transform: uppercase;
}
.kicker {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  grid-template-areas:
    "provider provider"
    "manual manual"
    "candidates judge"
    "tasks tasks"
    "controls controls";
  gap: 16px 24px;
  align-items: start;
  margin-top: 24px;
}
.provider { grid-area: provider; }
.manual { grid-area: manual; }
.candidates { grid-area: candidates; }
.judge { grid-area: judge; }
.tasks { grid-area: tasks; }
.controls { grid-area: controls; }
.workspace > .region { min-width: 0; }
.region h2, label.region-title, .section-line h2, .experiment > h2, #elim_section > h2, #hill_section > h2, .elim-round h3 {
  display: block;
  margin: 0 0 8px;
  padding: 0;
  border: 0;
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.region .meta { margin: 0 0 8px; color: var(--muted); font-size: .82rem; }
label { display: block; margin: 0 0 4px; color: var(--muted); font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
label.region-title { margin-top: 0; }
input[type=text], input[type=password], input[type=number], textarea, select {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid var(--line);
  border-radius: 0;
  background: var(--field);
  color: inherit;
  font: inherit;
}
input[type=text]:hover, input[type=password]:hover, input[type=number]:hover, textarea:hover, select:hover {
  border-color: #cfc6b8;
}
input:focus, textarea:focus, select:focus {
  outline: none;
  border-color: var(--accent);
}
textarea {
  min-height: 10rem;
  padding: 8px 12px;
  background: var(--editor);
  font-family: var(--mono);
  font-size: .84rem;
  line-height: 1.45;
}
#seed, #best_of { max-width: 8rem; font-family: var(--mono); font-size: .9rem; }
button {
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: 0;
  background: transparent;
  color: var(--ink);
  font: inherit;
  cursor: pointer;
}
button.secondary:hover { border-color: var(--ink); }
button.primary {
  padding: 8px 14px;
  border-color: var(--ink);
  background: var(--ink);
  color: var(--bg);
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
}
button.primary:hover { background: #000; border-color: #000; }
a.report-link {
  display: inline-block;
  margin: 8px 0 0;
  padding: 8px 14px;
  border: 1px solid var(--ink);
  background: var(--ink);
  color: var(--bg);
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  text-decoration: none;
}
a.report-link:hover { background: #000; border-color: #000; }
.report-actions { margin: 0 0 12px; }
button.text-button {
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  text-align: left;
}
tr.openable button.text-button::before {
  content: "";
  display: inline-block;
  width: 0;
  height: 0;
  margin-right: .4rem;
  border-style: solid;
  border-width: .28rem 0 .28rem .38rem;
  border-color: transparent transparent transparent currentColor;
  vertical-align: .05rem;
}
tr.open button.text-button::before {
  border-width: .38rem .28rem 0 .28rem;
  border-color: currentColor transparent transparent transparent;
  vertical-align: .1rem;
}
.segments {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
  width: auto;
  max-width: 100%;
  margin: 0 0 16px;
}
button.segment {
  margin: 0;
  padding: 0 0 4px;
  border: 0;
  border-bottom: 1px solid transparent;
  border-radius: 0;
  background: transparent;
  color: var(--muted);
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
}
button.segment.is-selected {
  color: var(--accent);
  border-bottom-color: var(--accent);
  background: transparent;
}
button.segment:hover { color: var(--ink); }
button.segment.is-selected:hover { color: var(--accent); }
.provider-fields {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 16px;
  align-items: flex-end;
}
.provider-fields > div { flex: 1 1 16rem; min-width: 0; }
.provider-fields > .actions { flex: 0 1 auto; margin: 0; }
.picker { position: relative; }
.picker.is-open, .region.menu-open { position: relative; z-index: 40; }
.picker-toggle {
  display: flex;
  width: 100%;
  justify-content: space-between;
  align-items: center;
  gap: .6rem;
  padding: .4rem .55rem;
  text-align: left;
  font-weight: 400;
  background: var(--field);
}
.picker-toggle span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.picker-toggle span.placeholder { color: var(--muted); }
.picker-toggle span.mono-label { font-family: var(--mono); font-size: .86rem; }
.picker-toggle::after {
  content: "";
  flex: 0 0 auto;
  width: 0;
  height: 0;
  border-style: solid;
  border-width: .34rem .26rem 0 .26rem;
  border-color: currentColor transparent transparent transparent;
}
.picker-menu {
  display: none;
  position: absolute;
  z-index: 2;
  left: 0;
  right: 0;
  top: calc(100% + 2px);
  flex-direction: column;
  max-height: min(16rem, 45vh);
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 0;
  background: var(--field);
}
.picker-menu.is-open { display: flex; }
.picker-menu.above { top: auto; bottom: calc(100% + 2px); }
.picker-menu input {
  border: 0;
  border-bottom: 1px solid var(--line);
  border-radius: 0;
}
.picker-options { min-width: 0; min-height: 0; overflow: auto; }
.picker-option, button.picker-option {
  display: flex;
  align-items: center;
  gap: .55rem;
  width: max-content;
  min-width: 100%;
  margin: 0;
  padding: .4rem .55rem;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: inherit;
  font-weight: 400;
  font-family: var(--mono);
  font-size: .86rem;
  text-align: left;
  white-space: nowrap;
}
label.picker-option { margin: 0; color: inherit; letter-spacing: 0; text-transform: none; }
button.picker-option.plain { font-family: var(--sans); font-size: .9rem; letter-spacing: 0; text-transform: none; }
.picker-option.is-selected { background: var(--select); border-left: 2px solid var(--accent); }
.picker-option:hover, button.picker-option:hover { background: var(--editor); }
.picker-option.is-selected:hover, button.picker-option.is-selected:hover { background: #efe2d6; }
.picker-empty { margin: 0; padding: 8px; color: var(--muted); font-size: .82rem; }
.selected-models {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 8px;
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
}
.selected-models:empty { display: none; }
.selected-models li {
  min-width: 0;
  max-width: 100%;
  padding: 3px 8px;
  border: 1px solid var(--line);
  background: var(--field);
  font-family: var(--mono);
  font-size: .82rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.inline { display: flex; gap: 8px; align-items: center; }
.inline input { flex: 1; }
.inline button { flex: 0 0 auto; }
.actions { display: flex; align-items: center; gap: 12px; min-width: 0; }
.actions .status { margin: 0; }
.status { margin: 8px 0; min-height: 1.3em; }
.status.error { color: var(--bad); }
.status.ok { color: var(--ok); }
#dash_error { margin: 16px 0 0; font-family: var(--mono); font-size: .84rem; overflow-wrap: anywhere; text-transform: none; letter-spacing: 0; font-weight: 400; }
#dash_error:empty { display: none; }
.experiment { margin-top: 24px; }
.experiment > h2.results-title { margin-top: 24px; }
.facts {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 6px 8px;
  margin: 0 0 16px;
}
.facts div {
  display: flex;
  align-items: baseline;
  gap: 6px;
  min-width: 0;
  max-width: 100%;
  padding: 3px 8px;
  border: 1px solid var(--line);
  background: var(--field);
}
.facts dt {
  margin: 0;
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  white-space: nowrap;
}
.facts dd {
  margin: 0;
  color: var(--ink);
  font-family: var(--mono);
  font-size: .82rem;
  line-height: 1.35;
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
a.back-link {
  display: inline-block;
  margin: 0 0 10px;
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  text-decoration: none;
}
a.back-link:hover { color: var(--ink); }
.candidate-models { margin: 0 0 24px; }
.candidate-models[hidden] { display: none; }
.candidate-models ul {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.candidate-models li {
  min-width: 0;
  max-width: 100%;
  padding: 3px 8px;
  border: 1px solid var(--line);
  background: var(--field);
  font-family: var(--mono);
  font-size: .82rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.mono { font-family: var(--mono); font-size: .86rem; overflow-wrap: anywhere; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.head-right { display: flex; align-items: baseline; gap: 16px; }
.run-status {
  color: var(--ink);
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.run-status.running { color: var(--accent); }
.run-status.complete { color: var(--ok); }
.run-status.incomplete { color: var(--warn); }
.run-status.failed { color: var(--bad); }
.elapsed { font-family: var(--mono); font-size: .86rem; font-variant-numeric: tabular-nums; color: var(--muted); }
.table-scroll { overflow-x: auto; }
table.sheet { width: 100%; border-collapse: collapse; background: transparent; }
table.sheet th {
  position: sticky;
  top: 0;
  z-index: 1;
  text-align: left;
  padding: 6px 12px 6px 0;
  border-bottom: 1px solid var(--line);
  background: var(--bg);
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  white-space: nowrap;
}
table.sheet td {
  padding: 8px 12px 8px 0;
  border-bottom: 1px solid var(--line);
  vertical-align: baseline;
}
input[type=checkbox] { width: auto; margin: 0; accent-color: var(--accent); }
#ov_resolved.hot { color: var(--ok); }
#ov_failed.hot { color: var(--bad); }
.progress-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  margin: 0;
}
.section-line { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }
.section-line h2 { margin: 0; }
.section-line .meta { margin: 0; color: var(--muted); font-family: var(--mono); font-size: .82rem; letter-spacing: 0; text-transform: none; }
.bar { height: 2px; margin-top: 8px; overflow: hidden; border-radius: 0; background: var(--line); }
.bar > span { display: block; height: 100%; width: 0; background: var(--accent); }
#dashboard.is-complete .bar > span { background: var(--ok); }
#dashboard.is-incomplete .bar > span { background: var(--warn); }
#dashboard.is-failed .bar > span { background: var(--bad); }
.results { margin-top: 8px; }
#elim_section, #hill_section { margin: 0 0 24px; }
#elim_status, #hill_status, #bracket_status { margin: 0 0 8px; color: var(--muted); font-family: var(--mono); font-size: .82rem; letter-spacing: 0; text-transform: none; }
#elim_status:empty, #hill_status:empty, #bracket_status:empty { display: none; }
#elim_section[hidden], #hill_section[hidden], .elim-board[hidden], #bracket_table[hidden] { display: none; }
.elim-board { display: flex; flex-direction: column; gap: 16px; }
.elim-task { margin: 0 0 8px; color: var(--muted); font-family: var(--mono); font-size: .82rem; }
.elim-columns { display: flex; align-items: stretch; gap: 16px; overflow-x: auto; }
.elim-round { flex: 1 0 12.5rem; min-width: 12.5rem; max-width: 20rem; display: flex; flex-direction: column; }
.elim-round h3 { margin: 0 0 8px; }
.elim-matches { flex: 1; display: flex; flex-direction: column; justify-content: space-around; gap: 8px; }
.elim-match { min-width: 0; padding: 6px 8px; border: 1px solid var(--line); background: var(--field); }
.elim-match.is-active { border-color: var(--accent); }
.elim-match.is-draw { border-color: var(--warn); }
.elim-match.is-incomplete { border-color: var(--bad); }
.elim-slot { margin: 0; color: var(--ink); font-family: var(--mono); font-size: .82rem; line-height: 1.35; overflow-wrap: anywhere; }
.elim-slot + .elim-slot { margin-top: 2px; }
.elim-slot.is-winner { font-weight: 600; }
.elim-slot.is-out, .elim-slot.is-open { color: var(--muted); }
.elim-detail { margin: 4px 0 0; color: var(--muted); font-size: .72rem; line-height: 1.35; overflow-wrap: anywhere; }
.elim-match.is-active .elim-detail { color: var(--accent); }
.elim-match.is-draw .elim-detail { color: var(--warn); }
.elim-match.is-incomplete .elim-detail { color: var(--bad); }
.elim-champion { margin: 8px 0 0; color: var(--muted); font-size: .68rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
.elim-champion strong { display: block; margin-top: 4px; color: var(--ink); font-family: var(--mono); font-size: .82rem; font-weight: 600; letter-spacing: 0; text-transform: none; overflow-wrap: anywhere; }
.hill-board { display: flex; flex-direction: column; gap: 16px; }
.hill-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 12px;
  align-items: stretch;
  min-width: 0;
  padding: 6px 8px;
  border: 1px solid var(--line);
  background: var(--field);
}
.hill-card.is-active { border-color: var(--accent); }
.hill-card.is-draw { border-color: var(--warn); }
.hill-card.is-incomplete { border-color: var(--bad); }
.hill-card.is-complete { border-color: var(--ok); }
.hill-side { min-width: 0; }
.hill-label {
  margin: 0 0 4px;
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.hill-vs {
  align-self: center;
  color: var(--muted);
  font: .82rem/1.4 var(--mono);
}
.order-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.order-index {
  width: 1.5rem;
  color: var(--muted);
  font: .82rem/1.4 var(--mono);
}
.order-name {
  flex: 1;
  min-width: 8rem;
  padding: 3px 8px;
  border: 1px solid var(--line);
  background: var(--field);
  font: .82rem/1.4 var(--mono);
}
.order-actions { display: flex; gap: 4px; }
.order-actions button {
  padding: 2px 8px;
  border: 1px solid var(--line);
  background: transparent;
  color: var(--ink);
  font: .72rem/1.4 var(--mono);
  cursor: pointer;
}
.order-actions button:disabled {
  color: var(--muted);
  cursor: default;
}
table.pairs { min-width: 42rem; }
table.pairs th { border-top: 1px solid var(--line); }
table.pairs td { padding: 8px 16px 8px 0; }
table.pairs td.mono { white-space: nowrap; }
.c-status { width: 8.5rem; }
.c-task { width: 8rem; }
.c-outcome { width: 16rem; }
td.state {
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  white-space: nowrap;
}
tr.resolved .state { color: var(--ok); }
tr.failed .state { color: var(--bad); }
tr.judging .state { color: var(--warn); }
tr.waiting .state { color: var(--muted); }
tr.detail-row td {
  padding: 12px 0 16px 24px;
  border-bottom: 1px solid var(--line);
  background: transparent;
  cursor: auto;
}
.detail { font-size: .88rem; }
.detail dl { display: grid; grid-template-columns: 14rem minmax(0, 1fr); gap: 8px 16px; margin: 0; }
.detail dt {
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.detail dd { margin: 0; min-width: 0; }
.detail pre {
  margin: 0;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 0;
  background: var(--field);
  white-space: pre-wrap;
  font: .8rem/1.4 var(--mono);
}
.launch {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
}
.launch-fields { display: flex; flex-wrap: wrap; gap: 16px 24px; align-items: end; }
.launch-fields select { width: 14rem; }
.launch-main { flex: 1; min-width: 0; }
.matchup-board { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
.matchup-board[hidden], #matchup_field[hidden], #order_board[hidden] { display: none; }
.matchup-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.matchup-row select {
  width: 11rem;
  padding: 3px 8px;
  font: .82rem/1.4 var(--mono);
}
.matchup-vs { color: var(--muted); font: .82rem/1.4 var(--mono); }
.matchup-name {
  padding: 3px 8px;
  border: 1px solid var(--line);
  background: var(--field);
  font: .82rem/1.4 var(--mono);
}
.launch-action { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.launch-action .status { margin: 0; text-align: right; }
.history-section {
  margin-top: 28px;
  padding-top: 20px;
  border-top: 1px solid var(--line);
}
.history-section h2 { margin: 0 0 8px; }
.history-section .sheet th,
.history-section .sheet td { padding: 6px 12px 6px 0; vertical-align: top; }
.history-section .sheet td.actions { white-space: nowrap; }
.history-section a.report-link { margin: 0; }
:focus-visible { outline: 1px solid var(--accent); outline-offset: 2px; }
@media (max-width: 800px) {
  .wrap { padding: 16px; }
  .workspace {
    grid-template-columns: 1fr;
    grid-template-areas: "provider" "manual" "candidates" "judge" "tasks" "controls";
    gap: 16px;
  }
  .segments { gap: 8px 16px; }
  .provider-fields { flex-direction: column; align-items: stretch; }
  .provider-fields > div { flex-basis: auto; width: 100%; }
  .launch { flex-direction: column; align-items: stretch; }
  .launch-action { align-items: flex-start; }
  .launch-action .status { text-align: left; }
  .detail dl { grid-template-columns: 1fr; }
  .mast { align-items: flex-start; }
}
</style>
</head>
<body>
<div class="wrap">
<div id="config">
<header class="mast">
  <div class="brand">
    <h1>Arena</h1>
    <p class="kicker">Model evaluation workbench</p>
  </div>
</header>

<div class="workspace">
<section class="region provider">
  <h2>Provider</h2>
  <div class="segments" role="group" aria-label="Provider">
    <button type="button" class="segment is-selected" data-provider="openai" aria-pressed="true">OpenAI</button>
    <button type="button" class="segment" data-provider="claude" aria-pressed="false">Claude</button>
    <button type="button" class="segment" data-provider="gemini" aria-pressed="false">Gemini</button>
    <button type="button" class="segment" data-provider="compatible" aria-pressed="false">OpenAI-compatible</button>
  </div>
  <div class="provider-fields">
    <div id="base_url_field" hidden>
      <label for="base_url">Base URL</label>
      <input id="base_url" class="mono" type="text" autocomplete="off" spellcheck="false">
    </div>
    <div>
      <label for="api_key">API key</label>
      <input id="api_key" type="password" autocomplete="off">
    </div>
    <div id="load_actions" class="actions">
      <button id="load" class="secondary" type="button">Load Models</button>
      <p id="status" class="status"></p>
    </div>
  </div>
</section>

<section class="region manual" id="manual_field" hidden>
  <h2>Model ID</h2>
  <p class="meta">This provider does not list models. Enter an ID to add it.</p>
  <div class="inline">
    <input id="manual" class="mono" type="text" placeholder="model-id" autocomplete="off">
    <button id="add" class="secondary" type="button">Add</button>
  </div>
</section>

<section class="region candidates">
  <h2>Candidate models</h2>
  <div class="picker" id="candidate_picker">
    <button id="candidate_toggle" class="picker-toggle" type="button" aria-expanded="false" aria-controls="candidate_menu">
      <span id="candidate_label" class="placeholder">Select candidate models</span>
    </button>
    <div id="candidate_menu" class="picker-menu">
      <input id="candidate_search" type="text" placeholder="Search models..." autocomplete="off">
      <div id="candidate_options" class="picker-options"></div>
    </div>
  </div>
  <ul id="selected_models" class="selected-models"></ul>
</section>

<section class="region judge">
  <h2>Judge model</h2>
  <div class="picker" id="judge_picker">
    <button id="judge_toggle" class="picker-toggle" type="button" aria-expanded="false" aria-controls="judge_menu">
      <span id="judge_label" class="placeholder">No judge</span>
    </button>
    <div id="judge_menu" class="picker-menu">
      <input id="judge_search" type="text" placeholder="Search models..." autocomplete="off">
      <div id="judge_options" class="picker-options"></div>
    </div>
  </div>
  <input id="judge" type="hidden" value="">
</section>

<section class="region tasks">
  <label for="tasks" class="region-title">Tasks JSON</label>
  <textarea id="tasks">[
  {
    "id": "t1",
    "prompt": "Explain Rust ownership in two sentences."
  }
]</textarea>
</section>

<section class="region controls">
  <div class="launch">
    <div class="launch-main">
    <div class="launch-fields">
      <div>
        <label for="tournament">Tournament</label>
        <select id="tournament">
          <option value="round_robin" selected>Round-robin</option>
          <option value="single_elimination">Single-elimination</option>
          <option value="king_of_the_hill">King of the Hill</option>
        </select>
      </div>
      <div>
        <label for="best_of">Best of</label>
        <input id="best_of" type="number" min="1" step="2" value="1">
      </div>
      <div>
        <label for="seed">Bootstrap seed</label>
        <input id="seed" type="text" inputmode="numeric" value="0">
      </div>
      <div id="matchup_field" hidden>
        <label for="matchups">Opening</label>
        <select id="matchups">
          <option value="automatic" selected>Automatic</option>
          <option value="custom">Custom</option>
        </select>
      </div>
    </div>
    <div id="opening_board" class="matchup-board" hidden></div>
    <div id="order_board" class="matchup-board" hidden></div>
    </div>
    <div class="launch-action">
      <button id="start" class="primary" type="button">Run Tournament →</button>
      <p id="run_status" class="status"></p>
    </div>
  </div>
</section>
</div>

<section class="history-section" id="history_section">
  <h2>History</h2>
  <p id="history_status" class="status">Loading saved runs…</p>
  <p id="history_errors" class="status error" hidden></p>
  <div class="table-scroll">
    <table class="sheet">
      <thead>
        <tr>
          <th>Started</th>
          <th>Run</th>
          <th>Models</th>
          <th>Judge</th>
          <th>Format</th>
          <th>Status</th>
          <th>Games</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="history_list"></tbody>
    </table>
  </div>
</section>
</div>

<div id="dashboard" hidden>
  <header class="mast">
    <div class="brand">
      <a class="back-link" href="/">← Back to workbench</a>
      <h1>Arena</h1>
      <p class="kicker">Model evaluation workbench</p>
    </div>
    <div class="head-right">
      <span id="dash_elapsed" class="elapsed">0:00</span>
      <span id="dash_status" class="run-status running">Running</span>
    </div>
  </header>
  <div class="experiment">
  <p id="dash_error" class="status error"></p>
  <p id="report_actions" class="report-actions" hidden>
    <a id="view_report" class="report-link">View report</a>
  </p>
  <h2>Experiment</h2>
  <dl class="facts">
    <div>
      <dt>Candidates</dt>
      <dd id="ov_candidates" class="num">0</dd>
    </div>
    <div>
      <dt>Tasks</dt>
      <dd id="ov_tasks" class="num">0</dd>
    </div>
    <div>
      <dt>Judge</dt>
      <dd id="ov_judge" class="mono">—</dd>
    </div>
    <div>
      <dt>Format</dt>
      <dd id="ov_format">—</dd>
    </div>
    <div>
      <dt>Resolved games</dt>
      <dd id="ov_resolved" class="num">0</dd>
    </div>
    <div>
      <dt>Failed games</dt>
      <dd id="ov_failed" class="num">0</dd>
    </div>
    <div>
      <dt>Submitted games</dt>
      <dd id="ov_submitted" class="num">0</dd>
    </div>
    <div>
      <dt>Planned games</dt>
      <dd id="ov_planned" class="num">0</dd>
    </div>
  </dl>
  <div id="candidate_models" class="candidate-models" hidden>
    <h2>Candidates</h2>
    <ul id="candidate_model_list"></ul>
  </div>
  <div class="progress-grid">
    <div>
      <div class="section-line">
        <h2>Candidates</h2>
        <p id="cand_label" class="meta">0 / 0</p>
      </div>
      <div class="bar"><span id="cand_bar"></span></div>
    </div>
    <div>
      <div class="section-line">
        <h2>Games</h2>
        <p id="pair_label" class="meta">0 submitted · up to 0 planned</p>
      </div>
      <div class="bar"><span id="pair_bar"></span></div>
    </div>
  </div>
  <div id="elim_section" hidden>
    <h2>Single-elimination</h2>
    <p id="elim_status" class="meta"></p>
    <div id="elim_board" class="elim-board"></div>
  </div>
  <div id="hill_section" hidden>
    <h2>King of the Hill</h2>
    <p id="hill_status" class="meta"></p>
    <div id="hill_board" class="hill-board"></div>
  </div>
  <h2 class="results-title">Judged games</h2>
  <div class="results table-scroll">
    <table class="sheet pairs">
      <colgroup>
        <col class="c-status">
        <col class="c-task">
        <col>
        <col class="c-outcome">
      </colgroup>
      <thead>
        <tr>
          <th>Status</th>
          <th>Task</th>
          <th>Models</th>
          <th>Latest game</th>
        </tr>
      </thead>
      <tbody id="pair_list"></tbody>
    </table>
  </div>
  <div id="bracket" hidden>
    <h2 id="bracket_title">Best of</h2>
    <p id="bracket_status" class="meta"></p>
    <div id="bracket_table" class="results table-scroll">
      <table class="sheet">
        <thead>
          <tr>
            <th>Task</th>
            <th>Round</th>
            <th>Series</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody id="bracket_list"></tbody>
      </table>
    </div>
  </div>
  </div>
</div>
</div>

<script>
const models = [];
const selected = new Set();
const openingSlots = [];
const kothOrder = [];
let launchCandidates = [];
let provider = "openai";
let candidateQuery = "";
let judgeQuery = "";
let activeMenu = "";
let judgeValue = "";
let view = null;
let seenSeq = 0;
let elapsedTimer = null;
const openPairs = new Set();

function status(id, message, ok) {
  const el = document.getElementById(id);
  el.textContent = message;
  el.className = "status " + (ok ? "ok" : message ? "error" : "");
}

function apiKey() {
  return document.getElementById("api_key").value;
}

function baseUrl() {
  return document.getElementById("base_url").value.trim();
}

function clearJudgeIfCandidate(id) {
  if (judgeValue !== id) return;
  judgeValue = "";
  document.getElementById("judge").value = "";
}

function toggleCandidate(id, on) {
  if (on) selected.add(id); else selected.delete(id);
  clearJudgeIfCandidate(id);
  render();
}

function placeMenu(menu, toggle) {
  menu.classList.remove("above");
  const rect = toggle.getBoundingClientRect();
  const spaceBelow = window.innerHeight - rect.bottom;
  if (spaceBelow < 180 && rect.top > spaceBelow) menu.classList.add("above");
}

function setMenu(name) {
  if (activeMenu === name) return;
  if (activeMenu === "candidate" || name !== "candidate") {
    candidateQuery = "";
    document.getElementById("candidate_search").value = "";
  }
  if (activeMenu === "judge" || name !== "judge") {
    judgeQuery = "";
    document.getElementById("judge_search").value = "";
  }
  activeMenu = name;
  const candidateOn = name === "candidate";
  const judgeOn = name === "judge";
  document.getElementById("candidate_menu").classList.toggle("is-open", candidateOn);
  document.getElementById("judge_menu").classList.toggle("is-open", judgeOn);
  document.getElementById("candidate_picker").classList.toggle("is-open", candidateOn);
  document.getElementById("judge_picker").classList.toggle("is-open", judgeOn);
  document.querySelector(".candidates").classList.toggle("menu-open", candidateOn);
  document.querySelector(".judge").classList.toggle("menu-open", judgeOn);
  document.getElementById("candidate_toggle").setAttribute("aria-expanded", candidateOn ? "true" : "false");
  document.getElementById("judge_toggle").setAttribute("aria-expanded", judgeOn ? "true" : "false");
  if (candidateOn) {
    const menu = document.getElementById("candidate_menu");
    placeMenu(menu, document.getElementById("candidate_toggle"));
    document.getElementById("candidate_search").focus();
  }
  if (judgeOn) {
    const menu = document.getElementById("judge_menu");
    placeMenu(menu, document.getElementById("judge_toggle"));
    document.getElementById("judge_search").focus();
  }
}

function powerOfTwo(count) {
  return count >= 2 && (count & (count - 1)) === 0;
}

function syncOpeningSlots() {
  const kept = openingSlots.filter((id) => selected.has(id));
  selected.forEach((id) => {
    if (!kept.includes(id)) kept.push(id);
  });
  openingSlots.length = 0;
  openingSlots.push(...kept);
}

function syncKothOrder() {
  const kept = kothOrder.filter((id) => selected.has(id));
  selected.forEach((id) => {
    if (!kept.includes(id)) kept.push(id);
  });
  kothOrder.length = 0;
  kothOrder.push(...kept);
}

function moveKothOrder(index, delta) {
  const next = index + delta;
  if (next < 0 || next >= kothOrder.length) return;
  const swap = kothOrder[index];
  kothOrder[index] = kothOrder[next];
  kothOrder[next] = swap;
  renderMatchups();
}

function setOpeningSlot(index, next) {
  const current = openingSlots[index];
  const other = openingSlots.indexOf(next);
  openingSlots[index] = next;
  if (other !== -1 && other !== index) openingSlots[other] = current;
  renderMatchups();
}

function openingSelect(index) {
  const select = document.createElement("select");
  openingSlots.forEach((id) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    option.selected = id === openingSlots[index];
    select.append(option);
  });
  select.addEventListener("change", () => setOpeningSlot(index, select.value));
  return select;
}

function renderMatchups() {
  const field = document.getElementById("matchup_field");
  const board = document.getElementById("opening_board");
  const orderBoard = document.getElementById("order_board");
  const mode = document.getElementById("matchups");
  const format = document.getElementById("tournament").value;
  const elimination = format === "single_elimination";
  const hill = format === "king_of_the_hill";
  field.hidden = !elimination;
  board.replaceChildren();
  orderBoard.replaceChildren();
  if (!elimination) {
    mode.value = "automatic";
    board.hidden = true;
  }
  if (hill) {
    syncKothOrder();
    orderBoard.hidden = false;
    if (!kothOrder.length) {
      const note = document.createElement("p");
      note.className = "meta";
      note.textContent = "Select candidates in the order they should challenge the hill.";
      orderBoard.append(note);
      return;
    }
    kothOrder.forEach((id, index) => {
      const row = document.createElement("div");
      row.className = "order-row";
      const rank = document.createElement("span");
      rank.className = "order-index";
      rank.textContent = String(index + 1);
      const name = document.createElement("span");
      name.className = "order-name";
      name.textContent = id;
      const actions = document.createElement("div");
      actions.className = "order-actions";
      const up = document.createElement("button");
      up.type = "button";
      up.textContent = "Up";
      up.disabled = index === 0;
      up.addEventListener("click", () => moveKothOrder(index, -1));
      const down = document.createElement("button");
      down.type = "button";
      down.textContent = "Down";
      down.disabled = index === kothOrder.length - 1;
      down.addEventListener("click", () => moveKothOrder(index, 1));
      actions.append(up, down);
      row.append(rank, name, actions);
      orderBoard.append(row);
    });
    return;
  }
  orderBoard.hidden = true;
  if (!elimination) return;
  const custom = mode.value === "custom";
  if (custom) syncOpeningSlots();
  const ids = custom ? openingSlots : [...selected];
  if (!powerOfTwo(ids.length)) {
    board.hidden = !custom;
    if (!custom) return;
    const note = document.createElement("p");
    note.className = "meta";
    note.textContent = "Select 2, 4, 8, or 16 candidates to assign opening matchups.";
    board.append(note);
    return;
  }
  board.hidden = false;
  for (let pair = 0; pair < ids.length; pair += 2) {
    const row = document.createElement("div");
    row.className = "matchup-row";
    const versus = document.createElement("span");
    versus.className = "matchup-vs";
    versus.textContent = "vs";
    if (custom) {
      row.append(openingSelect(pair), versus, openingSelect(pair + 1));
    } else {
      row.append(openingName(ids[pair]), versus, openingName(ids[pair + 1]));
    }
    board.append(row);
  }
}

function openingName(id) {
  const name = document.createElement("span");
  name.className = "matchup-name";
  name.textContent = id;
  return name;
}

function openingPayload() {
  const elimination = document.getElementById("tournament").value === "single_elimination";
  if (!elimination || document.getElementById("matchups").value !== "custom") return null;
  syncOpeningSlots();
  const pairs = [];
  for (let index = 0; index < openingSlots.length; index += 2) {
    pairs.push({
      model_a: openingSlots[index] || "",
      model_b: openingSlots[index + 1] || "",
    });
  }
  return pairs;
}

function render() {
  const label = document.getElementById("candidate_label");
  if (selected.size === 0) {
    label.textContent = "Select candidate models";
    label.className = "placeholder";
  } else if (selected.size === 1) {
    label.textContent = "1 model selected";
    label.className = "";
  } else {
    label.textContent = selected.size + " models selected";
    label.className = "";
  }
  const chips = document.getElementById("selected_models");
  chips.replaceChildren();
  selected.forEach((id) => {
    const item = document.createElement("li");
    item.textContent = id;
    chips.append(item);
  });
  const options = document.getElementById("candidate_options");
  options.replaceChildren();
  const visible = models.filter((id) => !candidateQuery || id.toLowerCase().includes(candidateQuery));
  if (visible.length === 0) {
    const empty = document.createElement("p");
    empty.className = "picker-empty";
    empty.textContent = models.length === 0 ? "No models yet" : "No matching models";
    options.append(empty);
  } else {
    visible.forEach((id) => {
      const row = document.createElement("label");
      row.className = "picker-option" + (selected.has(id) ? " is-selected" : "");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = selected.has(id);
      box.addEventListener("change", () => toggleCandidate(id, box.checked));
      const text = document.createElement("span");
      text.textContent = id;
      row.append(box, text);
      options.append(row);
    });
  }

  if (judgeValue && selected.has(judgeValue)) judgeValue = "";
  document.getElementById("judge").value = judgeValue;
  const judgeLabel = document.getElementById("judge_label");
  if (judgeValue) {
    judgeLabel.textContent = judgeValue;
    judgeLabel.className = "mono-label";
  } else {
    judgeLabel.textContent = "No judge";
    judgeLabel.className = "placeholder";
  }
  const judgeOptions = document.getElementById("judge_options");
  judgeOptions.replaceChildren();
  const none = document.createElement("button");
  none.type = "button";
  none.className = "picker-option plain" + (judgeValue ? "" : " is-selected");
  none.textContent = "No judge";
  none.addEventListener("click", () => {
    judgeValue = "";
    document.getElementById("judge").value = "";
    setMenu("");
    render();
  });
  judgeOptions.append(none);
  const judgeNeedle = judgeQuery.trim().toLowerCase();
  const judgeModels = models.filter((id) => !selected.has(id) && (!judgeNeedle || id.toLowerCase().includes(judgeNeedle)));
  if (models.length === 0) {
    const empty = document.createElement("p");
    empty.className = "picker-empty";
    empty.textContent = "No models yet";
    judgeOptions.append(empty);
  }
  judgeModels.forEach((id) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "picker-option" + (judgeValue === id ? " is-selected" : "");
    button.textContent = id;
    button.addEventListener("click", () => {
      judgeValue = id;
      document.getElementById("judge").value = id;
      setMenu("");
      render();
    });
    judgeOptions.append(button);
  });
  renderMatchups();
}

function applyProvider(next) {
  provider = next;
  models.length = 0;
  selected.clear();
  kothOrder.length = 0;
  openingSlots.length = 0;
  judgeValue = "";
  document.getElementById("judge").value = "";
  candidateQuery = "";
  judgeQuery = "";
  document.getElementById("candidate_search").value = "";
  document.getElementById("judge_search").value = "";
  document.getElementById("base_url").value = "";
  document.getElementById("api_key").value = "";
  status("status", "", true);
  document.querySelectorAll(".segment").forEach((button) => {
    const on = button.dataset.provider === next;
    button.classList.toggle("is-selected", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
  const discovery = next === "openai" || next === "compatible";
  document.getElementById("base_url_field").hidden = next !== "compatible";
  document.getElementById("load_actions").hidden = !discovery;
  document.getElementById("manual").value = "";
  document.getElementById("manual_field").hidden = discovery;
  setMenu("");
  render();
}

function addModel(id) {
  const trimmed = id.trim();
  if (!trimmed) return;
  if (!models.includes(trimmed)) models.push(trimmed);
  selected.add(trimmed);
  render();
}

function pairKey(row) {
  return row.task_id + "\0" + row.model_a + "\0" + row.model_b;
}

function findPair(taskId, modelA, modelB) {
  return view.pairs.find((item) =>
    item.task_id === taskId &&
    ((item.model_a === modelA && item.model_b === modelB) ||
     (item.model_a === modelB && item.model_b === modelA))
  );
}

function decisionText(decision, modelA, modelB) {
  if (decision === "a") return modelA + " wins";
  if (decision === "b") return modelB + " wins";
  if (decision === "draw") return "Game draw";
  return decision || "—";
}

function seriesGames(row) {
  if (row.games_resolved != null) return row.games_resolved;
  if (row.games != null) return row.games;
  return null;
}

function seriesScore(row) {
  if (row.wins_a == null || row.wins_b == null) return null;
  return row.wins_a + "–" + row.wins_b;
}

function compactOutcome(row) {
  const games = seriesGames(row);
  const score = seriesScore(row);
  const gamesText = games == null ? "games unknown" : games + (games === 1 ? " game judged" : " games judged");
  if (row.seeded_fallback && row.series_winner) {
    return row.series_winner + " advances by seeded fallback · " + gamesText;
  }
  if (row.series_winner) {
    let text = row.series_winner + " wins series";
    if (score) text += " (" + score + ")";
    text += " · " + gamesText;
    return text;
  }
  if (row.series_failed || row.status === "failed") {
    const first = row.failure && row.failure.orientations && row.failure.orientations[0];
    const fail = first && first.error ? first.kind + ": " + first.error : "Failed";
    return fail + (games ? " · " + gamesText : "");
  }
  if (row.series_draw) {
    return "Series draw · " + gamesText;
  }
  if (row.awaiting_tiebreak) {
    let text = "Series draw · tiebreak";
    if (score) text += " (" + score + ")";
    text += " · " + gamesText;
    return text;
  }
  if (games && games > 0 && row.status === "resolved") {
    let text = "Series in progress";
    if (score) text += " (" + score + ")";
    text += " · " + gamesText;
    if (row.judgment) {
      text += " · latest " + decisionText(row.judgment.winner, row.judgment.model_a, row.judgment.model_b);
    }
    return text;
  }
  if (row.status === "resolved" && row.judgment) {
    return decisionText(row.judgment.winner, row.judgment.model_a, row.judgment.model_b);
  }
  if (row.status === "judging") return "Judging";
  return "Waiting";
}

function statusLabel(status) {
  if (status === "complete") return "Complete";
  if (status === "incomplete") return "Incomplete";
  if (status === "failed") return "Failed";
  return "Running";
}

function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m + ":" + String(s).padStart(2, "0");
}

function startedAtMs() {
  if (!view || !view.started_at) return Date.now();
  const parsed = Date.parse(view.started_at);
  return Number.isNaN(parsed) ? Date.now() : parsed;
}

function renderElapsed() {
  if (!view) return;
  const end = view.status === "running" ? Date.now() : (view.finished_at || Date.now());
  document.getElementById("dash_elapsed").textContent = formatElapsed(end - startedAtMs());
}

function startElapsed() {
  stopElapsed();
  renderElapsed();
  if (view && view.status === "running") {
    elapsedTimer = setInterval(renderElapsed, 1000);
  }
}

function stopElapsed() {
  if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

function maybeMarkJudging() {
  if (!view || view.candidate_completed < view.candidate_total) return;
  view.pairs.forEach((row) => {
    if (row.status === "waiting") row.status = "judging";
  });
}

function applySeriesFields(row, msg) {
  if (!row || !msg) return;
  if (msg.games_resolved != null) {
    row.games_resolved = msg.games_resolved;
    row.games = msg.games_resolved;
  }
  if (msg.wins_a != null) row.wins_a = msg.wins_a;
  if (msg.wins_b != null) row.wins_b = msg.wins_b;
  if (msg.tiebreak_games != null) row.tiebreak_games = msg.tiebreak_games;
  if (msg.series_winner !== undefined) row.series_winner = msg.series_winner || null;
  if (msg.seeded_fallback != null) row.seeded_fallback = !!msg.seeded_fallback;
  if (msg.series_draw != null) row.series_draw = !!msg.series_draw;
  if (msg.series_failed != null) row.series_failed = !!msg.series_failed;
  if (msg.awaiting_tiebreak != null) row.awaiting_tiebreak = !!msg.awaiting_tiebreak;
}

function upsertPair(taskId, modelA, modelB, status, judgment, failure, seriesMsg) {
  const row = findPair(taskId, modelA, modelB);
  if (row) {
    row.status = status;
    row.judgment = judgment || null;
    row.failure = failure || null;
    applySeriesFields(row, seriesMsg);
    return row;
  }
  const created = {
    task_id: taskId,
    model_a: modelA,
    model_b: modelB,
    status,
    games: seriesMsg && seriesMsg.games_resolved != null ? seriesMsg.games_resolved : 1,
    games_resolved: seriesMsg && seriesMsg.games_resolved != null ? seriesMsg.games_resolved : 1,
    wins_a: 0,
    wins_b: 0,
    tiebreak_games: 0,
    series_winner: null,
    seeded_fallback: false,
    series_draw: false,
    series_failed: false,
    awaiting_tiebreak: false,
    judgment: judgment || null,
    failure: failure || null,
  };
  applySeriesFields(created, seriesMsg);
  view.pairs.push(created);
  return created;
}

function appendField(dl, label, value, mono) {
  if (value == null || value === "") return;
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  if (mono) dd.className = "mono";
  dd.textContent = value;
  dl.append(dt, dd);
}

function appendPre(container, label, value) {
  if (value == null || value === "") return;
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  const pre = document.createElement("pre");
  pre.textContent = value;
  dd.append(pre);
  container.append(dt, dd);
}

function pairDetail(row) {
  const wrap = document.createElement("div");
  wrap.className = "detail";
  const dl = document.createElement("dl");
  appendField(dl, "Task", row.task_id, true);
  appendField(dl, "Model A", row.model_a, true);
  appendField(dl, "Model B", row.model_b, true);
  const games = seriesGames(row);
  if (games != null) appendField(dl, "Games judged", String(games), true);
  const score = seriesScore(row);
  if (score) appendField(dl, "Series score", score, true);
  if (row.series_winner) {
    appendField(
      dl,
      "Series result",
      row.seeded_fallback ? row.series_winner + " (seeded fallback)" : row.series_winner + " wins series",
      true
    );
  } else if (row.awaiting_tiebreak) {
    appendField(dl, "Series result", "draw — playing tiebreaks", true);
  } else if (row.series_draw) {
    appendField(dl, "Series result", "series draw", true);
  }
  if (row.judgment) {
    const j = row.judgment;
    appendField(dl, "Game winner", decisionText(j.winner, j.model_a, j.model_b), true);
    appendField(dl, "Agreement", j.agreement ? "agree" : "disagree", false);
    appendField(dl, "Duration", j.duration_ms + " ms", true);
    appendField(dl, "Judge", j.judge_model, true);
    appendField(dl, "AB winner (mapped A/B frame)", j.orientation_ab == null ? null : decisionText(j.orientation_ab, j.model_a, j.model_b), true);
    appendField(dl, "BA winner (mapped A/B frame)", j.orientation_ba == null ? null : decisionText(j.orientation_ba, j.model_a, j.model_b), true);
    appendField(dl, "Reason AB", j.reason_ab, false);
    appendField(dl, "Reason BA", j.reason_ba, false);
    appendPre(dl, "Raw AB (prompt frame)", j.raw_ab);
    appendPre(dl, "Raw BA (prompt frame)", j.raw_ba);
  }
  if (row.failure) {
    appendField(dl, "Judge", row.failure.judge_model, true);
    (row.failure.orientations || []).forEach((item) => {
      appendField(
        dl,
        item.orientation.toUpperCase() + " " + item.kind,
        item.error + " (attempts: " + item.attempts + ")",
        true
      );
    });
    appendField(dl, "AB reason", row.failure.reason_ab, false);
    appendPre(dl, "AB raw completion", row.failure.raw_ab);
    appendField(dl, "BA reason", row.failure.reason_ba, false);
    appendPre(dl, "BA raw completion", row.failure.raw_ba);
  }
  wrap.append(dl);
  return wrap;
}

function candidateModels() {
  const tournament = view.tournament;
  if (tournament && tournament.candidates && tournament.candidates.length) {
    return tournament.candidates;
  }
  if (launchCandidates.length) return launchCandidates.slice();
  const ids = [];
  const seen = new Set();
  (view.pairs || []).forEach((row) => {
    [row.model_a, row.model_b].forEach((id) => {
      if (!id || seen.has(id)) return;
      seen.add(id);
      ids.push(id);
    });
  });
  return ids;
}

function renderCandidates() {
  const section = document.getElementById("candidate_models");
  const list = document.getElementById("candidate_model_list");
  const ids = candidateModels();
  list.replaceChildren();
  if (!ids.length) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  ids.forEach((id) => {
    const item = document.createElement("li");
    item.textContent = id;
    list.append(item);
  });
}

function renderDash() {
  if (!view) return;
  const dash = document.getElementById("dashboard");
  dash.classList.remove("is-running", "is-complete", "is-incomplete", "is-failed");
  dash.classList.add(view.status === "complete" ? "is-complete" : view.status === "incomplete" ? "is-incomplete" : view.status === "failed" ? "is-failed" : "is-running");
  const statusEl = document.getElementById("dash_status");
  statusEl.textContent = statusLabel(view.status);
  statusEl.className = "run-status " + (view.status === "complete" ? "complete" : view.status === "incomplete" ? "incomplete" : view.status === "failed" ? "failed" : "running");
  const note = document.getElementById("dash_error");
  if (view.error) {
    note.textContent = view.error;
    note.className = "status error";
  } else if (view.output_path) {
    note.textContent = "Saved " + view.output_path;
    note.className = "status";
  } else {
    note.textContent = "";
    note.className = "status error";
  }
  const reportActions = document.getElementById("report_actions");
  const viewReport = document.getElementById("view_report");
  const canViewReport =
    (view.status === "complete" || view.status === "incomplete") &&
    !!view.output_path &&
    !!view.run_id;
  reportActions.hidden = !canViewReport;
  if (canViewReport) {
    viewReport.href = "/api/runs/" + encodeURIComponent(view.run_id) + "/report";
  } else {
    viewReport.removeAttribute("href");
  }
  document.getElementById("ov_candidates").textContent = String(view.candidate_count || 0);
  renderCandidates();
  document.getElementById("ov_tasks").textContent = String(view.task_count || 0);
  document.getElementById("ov_judge").textContent = view.judge || "None";
  const format = view.tournament_format || "—";
  document.getElementById("ov_format").textContent = view.best_of > 1
    ? format + " · best of " + view.best_of
    : format;
  const resolved = document.getElementById("ov_resolved");
  resolved.textContent = String(view.resolved_pairs);
  resolved.className = "num" + (view.resolved_pairs > 0 ? " hot" : "");
  const failed = document.getElementById("ov_failed");
  failed.textContent = String(view.failed_pairs);
  failed.className = "num" + (view.failed_pairs > 0 ? " hot" : "");
  const submitted = (view.resolved_pairs || 0) + (view.failed_pairs || 0);
  const planned = view.planned_games != null ? view.planned_games : view.expected_pairs;
  const submittedEl = document.getElementById("ov_submitted");
  submittedEl.textContent = String(submitted);
  submittedEl.className = "num" + (submitted > 0 ? " hot" : "");
  document.getElementById("ov_planned").textContent = String(planned || 0);
  renderElapsed();
  const candDone = view.candidate_completed;
  const candTotal = view.candidate_total;
  document.getElementById("cand_bar").style.width = candTotal ? (100 * candDone / candTotal) + "%" : "0%";
  document.getElementById("cand_label").textContent = candDone + " / " + candTotal;
  const barDenom = Math.max(planned || 0, submitted, 1);
  document.getElementById("pair_bar").style.width = (100 * submitted / barDenom) + "%";
  let gameLabel = submitted + " submitted";
  if (view.status === "running") {
    gameLabel += " · up to " + (planned || 0) + " planned";
  } else if (submitted > (planned || 0)) {
    gameLabel += " · " + (planned || 0) + " planned regulation (includes tiebreaks)";
  } else {
    gameLabel += " · " + (planned || 0) + " planned";
  }
  gameLabel += " · " + (view.resolved_pairs || 0) + " resolved · " + (view.failed_pairs || 0) + " failed";
  document.getElementById("pair_label").textContent = gameLabel;
  const list = document.getElementById("pair_list");
  list.replaceChildren();
  view.pairs.forEach((row) => {
    const key = pairKey(row);
    const expandable = row.status === "resolved" || row.status === "failed";
    const tr = document.createElement("tr");
    tr.className = row.status + (expandable ? " openable" : "") + (openPairs.has(key) ? " open" : "");
    const state = document.createElement("td");
    state.className = "state";
    const stateText = row.status === "resolved" ? "Resolved" : row.status === "failed" ? "Failed" : row.status === "judging" ? "Judging" : "Waiting";
    if (expandable) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "text-button";
      button.textContent = stateText;
      button.addEventListener("click", () => {
        if (openPairs.has(key)) openPairs.delete(key); else openPairs.add(key);
        renderDash();
      });
      state.append(button);
    } else {
      state.textContent = stateText;
    }
    const task = document.createElement("td");
    task.className = "mono";
    task.textContent = row.task_id;
    const modelsCell = document.createElement("td");
    modelsCell.className = "mono";
    modelsCell.textContent = row.model_a + "  ↔  " + row.model_b;
    const out = document.createElement("td");
    out.textContent = compactOutcome(row);
    tr.append(state, task, modelsCell, out);
    list.append(tr);
    if (expandable && openPairs.has(key)) {
      const detail = document.createElement("tr");
      detail.className = "detail-row";
      const td = document.createElement("td");
      td.colSpan = 4;
      td.append(pairDetail(row));
      detail.append(td);
      list.append(detail);
    }
  });
  renderBracket();
}

function isSingleElimination() {
  const format = (view && view.tournament_format) || "";
  if (format === "single-elimination" || format === "single_elimination") return true;
  const tournament = view && view.tournament;
  return !!(tournament && tournament.format === "single_elimination");
}

function isKingOfTheHill() {
  const format = (view && view.tournament_format) || "";
  if (format === "king-of-the-hill" || format === "king_of_the_hill") return true;
  const tournament = view && view.tournament;
  return !!(tournament && tournament.format === "king_of_the_hill");
}

function roundHeading(matchCount) {
  if (matchCount === 1) return "Final";
  if (matchCount === 2) return "Semifinals";
  if (matchCount === 4) return "Quarterfinals";
  return "Round of " + (matchCount * 2);
}

function isPowerOfTwo(count) {
  return count >= 2 && (count & (count - 1)) === 0;
}

function pairsByTask() {
  const tasks = [];
  const index = new Map();
  (view.pairs || []).forEach((row) => {
    if (!index.has(row.task_id)) {
      index.set(row.task_id, tasks.length);
      tasks.push({ task_id: row.task_id, pairs: [] });
    }
    tasks[index.get(row.task_id)].pairs.push(row);
  });
  return tasks;
}

function liveSeriesWinner(row) {
  if (!row) return null;
  if (row.series_winner) return row.series_winner;
  if (row.seeded_fallback && row.series_winner) return row.series_winner;
  if (row.series_draw || row.series_failed || row.awaiting_tiebreak) return null;
  if (view.best_of > 1) return null;
  if (row.status !== "resolved" || !row.judgment) return null;
  if (row.judgment.winner === "a") return row.model_a;
  if (row.judgment.winner === "b") return row.model_b;
  return null;
}

function liveState(row) {
  if (!row || row.status === "waiting") return "pending";
  if (row.series_failed || row.status === "failed") return "incomplete";
  if (row.seeded_fallback || row.series_winner) return "complete";
  if (row.series_draw) return "draw";
  if (row.awaiting_tiebreak) return "active";
  if (row.status === "judging") return "active";
  const games = seriesGames(row);
  if (games && games > 0 && !row.series_winner && !row.series_draw) return "active";
  if (row.status === "resolved" && row.judgment && row.judgment.winner === "draw") return "draw";
  if (row.status === "resolved") return "complete";
  return "pending";
}

function liveDetail(row, state) {
  if (!row || state === "pending") return "Pending";
  if (state === "incomplete") return "Judgment failed";
  const games = seriesGames(row);
  const score = seriesScore(row);
  const gamesSuffix = games == null ? "" : " (" + games + (games === 1 ? " game" : " games") + ")";
  if (row.seeded_fallback && row.series_winner) {
    return row.series_winner + " advances by seeded fallback" + gamesSuffix;
  }
  if (state === "active") {
    if ((!games || games === 0) && row.status === "judging" && !row.awaiting_tiebreak) {
      return "Judging · " + row.model_a + " vs " + row.model_b;
    }
    let text = row.awaiting_tiebreak ? "Tiebreak" : "Series in progress";
    if (score) text += " " + score;
    if (row.model_a && row.model_b) text += " · " + row.model_a + " vs " + row.model_b;
    if (games != null) text += " · " + games + (games === 1 ? " game" : " games");
    return text;
  }
  if (state === "draw") return "Series draw" + gamesSuffix;
  const winner = liveSeriesWinner(row);
  if (!winner) return "Pending";
  const verb = isKingOfTheHill() ? " remains" : " advances";
  return winner + verb + gamesSuffix;
}

function liveCard(row, projected) {
  if (!row) {
    const modelA = projected && projected.modelA;
    const modelB = projected && projected.modelB;
    return {
      modelA: modelA || "Pending",
      modelB: modelB || "Pending",
      winner: null,
      state: "pending",
      detail: "Pending",
      openA: !modelA,
      openB: !modelB,
    };
  }
  const state = liveState(row);
  return {
    modelA: row.model_a,
    modelB: row.model_b,
    winner: state === "complete" ? liveSeriesWinner(row) : null,
    state: state,
    detail: liveDetail(row, state),
    openA: false,
    openB: false,
  };
}

function liveRounds(pairs, field) {
  if (!isPowerOfTwo(field)) {
    return [pairs.map((row) => liveCard(row))];
  }
  const slots = [];
  let cursor = 0;
  let count = field / 2;
  while (count >= 1) {
    const round = [];
    for (let i = 0; i < count; i += 1) {
      round.push(cursor < pairs.length ? pairs[cursor++] : null);
    }
    slots.push(round);
    count /= 2;
  }
  return slots.map((round, roundIndex) => round.map((row, index) => {
    if (row) return liveCard(row);
    const prev = roundIndex > 0 ? slots[roundIndex - 1] : null;
    const modelA = prev ? liveSeriesWinner(prev[index * 2]) : null;
    const modelB = prev ? liveSeriesWinner(prev[index * 2 + 1]) : null;
    return liveCard(null, { modelA: modelA, modelB: modelB });
  }));
}

function liveChampion(rounds) {
  const finalRound = rounds[rounds.length - 1];
  if (!finalRound || finalRound.length !== 1) return null;
  return finalRound[0].winner || null;
}

function recordedState(match) {
  if (match.outcome === "winner") return "complete";
  if (match.outcome === "draw") return "draw";
  if (match.outcome === "judgment_failed" || match.outcome === "incomplete") return "incomplete";
  return "pending";
}

function recordedRounds(task) {
  const byRound = new Map();
  (task.matches || []).forEach((match) => {
    const round = match.round || 1;
    if (!byRound.has(round)) byRound.set(round, []);
    byRound.get(round).push(match);
  });
  return [...byRound.keys()].sort((a, b) => a - b).map((round) => byRound.get(round).map((match) => ({
    modelA: match.model_a,
    modelB: match.model_b,
    winner: match.outcome === "winner" ? match.winner : null,
    state: recordedState(match),
    detail: matchResult(match, true),
    openA: false,
    openB: false,
  })));
}

function elimSlot(name, winner, open) {
  const slot = document.createElement("p");
  slot.className = "elim-slot";
  const text = name || "Pending";
  slot.textContent = text;
  slot.title = text;
  if (open || text === "Pending") slot.classList.add("is-open");
  else if (winner && winner === text) slot.classList.add("is-winner");
  else if (winner) slot.classList.add("is-out");
  return slot;
}

function elimCard(card) {
  const match = document.createElement("div");
  match.className = "elim-match is-" + card.state;
  match.append(
    elimSlot(card.modelA, card.winner, card.openA),
    elimSlot(card.modelB, card.winner, card.openB),
  );
  const detail = document.createElement("p");
  detail.className = "elim-detail";
  detail.textContent = card.detail;
  match.append(detail);
  return match;
}

function championLine(name) {
  const line = document.createElement("p");
  line.className = "elim-champion";
  line.append("Champion");
  const model = document.createElement("strong");
  model.textContent = name;
  model.title = name;
  line.append(model);
  return line;
}

function appendElimBoard(taskId, rounds, champion, multi) {
  const board = document.getElementById("elim_board");
  const block = document.createElement("div");
  if (multi) {
    const label = document.createElement("p");
    label.className = "elim-task";
    label.textContent = taskId;
    block.append(label);
  }
  const columns = document.createElement("div");
  columns.className = "elim-columns";
  rounds.forEach((matches, roundIndex) => {
    const column = document.createElement("section");
    column.className = "elim-round";
    const heading = document.createElement("h3");
    heading.textContent = roundHeading(matches.length);
    const list = document.createElement("div");
    list.className = "elim-matches";
    matches.forEach((card) => list.append(elimCard(card)));
    column.append(heading, list);
    if (champion && roundIndex === rounds.length - 1) column.append(championLine(champion));
    columns.append(column);
  });
  block.append(columns);
  board.append(block);
}

function renderElimSection() {
  const section = document.getElementById("elim_section");
  const board = document.getElementById("elim_board");
  if (!isSingleElimination()) {
    section.hidden = true;
    board.replaceChildren();
    return;
  }
  section.hidden = false;
  board.replaceChildren();
  const tournament = view.tournament;
  const recorded = tournament && tournament.format === "single_elimination" && (tournament.tasks || []).some((task) => (task.matches || []).length);
  const lines = recorded ? (tournament.tasks || []).map((task) => taskStatusLine(task, "single_elimination")) : [];
  document.getElementById("elim_status").textContent = lines.length ? lines.join(" · ") : "";
  if (recorded) {
    const multi = tournament.tasks.length > 1;
    tournament.tasks.forEach((task) => {
      if (!(task.matches || []).length) return;
      appendElimBoard(task.task_id, recordedRounds(task), task.winner || null, multi);
    });
    return;
  }
  const groups = pairsByTask();
  if (!groups.length) {
    document.getElementById("elim_status").textContent = view.status === "running" ? "" : "No matches were played.";
    return;
  }
  const field = view.candidate_count || candidateModels().length;
  const multi = groups.length > 1;
  groups.forEach((group) => {
    const rounds = liveRounds(group.pairs, field);
    appendElimBoard(group.task_id, rounds, liveChampion(rounds), multi);
  });
}

function taskStatusLine(task, format) {
  if (task.winner) {
    if (format === "king_of_the_hill" || format === "king-of-the-hill") {
      return task.task_id + ": " + task.winner + " holds the hill";
    }
    return task.task_id + ": " + task.winner + " won";
  }
  if (task.status === "draw") return task.task_id + ": series draw, no winner";
  if (task.status === "incomplete") return task.task_id + ": no winner";
  return task.task_id + ": " + (task.status || "");
}

function matchResult(match, format) {
  let text = "—";
  const elimination = format === true || format === "single_elimination" || format === "single-elimination";
  const hill = format === "king_of_the_hill" || format === "king-of-the-hill";
  if (match.outcome === "winner" && match.winner) {
    text = match.winner + (elimination ? " advances" : hill ? " remains" : " won");
  } else if (match.outcome === "draw") {
    text = "Series draw";
  } else if (match.outcome === "judgment_failed") {
    text = "Judgment failed";
  } else if (match.outcome === "incomplete") {
    text = "Incomplete";
  } else if (match.outcome) {
    text = match.outcome;
  }
  const games = match.games || [];
  if (!games.length && !match.seeded_fallback) return text;
  if (!games.length) return text + " (seeded fallback)";
  text += " (" + games.length + (games.length === 1 ? " game" : " games");
  const tiebreaks = games.filter((game) => game.tiebreak).length;
  if (tiebreaks) {
    text += ", " + tiebreaks + (tiebreaks === 1 ? " tie-break" : " tie-breaks");
  }
  if (match.seeded_fallback) text += ", seeded fallback";
  text += ")";
  return text;
}

function findTaskPair(pairs, modelA, modelB) {
  return (pairs || []).find((row) =>
    (row.model_a === modelA && row.model_b === modelB) ||
    (row.model_a === modelB && row.model_b === modelA)
  );
}

function hillFromMatches(candidates, matches) {
  if (!candidates.length) {
    return { holder: null, challenger: null, champion: null, state: "pending", detail: "Pending", openChallenger: true };
  }
  if (candidates.length === 1) {
    return {
      holder: candidates[0],
      challenger: null,
      champion: candidates[0],
      state: "complete",
      detail: "Champion",
      openChallenger: false,
    };
  }
  let holder = candidates[0];
  let next = 1;
  for (const match of matches || []) {
    const challenger = candidates[next];
    if (!challenger) break;
    const state = recordedState(match);
    if (state === "complete" && match.winner) {
      holder = match.winner;
      next += 1;
      continue;
    }
    return {
      holder,
      challenger,
      champion: null,
      state,
      detail: matchResult(match, "king_of_the_hill"),
      openChallenger: false,
    };
  }
  if (next >= candidates.length) {
    return {
      holder,
      challenger: null,
      champion: holder,
      state: "complete",
      detail: "Champion",
      openChallenger: false,
    };
  }
  return {
    holder,
    challenger: candidates[next],
    champion: null,
    state: "pending",
    detail: "Pending",
    openChallenger: true,
  };
}

function hillFromPairs(candidates, pairs) {
  if (!candidates.length) {
    return { holder: null, challenger: null, champion: null, state: "pending", detail: "Pending", openChallenger: true };
  }
  if (candidates.length === 1) {
    return {
      holder: candidates[0],
      challenger: null,
      champion: candidates[0],
      state: "complete",
      detail: "Champion",
      openChallenger: false,
    };
  }
  let holder = candidates[0];
  let next = 1;
  while (next < candidates.length) {
    const challenger = candidates[next];
    const row = findTaskPair(pairs, holder, challenger);
    if (!row || row.status === "waiting") {
      return {
        holder,
        challenger,
        champion: null,
        state: "pending",
        detail: "Pending",
        openChallenger: !row,
      };
    }
    if (row.series_failed || row.status === "failed") {
      return {
        holder,
        challenger,
        champion: null,
        state: "incomplete",
        detail: "Judgment failed",
        openChallenger: false,
      };
    }
    const state = liveState(row);
    if (state === "active" || row.status === "judging" || row.awaiting_tiebreak) {
      return {
        holder,
        challenger,
        champion: null,
        state: "active",
        detail: liveDetail(row, "active"),
        openChallenger: false,
      };
    }
    const winner = liveSeriesWinner(row);
    if (!winner) {
      return {
        holder,
        challenger,
        champion: null,
        state: state === "draw" ? "draw" : "active",
        detail: liveDetail(row, state === "draw" ? "draw" : "active"),
        openChallenger: false,
      };
    }
    holder = winner;
    next += 1;
  }
  return {
    holder,
    challenger: null,
    champion: holder,
    state: "complete",
    detail: "Champion",
    openChallenger: false,
  };
}

function appendHillBoard(taskId, hill, multi) {
  const board = document.getElementById("hill_board");
  const block = document.createElement("div");
  if (multi) {
    const label = document.createElement("p");
    label.className = "elim-task";
    label.textContent = taskId;
    block.append(label);
  }
  if (hill.champion) {
    block.append(championLine(hill.champion));
    board.append(block);
    return;
  }
  const card = document.createElement("div");
  card.className = "hill-card is-" + hill.state;
  const holder = document.createElement("div");
  holder.className = "hill-side";
  const holderLabel = document.createElement("p");
  holderLabel.className = "hill-label";
  holderLabel.textContent = "Hill holder";
  holder.append(holderLabel, elimSlot(hill.holder, hill.holder, !hill.holder));
  const versus = document.createElement("span");
  versus.className = "hill-vs";
  versus.textContent = "vs";
  const challenger = document.createElement("div");
  challenger.className = "hill-side";
  const challengerLabel = document.createElement("p");
  challengerLabel.className = "hill-label";
  challengerLabel.textContent = "Challenger";
  challenger.append(
    challengerLabel,
    elimSlot(hill.challenger, null, hill.openChallenger || !hill.challenger)
  );
  card.append(holder, versus, challenger);
  const detail = document.createElement("p");
  detail.className = "elim-detail";
  detail.style.gridColumn = "1 / -1";
  detail.textContent = hill.detail;
  card.append(detail);
  block.append(card);
  board.append(block);
}

function renderHillSection() {
  const section = document.getElementById("hill_section");
  const board = document.getElementById("hill_board");
  if (!isKingOfTheHill()) {
    section.hidden = true;
    board.replaceChildren();
    return;
  }
  section.hidden = false;
  board.replaceChildren();
  const tournament = view.tournament;
  const recorded = tournament && tournament.format === "king_of_the_hill" && (tournament.tasks || []).some((task) => (task.matches || []).length || task.winner);
  const lines = recorded ? (tournament.tasks || []).map((task) => taskStatusLine(task, "king_of_the_hill")) : [];
  document.getElementById("hill_status").textContent = lines.length ? lines.join(" · ") : "";
  if (recorded) {
    const multi = tournament.tasks.length > 1;
    tournament.tasks.forEach((task) => {
      appendHillBoard(task.task_id, hillFromMatches(tournament.candidates || [], task.matches || []), multi);
    });
    return;
  }
  const groups = pairsByTask();
  const candidates = candidateModels();
  if (!groups.length) {
    if (candidates.length) {
      appendHillBoard("t1", hillFromPairs(candidates, []), false);
      return;
    }
    document.getElementById("hill_status").textContent = view.status === "running" ? "" : "No matches were played.";
    return;
  }
  const multi = groups.length > 1;
  groups.forEach((group) => {
    appendHillBoard(group.task_id, hillFromPairs(candidates, group.pairs), multi);
  });
}

function renderBracket() {
  renderElimSection();
  renderHillSection();
  const section = document.getElementById("bracket");
  const tournament = view && view.tournament;
  const series = tournament && tournament.best_of > 1;
  if (isSingleElimination() || isKingOfTheHill() || !tournament || !series) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  document.getElementById("bracket_title").textContent = "Best of " + tournament.best_of;
  const lines = (tournament.tasks || []).map((task) => taskStatusLine(task, tournament.format));
  document.getElementById("bracket_status").textContent = lines.length
    ? lines.join(" · ")
    : "No series were played.";
  const body = document.getElementById("bracket_list");
  body.replaceChildren();
  (tournament.tasks || []).forEach((task) => {
    (task.matches || []).forEach((match) => {
      const tr = document.createElement("tr");
      [task.task_id, String(match.round), match.model_a + "  ↔  " + match.model_b, matchResult(match, false)].forEach((text, index) => {
        const td = document.createElement("td");
        if (index < 3) td.className = "mono";
        td.textContent = text;
        tr.append(td);
      });
      body.append(tr);
    });
  });
}

function applyEvent(msg) {
  if (msg.type !== "snapshot" && msg.seq && msg.seq <= seenSeq) return;
  if (msg.seq) seenSeq = msg.seq;
  switch (msg.type) {
    case "snapshot":
      view = msg;
      seenSeq = msg.seq || 0;
      if (view.status !== "running") view.finished_at = Date.now();
      startElapsed();
      break;
    case "candidate_finished":
      view.candidate_completed += 1;
      maybeMarkJudging();
      break;
    case "pair_resolved": {
      const j = msg.judgment;
      view.resolved_pairs += 1;
      upsertPair(j.task_id, j.model_a, j.model_b, "resolved", j, null, msg);
      break;
    }
    case "pair_failed": {
      const f = msg.failure;
      view.failed_pairs += 1;
      upsertPair(f.task_id, f.model_a, f.model_b, "failed", null, f, msg);
      break;
    }
    case "run_complete":
      view.status = msg.failed_pairs > 0 ? "incomplete" : "complete";
      if (msg.planned_games != null) view.planned_games = msg.planned_games;
      view.expected_pairs = msg.expected_pairs;
      view.resolved_pairs = msg.resolved_pairs;
      view.failed_pairs = msg.failed_pairs;
      view.output_path = msg.output_path || null;
      if (msg.tournament) view.tournament = msg.tournament;
      view.finished_at = Date.now();
      stopElapsed();
      loadHistory();
      break;
    case "run_failed":
      view.status = "failed";
      view.error = msg.error;
      view.finished_at = Date.now();
      stopElapsed();
      break;
    default:
      return;
  }
  renderDash();
}

function showDashboard(runId) {
  document.getElementById("config").hidden = true;
  document.getElementById("dashboard").hidden = false;
  document.getElementById("report_actions").hidden = true;
  openPairs.clear();
  view = {
    run_id: runId,
    status: "running",
    seq: 0,
    candidate_count: 0,
    task_count: 0,
    judge: null,
    started_at: new Date().toISOString(),
    candidate_completed: 0,
    candidate_total: 0,
    planned_games: 0,
    expected_pairs: 0,
    resolved_pairs: 0,
    failed_pairs: 0,
    pairs: [],
    error: null,
    output_path: null,
    tournament_format: "",
    best_of: 1,
    seed: 0,
    tournament: null,
  };
  seenSeq = 0;
  startElapsed();
  renderDash();
  const source = new EventSource("/api/runs/" + encodeURIComponent(runId) + "/events");
  source.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    applyEvent(msg);
    if (msg.type === "run_complete" || msg.type === "run_failed" ||
        (msg.type === "snapshot" && msg.status && msg.status !== "running")) {
      source.close();
      stopElapsed();
      renderElapsed();
    }
  };
}

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => applyProvider(button.dataset.provider));
});
document.getElementById("candidate_toggle").addEventListener("click", () => {
  setMenu(activeMenu === "candidate" ? "" : "candidate");
  render();
});
document.getElementById("judge_toggle").addEventListener("click", () => {
  setMenu(activeMenu === "judge" ? "" : "judge");
  render();
});
document.getElementById("candidate_search").addEventListener("input", () => {
  candidateQuery = document.getElementById("candidate_search").value.trim().toLowerCase();
  render();
});
document.getElementById("judge_search").addEventListener("input", () => {
  judgeQuery = document.getElementById("judge_search").value;
  render();
});
document.addEventListener("click", (event) => {
  if (!activeMenu) return;
  const root = document.getElementById(activeMenu === "candidate" ? "candidate_picker" : "judge_picker");
  if (root.contains(event.target)) return;
  setMenu("");
  render();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !activeMenu) return;
  setMenu("");
  render();
});
function submitManual() {
  addModel(document.getElementById("manual").value);
  document.getElementById("manual").value = "";
}
document.getElementById("add").addEventListener("click", submitManual);
document.getElementById("manual").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  submitManual();
});

document.getElementById("load").addEventListener("click", async () => {
  status("status", "", true);
  const response = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider,
      base_url: provider === "compatible" ? baseUrl() : "",
      api_key: apiKey(),
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    status("status", body.error || "Failed to load models", false);
    return;
  }
  for (const id of body.models || []) {
    if (!models.includes(id)) models.push(id);
  }
  status("status", (body.models || []).length + " models loaded", true);
  render();
});

document.getElementById("tournament").addEventListener("change", renderMatchups);
document.getElementById("matchups").addEventListener("change", renderMatchups);

document.getElementById("start").addEventListener("click", async () => {
  status("run_status", "", true);
  let tasks;
  try {
    tasks = JSON.parse(document.getElementById("tasks").value);
  } catch (error) {
    status("run_status", "Tasks JSON is invalid", false);
    return;
  }
  const format = document.getElementById("tournament").value;
  if (format === "king_of_the_hill") syncKothOrder();
  const modelsPayload = format === "king_of_the_hill" ? kothOrder.slice() : [...selected];
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider,
      base_url: provider === "compatible" ? baseUrl() : "",
      api_key: apiKey(),
      models: modelsPayload,
      judge: document.getElementById("judge").value || null,
      tasks,
      seed: document.getElementById("seed").value.trim() || "0",
      tournament: format,
      best_of: Number(document.getElementById("best_of").value),
      opening_matchups: openingPayload(),
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    status("run_status", body.error || "Failed to start", false);
    return;
  }
  launchCandidates = modelsPayload.slice();
  showDashboard(body.run_id);
});

function historyStatusLabel(run) {
  if (run.complete === true) return "Complete";
  if (run.complete === false) return "Incomplete";
  if (run.tournament_status === "not judged") return "Not judged";
  return run.tournament_status || "Saved";
}

function historyGamesLabel(run) {
  const resolved = run.resolved_pairs;
  const failed = run.failed_pairs;
  const expected = run.expected_pairs;
  if (resolved == null && failed == null && expected == null) return "—";
  const done = (resolved || 0) + (failed || 0);
  if (expected == null) return String(done);
  return done + " / " + expected;
}

function renderHistory(payload) {
  const list = document.getElementById("history_list");
  const statusEl = document.getElementById("history_status");
  const errorsEl = document.getElementById("history_errors");
  list.replaceChildren();
  const runs = (payload && payload.runs) || [];
  const errors = (payload && payload.errors) || [];
  if (!runs.length) {
    statusEl.textContent = "No saved runs yet.";
  } else {
    statusEl.textContent = runs.length === 1 ? "1 saved run" : runs.length + " saved runs";
  }
  if (errors.length) {
    errorsEl.hidden = false;
    errorsEl.textContent = errors.map((item) => item.file + ": " + item.error).join(" · ");
  } else {
    errorsEl.hidden = true;
    errorsEl.textContent = "";
  }
  runs.forEach((run) => {
    const tr = document.createElement("tr");
    const started = document.createElement("td");
    started.className = "mono";
    started.textContent = run.started_at || "—";
    const id = document.createElement("td");
    id.className = "mono";
    id.textContent = run.run_id || "—";
    const models = document.createElement("td");
    models.className = "mono";
    models.textContent = (run.models || []).join(", ") || "—";
    const judge = document.createElement("td");
    judge.className = "mono";
    judge.textContent = run.judge || "None";
    const format = document.createElement("td");
    format.textContent = run.tournament_format || "—";
    const runStatus = document.createElement("td");
    runStatus.textContent = historyStatusLabel(run);
    const games = document.createElement("td");
    games.className = "mono";
    games.textContent = historyGamesLabel(run);
    const actions = document.createElement("td");
    actions.className = "actions";
    const link = document.createElement("a");
    link.className = "report-link";
    link.href = "/api/runs/" + encodeURIComponent(run.run_id) + "/report";
    link.textContent = "View report";
    actions.append(link);
    tr.append(started, id, models, judge, format, runStatus, games, actions);
    list.append(tr);
  });
}

async function loadHistory() {
  const statusEl = document.getElementById("history_status");
  const errorsEl = document.getElementById("history_errors");
  try {
    const response = await fetch("/api/runs");
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      statusEl.textContent = "Failed to load history";
      errorsEl.hidden = false;
      errorsEl.textContent = body.error || "Failed to load history";
      return;
    }
    renderHistory(body);
  } catch (error) {
    statusEl.textContent = "Failed to load history";
    errorsEl.hidden = false;
    errorsEl.textContent = String(error);
  }
}

applyProvider("openai");
loadHistory();
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::pending;
    use std::time::Duration;

    use crate::event::ExperimentEvent;
    use crate::exec::collect_exec_with_events;
    use crate::judge::{
        JudgeDecision, Judgment, JudgmentFailure, JudgmentFailureKind, OrientationFailure,
    };
    use crate::provider::{CompletionRequest, CompletionResponse, ModelProvider, ProviderError};

    fn task() -> Task {
        Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        }
    }

    fn judgment(winner: JudgeDecision, agreement: bool) -> Judgment {
        Judgment {
            task_id: "t1".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            judge_model: ModelId::new("judge"),
            winner,
            reason: "ok".into(),
            duration_ms: 1,
            agreement,
            orientation_ab: None,
            orientation_ba: None,
            reason_ab: None,
            reason_ba: None,
            raw_ab: Some("raw-ab".into()),
            raw_ba: Some("raw-ba".into()),
            raw: None,
        }
    }

    fn sample_config(models: &[&str], judge: Option<&str>) -> ExecConfig {
        build_exec_config(
            models.iter().map(|model| (*model).to_string()).collect(),
            judge.map(str::to_string),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap()
    }

    fn live_run(id: &str, models: &[&str], judge: Option<&str>) -> Arc<LiveRun> {
        Arc::new(LiveRun::from_config(
            id.to_string(),
            &sample_config(models, judge),
        ))
    }

    fn pair_failure() -> JudgmentFailure {
        JudgmentFailure {
            task_id: "t1".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            judge_model: ModelId::new("judge"),
            orientations: vec![OrientationFailure {
                orientation: crate::judge::JudgeOrientation::Ab,
                kind: JudgmentFailureKind::Provider,
                error: "nope".into(),
                attempts: 1,
            }],
            raw_ab: None,
            reason_ab: None,
            raw_ba: None,
            reason_ba: None,
        }
    }

    fn test_output_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("arena-web-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn history_fixture_output(
        started_at: &str,
        models: &[&str],
        judge: Option<&str>,
        complete: Option<bool>,
        format: TournamentFormat,
    ) -> persist::Output {
        let mut run = persist::RunMetadata::new(
            models.iter().map(|model| ModelId::new(*model)).collect(),
            judge.map(ModelId::new),
            None,
            started_at.into(),
            "https://example.test/v1",
        );
        if let Some(complete) = complete {
            let (expected, resolved, failed) = if complete { (1, 1, 0) } else { (1, 0, 1) };
            run = run.with_judge_coverage(expected, resolved, failed);
        }
        persist::Output {
            run,
            tasks: vec![task()],
            results: vec![],
            comparisons: vec![],
            judgments: if complete.is_some() {
                Some(vec![])
            } else {
                None
            },
            judgment_failures: if complete.is_some() {
                Some(vec![])
            } else {
                None
            },
            statistics: if complete.is_some() {
                Some(vec![])
            } else {
                None
            },
            ratings: if complete.is_some() {
                Some(vec![])
            } else {
                None
            },
            tournament: Some(Tournament {
                format,
                candidates: models.iter().map(|model| ModelId::new(*model)).collect(),
                status: if complete == Some(false) {
                    TournamentStatus::Incomplete
                } else if judge.is_some() {
                    TournamentStatus::Complete
                } else {
                    TournamentStatus::NotJudged
                },
                best_of: tournament::DEFAULT_BEST_OF,
                opening_matchups: None,
                tasks: Vec::new(),
            }),
        }
    }

    fn write_history_fixture(dir: &FilePath, run_id: &str, output: &persist::Output) {
        let path = web_output_path(dir, run_id);
        std::fs::write(&path, persist::to_pretty_json(output).unwrap()).unwrap();
    }

    async fn wait_until_idle(state: &AppState) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            {
                let slot = state.run.lock().await;
                if slot.as_ref().is_some_and(|run| !run.is_running()) {
                    return;
                }
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("timed out waiting for run state");
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[test]
    fn duplicate_candidate_models_are_rejected() {
        let error = build_exec_config(
            vec!["m0".into(), "m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap_err();
        match error {
            Error::DuplicateModel(id) => assert_eq!(id, ModelId::new("m0")),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn judge_matching_a_candidate_is_rejected() {
        let error = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("m0".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap_err();
        match error {
            Error::JudgeIsCandidate(id) => assert_eq!(id, ModelId::new("m0")),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[derive(Clone)]
    struct OkProvider;

    impl ModelProvider for OkProvider {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            if request.model == ModelId::new("judge") {
                return Ok(CompletionResponse {
                    text: r#"{"winner":"a","reason":"ok"}"#.into(),
                });
            }
            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[derive(Clone)]
    struct HangProvider;

    impl ModelProvider for HangProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            pending::<std::result::Result<CompletionResponse, ProviderError>>().await
        }
    }

    #[derive(Clone)]
    struct FailProvider;

    impl ModelProvider for FailProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            Err(ProviderError::RequestFailed("upstream down".into()))
        }
    }

    #[tokio::test]
    async fn start_configuration_reaches_collect_exec_with_events() {
        let config = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        assert_eq!(config.models, vec![ModelId::new("m0"), ModelId::new("m1")]);
        assert_eq!(config.judge, Some(ModelId::new("judge")));
        assert_eq!(config.tasks, vec![task()]);
        assert_eq!(config.base_url, "https://example.test/v1");
        assert_eq!(config.seed, 0);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let (output, failed_pairs) = collect_exec_with_events(&OkProvider, &config, tx)
            .await
            .unwrap();
        assert_eq!(failed_pairs, 0);
        assert_eq!(output.results.len(), 2);
        assert_eq!(output.run.models, config.models);
        assert_eq!(output.run.judge, config.judge);
        assert_eq!(output.run.base_url, config.base_url);

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ExperimentEvent::CandidateFinished { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ExperimentEvent::PairResolved { .. }))
        );
        match events.last() {
            Some(ExperimentEvent::RunComplete {
                expected_pairs,
                resolved_pairs,
                failed_pairs,
            }) => {
                assert_eq!(*expected_pairs, 1);
                assert_eq!(*resolved_pairs, 1);
                assert_eq!(*failed_pairs, 0);
            }
            other => panic!("expected RunComplete, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn starting_a_run_returns_a_run_id() {
        let state = Arc::new(AppState::new());
        let config = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        let run_id = start_experiment(
            state.clone(),
            config,
            OkProvider,
            test_output_dir("start-id"),
        )
        .await
        .unwrap();
        assert!(!run_id.is_empty());
        assert!(
            run_id.chars().all(|ch| ch.is_ascii_digit() || ch == '-'),
            "{run_id}"
        );
        assert_eq!(run_id.matches('-').count(), 2, "{run_id}");
        let live = state.run.lock().await.clone().unwrap();
        assert_eq!(live.id, run_id);
    }

    #[test]
    fn web_run_ids_skip_existing_files_after_counter_reset() {
        let dir = test_output_dir("id-skip");
        let taken = format_web_run_id(1_700_000_000_000, 42, 1);
        std::fs::write(web_output_path(&dir, &taken), "{}").unwrap();

        // Restarted process: counter starts at 1 again with the same millis/pid.
        let counter = AtomicU64::new(1);
        let id = allocate_web_run_id_with(&dir, &counter, 1_700_000_000_000, 42);
        assert_eq!(id, format_web_run_id(1_700_000_000_000, 42, 2));
        assert!(!web_output_path(&dir, &id).exists());
    }

    #[test]
    fn successive_web_run_ids_differ() {
        let dir = test_output_dir("id-unique");
        let state = AppState::new();
        let first = state.alloc_id(&dir);
        let second = state.alloc_id(&dir);
        assert_ne!(first, second);
        for id in [&first, &second] {
            assert!(
                id.chars().all(|ch| ch.is_ascii_digit() || ch == '-'),
                "{id}"
            );
            assert_eq!(id.matches('-').count(), 2, "{id}");
        }
    }

    #[test]
    fn write_experiment_does_not_overwrite_existing_file() {
        let dir = test_output_dir("no-overwrite");
        let path = web_output_path(&dir, "kept");
        std::fs::write(&path, "original").unwrap();
        let output = persist::Output {
            run: persist::RunMetadata::new(
                vec![ModelId::new("m0")],
                None,
                None,
                "2026-01-02T03:04:05Z".into(),
                "https://example.test/v1",
            ),
            tasks: vec![],
            results: vec![],
            comparisons: vec![],
            judgments: None,
            judgment_failures: None,
            statistics: None,
            ratings: None,
            tournament: None,
        };
        assert!(write_experiment(&path, &output).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original");
    }

    #[tokio::test]
    async fn a_run_cannot_start_while_another_is_active() {
        let state = Arc::new(AppState::new());
        let config = build_exec_config(
            vec!["m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        start_experiment(state.clone(), config, HangProvider, test_output_dir("busy"))
            .await
            .unwrap();
        let config = build_exec_config(
            vec!["m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        assert!(matches!(
            start_experiment(state, config, HangProvider, test_output_dir("busy-again")).await,
            Err(StartRunError::Busy)
        ));
    }

    #[tokio::test]
    async fn live_run_emits_semantic_client_events() {
        let live = live_run("1", &["m0", "m1"], Some("judge"));
        let mut rx = live.subscribe();
        live.apply(ExperimentEvent::CandidateFinished {
            task_id: "t1".into(),
            model: ModelId::new("m0"),
            duration_ms: 3,
        })
        .await;
        live.apply(ExperimentEvent::PairResolved {
            task_id: "t1".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            judgment: judgment(JudgeDecision::A, true),
        })
        .await;
        live.apply(ExperimentEvent::RunComplete {
            expected_pairs: 1,
            resolved_pairs: 1,
            failed_pairs: 0,
        })
        .await;

        match rx.recv().await.unwrap() {
            ClientEvent::CandidateFinished { task_id, model, .. } => {
                assert_eq!(task_id, "t1");
                assert_eq!(model, "m0");
            }
            other => panic!("expected CandidateFinished, got {other:?}"),
        }
        match rx.recv().await.unwrap() {
            ClientEvent::PairResolved {
                judgment,
                games_resolved,
                wins_a,
                wins_b,
                series_winner,
                seeded_fallback,
                series_draw,
                awaiting_tiebreak,
                ..
            } => {
                assert_eq!(judgment.model_a, ModelId::new("m0"));
                assert_eq!(judgment.model_b, ModelId::new("m1"));
                assert_eq!(judgment.winner, JudgeDecision::A);
                assert!(judgment.agreement);
                assert_eq!(judgment.raw_ab.as_deref(), Some("raw-ab"));
                assert_eq!(games_resolved, 1);
                assert_eq!(wins_a, 1);
                assert_eq!(wins_b, 0);
                assert_eq!(series_winner.as_deref(), Some("m0"));
                assert!(!seeded_fallback);
                assert!(!series_draw);
                assert!(!awaiting_tiebreak);
            }
            other => panic!("expected PairResolved, got {other:?}"),
        }
        match rx.recv().await.unwrap() {
            ClientEvent::RunComplete {
                expected_pairs,
                resolved_pairs,
                failed_pairs,
                ..
            } => {
                assert_eq!(expected_pairs, 1);
                assert_eq!(resolved_pairs, 1);
                assert_eq!(failed_pairs, 0);
            }
            other => panic!("expected RunComplete, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn snapshot_preserves_state_for_late_subscribers() {
        let live = live_run("9", &["m0", "m1"], Some("judge"));
        live.apply(ExperimentEvent::CandidateFinished {
            task_id: "t1".into(),
            model: ModelId::new("m0"),
            duration_ms: 3,
        })
        .await;
        live.apply(ExperimentEvent::PairFailed {
            task_id: "t1".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            failure: pair_failure(),
        })
        .await;

        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.run_id, "9");
                assert_eq!(snapshot.status, RunStatus::Running);
                assert_eq!(snapshot.candidate_completed, 1);
                assert_eq!(snapshot.candidate_total, 2);
                assert_eq!(snapshot.expected_pairs, 1);
                assert_eq!(snapshot.resolved_pairs, 0);
                assert_eq!(snapshot.failed_pairs, 1);
                assert_eq!(snapshot.pairs[0].status, PairStatus::Failed);
                assert!(snapshot.pairs[0].series_failed);
                assert_eq!(snapshot.planned_games, 1);
                assert_eq!(
                    snapshot.pairs[0].failure.as_ref().unwrap().orientations[0].error,
                    "nope"
                );
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_complete_sets_completed_counts() {
        let state = Arc::new(AppState::new());
        let config = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        start_experiment(
            state.clone(),
            config,
            OkProvider,
            test_output_dir("complete-counts"),
        )
        .await
        .unwrap();
        wait_until_idle(&state).await;

        let live = state.run.lock().await.clone().unwrap();
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Complete);
                assert_eq!(snapshot.candidate_completed, 2);
                assert_eq!(snapshot.candidate_total, 2);
                assert_eq!(snapshot.expected_pairs, 1);
                assert_eq!(snapshot.resolved_pairs, 1);
                assert_eq!(snapshot.failed_pairs, 0);
                assert!(snapshot.error.is_none());
                assert_eq!(snapshot.pairs.len(), 1);
                assert_eq!(snapshot.pairs[0].status, PairStatus::Resolved);
                assert_eq!(
                    snapshot.pairs[0].judgment.as_ref().unwrap().winner,
                    JudgeDecision::Draw
                );
                assert_eq!(snapshot.candidate_count, 2);
                assert_eq!(snapshot.task_count, 1);
                assert_eq!(snapshot.judge.as_deref(), Some("judge"));
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn incomplete_judging_is_not_marked_complete() {
        let live = live_run("1", &["m0", "m1"], Some("judge"));
        live.apply(ExperimentEvent::RunComplete {
            expected_pairs: 1,
            resolved_pairs: 0,
            failed_pairs: 1,
        })
        .await;

        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Incomplete);
                assert_eq!(snapshot.expected_pairs, 1);
                assert_eq!(snapshot.resolved_pairs, 0);
                assert_eq!(snapshot.failed_pairs, 1);
                assert!(snapshot.error.is_none());
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn failed_run_does_not_stay_marked_running() {
        let state = Arc::new(AppState::new());
        let config = build_exec_config(
            vec!["m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        start_experiment(
            state.clone(),
            config,
            FailProvider,
            test_output_dir("failed-then-ok"),
        )
        .await
        .unwrap();
        wait_until_idle(&state).await;

        let live = state.run.lock().await.clone().unwrap();
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Failed);
                assert!(snapshot.error.is_some());
            }
            other => panic!("expected snapshot, got {other:?}"),
        }

        let config = build_exec_config(
            vec!["m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        start_experiment(
            state,
            config,
            OkProvider,
            test_output_dir("failed-then-ok-next"),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn snapshot_then_live_events_skip_duplicates() {
        let live = live_run("2", &["m0"], None);
        live.apply(ExperimentEvent::CandidateFinished {
            task_id: "t1".into(),
            model: ModelId::new("m0"),
            duration_ms: 1,
        })
        .await;
        let snapshot = live.snapshot().await;
        let seq = snapshot.seq();
        let mut rx = live.subscribe();
        live.apply(ExperimentEvent::CandidateFinished {
            task_id: "t1".into(),
            model: ModelId::new("m1"),
            duration_ms: 2,
        })
        .await;
        let next = rx.recv().await.unwrap();
        assert!(next.seq() > seq);
        match next {
            ClientEvent::CandidateFinished { model, .. } => assert_eq!(model, "m1"),
            other => panic!("expected later candidate, got {other:?}"),
        }
    }

    #[test]
    fn snapshot_json_is_flat_for_the_browser() {
        let event = ClientEvent::Snapshot {
            snapshot: Box::new(RunSnapshot {
                seq: 2,
                run_id: "7".into(),
                status: RunStatus::Running,
                error: None,
                candidate_count: 2,
                task_count: 1,
                judge: Some("judge".into()),
                started_at: "2026-01-01T00:00:00Z".into(),
                candidate_completed: 1,
                candidate_total: 2,
                planned_games: 1,
                expected_pairs: 1,
                resolved_pairs: 0,
                failed_pairs: 0,
                pairs: Vec::new(),
                output_path: None,
                tournament_format: "round-robin".into(),
                best_of: 1,
                seed: 0,
                tournament: None,
            }),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "snapshot");
        assert_eq!(value["run_id"], "7");
        assert_eq!(value["status"], "running");
        assert_eq!(value["seq"], 2);
        assert_eq!(value["candidate_completed"], 1);
        assert_eq!(value["candidate_count"], 2);
        assert_eq!(value["judge"], "judge");
    }

    #[test]
    fn unordered_pair_count_does_not_include_orientations() {
        let models = [ModelId::new("m0"), ModelId::new("m1"), ModelId::new("m2")];
        let pairs = waiting_pairs(&[task()], &models, TournamentFormat::RoundRobin, None);
        assert_eq!(pairs.len(), 3);
        assert_eq!(exec::expected_pairs(1, 3), 3);
        assert_eq!(exec::expected_pairs(2, 3), 6);
    }

    #[tokio::test]
    async fn candidate_completion_counts_and_judging_phase() {
        let live = live_run("1", &["m0", "m1"], Some("judge"));
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.candidate_completed, 0);
                assert_eq!(snapshot.candidate_total, 2);
                assert_eq!(snapshot.candidate_count, 2);
                assert_eq!(snapshot.task_count, 1);
                assert_eq!(snapshot.pairs.len(), 1);
                assert_eq!(snapshot.pairs[0].status, PairStatus::Waiting);
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
        live.apply(ExperimentEvent::CandidateFinished {
            task_id: "t1".into(),
            model: ModelId::new("m0"),
            duration_ms: 1,
        })
        .await;
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.candidate_completed, 1);
                assert_eq!(snapshot.pairs[0].status, PairStatus::Waiting);
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
        live.apply(ExperimentEvent::CandidateFinished {
            task_id: "t1".into(),
            model: ModelId::new("m1"),
            duration_ms: 1,
        })
        .await;
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.candidate_completed, 2);
                assert_eq!(snapshot.pairs[0].status, PairStatus::Judging);
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolved_pair_identity_is_unordered_and_unique() {
        let live = live_run("1", &["m0", "m1"], Some("judge"));
        let mut swapped = judgment(JudgeDecision::B, true);
        swapped.model_a = ModelId::new("m1");
        swapped.model_b = ModelId::new("m0");
        live.apply(ExperimentEvent::PairResolved {
            task_id: "t1".into(),
            model_a: ModelId::new("m1"),
            model_b: ModelId::new("m0"),
            judgment: swapped,
        })
        .await;
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.pairs.len(), 1);
                assert_eq!(snapshot.resolved_pairs, 1);
                assert_eq!(snapshot.pairs[0].model_a, "m0");
                assert_eq!(snapshot.pairs[0].model_b, "m1");
                assert_eq!(snapshot.pairs[0].status, PairStatus::Resolved);
                assert_eq!(
                    snapshot.pairs[0].judgment.as_ref().unwrap().winner,
                    JudgeDecision::B
                );
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    const SECRET_API_KEY: &str = "super-secret-web-key";

    #[derive(Clone)]
    struct SecretProvider {
        api_key: &'static str,
    }

    impl ModelProvider for SecretProvider {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            let text = if request.model == ModelId::new("judge") {
                r#"{"winner":"a","reason":"ok"}"#.to_string()
            } else {
                request.model.to_string()
            };
            if text.contains(self.api_key) {
                return Err(ProviderError::RequestFailed(
                    "api key leaked into completion".into(),
                ));
            }
            Ok(CompletionResponse { text })
        }
    }

    #[derive(Clone)]
    struct FailJudge;

    impl ModelProvider for FailJudge {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> std::result::Result<CompletionResponse, ProviderError> {
            if request.model == ModelId::new("judge") {
                return Ok(CompletionResponse {
                    text: "not a judgment".into(),
                });
            }
            Ok(CompletionResponse {
                text: request.model.to_string(),
            })
        }
    }

    #[tokio::test]
    async fn completed_web_run_writes_the_exec_output_for_report() {
        let state = Arc::new(AppState::new());
        let dir = test_output_dir("saved-output");
        let config = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        let run_id = start_experiment(
            state.clone(),
            config,
            SecretProvider {
                api_key: SECRET_API_KEY,
            },
            dir.clone(),
        )
        .await
        .unwrap();
        wait_until_idle(&state).await;

        let path = web_output_path(&dir, &run_id);
        let saved = path.display().to_string();
        let live = state.run.lock().await.clone().unwrap();
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Complete);
                assert_eq!(snapshot.output_path.as_deref(), Some(saved.as_str()));
            }
            other => panic!("expected snapshot, got {other:?}"),
        }

        let json = std::fs::read_to_string(&path).unwrap();
        assert!(!json.contains(SECRET_API_KEY), "{json}");
        assert!(!json.contains("api_key"), "{json}");
        assert!(!json.contains("authorization"), "{json}");

        let output = persist::read(&path).unwrap();
        assert_eq!(output.tasks.len(), 1);
        assert_eq!(output.tasks[0].id, "t1");
        assert_eq!(output.results.len(), 2);
        assert_eq!(output.results[0].response.text, "m0");
        assert_eq!(output.results[1].response.text, "m1");
        assert!(output.results[0].evaluation.is_none());
        assert!(output.comparisons.is_empty());
        assert_eq!(output.judgments.as_ref().map(Vec::len), Some(1));
        assert_eq!(output.judgment_failures.as_ref().map(Vec::len), Some(0));
        assert_eq!(output.statistics.as_ref().map(Vec::len), Some(2));
        assert_eq!(output.ratings.as_ref().map(Vec::len), Some(2));
        assert_eq!(output.run.complete, Some(true));
        assert_eq!(output.run.base_url, "https://example.test/v1");
        assert_eq!(
            output.run.models,
            vec![ModelId::new("m0"), ModelId::new("m1")]
        );
        assert_eq!(output.run.judge, Some(ModelId::new("judge")));
        assert!(output.run.bootstrap_seed.is_some());
        assert_eq!(
            output.judgments.as_ref().unwrap()[0].winner,
            JudgeDecision::Draw
        );

        let report = crate::report::from_output(&output);
        assert_eq!(report.summary.complete, Some(true));
        assert_eq!(report.results.len(), 2);
        assert_eq!(report.results[0].response, "m0");
        assert_eq!(report.pairs.len(), 1);
        let html = crate::html::render_with_nav(&report, Some("/"));
        assert!(html.contains("m0"));
        assert!(html.contains("m1"));
        assert!(!html.contains(SECRET_API_KEY));
        assert!(html.contains("← Back to workbench"));

        let live = state.run.lock().await.clone().unwrap();
        let served = render_live_run_report(&live)
            .await
            .expect("completed report");
        assert_eq!(served, html);
        assert!(served.contains("Arena experiment report"));
        assert!(!served.contains(SECRET_API_KEY));

        let listed = list_saved_runs(&dir);
        assert!(
            listed.runs.iter().any(|run| run.run_id == run_id),
            "{:?}",
            listed.runs
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn incomplete_run_report_is_served_from_saved_output() {
        let state = Arc::new(AppState::new());
        let dir = test_output_dir("incomplete-report");
        let config = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        let run_id = start_experiment(state.clone(), config, FailJudge, dir.clone())
            .await
            .unwrap();
        wait_until_idle(&state).await;

        let live = state.run.lock().await.clone().unwrap();
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Incomplete);
                assert!(snapshot.output_path.is_some());
                assert_eq!(snapshot.run_id, run_id);
            }
            other => panic!("expected snapshot, got {other:?}"),
        }

        let served = render_live_run_report(&live)
            .await
            .expect("incomplete report");
        let output = persist::read(web_output_path(&dir, &run_id)).unwrap();
        assert_eq!(output.run.complete, Some(false));
        assert_eq!(
            served,
            html::render_with_nav(&report::from_output(&output), Some("/"))
        );
        assert!(served.contains("INCOMPLETE"));
        assert!(served.contains("← Back to workbench"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn failed_run_without_saved_output_has_no_report() {
        let state = Arc::new(AppState::new());
        let dir = test_output_dir("failed-report");
        let config = build_exec_config(
            vec!["m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        start_experiment(state.clone(), config, FailProvider, dir.clone())
            .await
            .unwrap();
        wait_until_idle(&state).await;

        let live = state.run.lock().await.clone().unwrap();
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Failed);
                assert!(snapshot.output_path.is_none());
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
        let error = render_live_run_report(&live).await.expect_err("no report");
        assert_eq!(error.0, StatusCode::NOT_FOUND);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn running_live_run_report_conflicts() {
        let live = live_run("running-report", &["m0", "m1"], Some("judge"));
        assert!(live.is_running());
        let error = render_live_run_report(&live)
            .await
            .expect_err("still running");
        assert_eq!(error.0, StatusCode::CONFLICT);
    }

    #[test]
    fn workbench_exposes_a_completed_report_link() {
        assert!(PAGE.contains("id=\"report_actions\""));
        assert!(PAGE.contains("id=\"view_report\""));
        assert!(PAGE.contains("/api/runs/\" + encodeURIComponent(view.run_id) + \"/report\""));
        assert!(PAGE.contains("View report"));
    }

    #[test]
    fn workbench_exposes_run_history() {
        assert!(PAGE.contains("id=\"history_section\""));
        assert!(PAGE.contains("id=\"history_list\""));
        assert!(PAGE.contains("loadHistory()"));
        assert!(PAGE.contains("fetch(\"/api/runs\")"));
    }

    #[test]
    fn run_id_validation_rejects_path_traversal() {
        assert!(is_safe_run_id("1700000000000-42-1"));
        assert!(is_safe_run_id("1"));
        assert!(!is_safe_run_id(""));
        assert!(!is_safe_run_id("../secret"));
        assert!(!is_safe_run_id("a/b"));
        assert!(!is_safe_run_id("a\\b"));
        assert!(!is_safe_run_id("a.b"));
        assert!(!is_safe_run_id(".hidden"));
        let dir = test_output_dir("id-validate");
        let err = resolve_saved_run_path(&dir, "../x").expect_err("traversal");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let err = resolve_saved_run_path(&dir, "missing-id").expect_err("missing");
        assert_eq!(err.0, StatusCode::NOT_FOUND);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_lists_newest_saved_runs_first() {
        let dir = test_output_dir("history-order");
        write_history_fixture(
            &dir,
            "older",
            &history_fixture_output(
                "2026-01-01T00:00:00Z",
                &["m0", "m1"],
                Some("judge"),
                Some(true),
                TournamentFormat::RoundRobin,
            ),
        );
        write_history_fixture(
            &dir,
            "newer",
            &history_fixture_output(
                "2026-02-01T00:00:00Z",
                &["a", "b"],
                Some("judge"),
                Some(false),
                TournamentFormat::SingleElimination,
            ),
        );
        std::fs::write(dir.join("notes.txt"), "ignore").unwrap();

        let listed = list_saved_runs(&dir);
        assert!(listed.errors.is_empty(), "{:?}", listed.errors);
        assert_eq!(listed.runs.len(), 2);
        assert_eq!(listed.runs[0].run_id, "newer");
        assert_eq!(listed.runs[0].started_at, "2026-02-01T00:00:00Z");
        assert_eq!(listed.runs[0].complete, Some(false));
        assert_eq!(listed.runs[0].tournament_format, Some("single-elimination"));
        assert_eq!(listed.runs[0].tournament_status, Some("incomplete"));
        assert_eq!(
            listed.runs[0].models,
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(listed.runs[1].run_id, "older");
        assert_eq!(listed.runs[1].started_at, "2026-01-01T00:00:00Z");
        assert_eq!(listed.runs[1].complete, Some(true));
        assert_eq!(listed.runs[1].tournament_format, Some("round-robin"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_reports_malformed_files_without_failing_the_list() {
        let dir = test_output_dir("history-bad");
        write_history_fixture(
            &dir,
            "good",
            &history_fixture_output(
                "2026-03-01T00:00:00Z",
                &["m0"],
                None,
                None,
                TournamentFormat::KingOfTheHill,
            ),
        );
        std::fs::write(dir.join("broken.json"), "{not-json").unwrap();
        std::fs::write(dir.join("also..bad.json"), "{}").unwrap();

        let listed = list_saved_runs(&dir);
        assert_eq!(listed.runs.len(), 1);
        assert_eq!(listed.runs[0].run_id, "good");
        assert!(
            listed
                .errors
                .iter()
                .any(|error| error.file.contains("broken.json")),
            "{:?}",
            listed.errors
        );
        assert!(
            listed
                .errors
                .iter()
                .any(|error| error.file.contains("also..bad.json")),
            "{:?}",
            listed.errors
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saved_run_report_reuses_html_render() {
        let dir = test_output_dir("history-report");
        let output = history_fixture_output(
            "2026-04-01T00:00:00Z",
            &["m0", "m1"],
            Some("judge"),
            Some(true),
            TournamentFormat::RoundRobin,
        );
        write_history_fixture(&dir, "report-me", &output);
        let served = render_saved_run_report(&dir, "report-me").expect("report");
        assert_eq!(
            served,
            html::render_with_nav(&report::from_output(&output), Some("/"))
        );
        assert!(served.contains("Arena experiment report"));
        assert!(served.contains("← Back to workbench"));
        assert!(served.contains("href=\"/\""));
        let loaded = persist::read(web_output_path(&dir, "report-me")).unwrap();
        assert_eq!(loaded.run.started_at, "2026-04-01T00:00:00Z");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_keeps_incomplete_runs_reportable() {
        let dir = test_output_dir("history-incomplete");
        let output = history_fixture_output(
            "2026-05-01T00:00:00Z",
            &["m0", "m1"],
            Some("judge"),
            Some(false),
            TournamentFormat::RoundRobin,
        );
        write_history_fixture(&dir, "inc1", &output);
        let listed = list_saved_runs(&dir);
        assert_eq!(listed.runs.len(), 1);
        assert_eq!(listed.runs[0].complete, Some(false));
        let served = render_saved_run_report(&dir, "inc1").expect("incomplete report");
        assert!(served.contains("INCOMPLETE"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn candidate_failure_does_not_write_an_experiment() {
        let state = Arc::new(AppState::new());
        let dir = test_output_dir("candidate-fail");
        let config = build_exec_config(
            vec!["m0".into()],
            None,
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        let run_id = start_experiment(state.clone(), config, FailProvider, dir.clone())
            .await
            .unwrap();
        wait_until_idle(&state).await;

        let path = web_output_path(&dir, &run_id);
        assert!(!path.exists());
        let live = state.run.lock().await.clone().unwrap();
        match live.snapshot().await {
            ClientEvent::Snapshot { snapshot } => {
                assert_eq!(snapshot.status, RunStatus::Failed);
                assert!(snapshot.output_path.is_none());
                assert!(snapshot.error.is_some());
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn incomplete_judge_run_still_writes_the_exec_output() {
        let state = Arc::new(AppState::new());
        let dir = test_output_dir("incomplete-judge");
        let config = build_exec_config(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        let run_id = start_experiment(state.clone(), config, FailJudge, dir.clone())
            .await
            .unwrap();
        wait_until_idle(&state).await;

        let output = persist::read(web_output_path(&dir, &run_id)).unwrap();
        assert_eq!(output.results.len(), 2);
        assert_eq!(output.run.complete, Some(false));
        assert_eq!(output.run.failed_pairs, Some(1));
        assert_eq!(output.judgments.as_ref().map(Vec::len), Some(0));
        assert_eq!(output.judgment_failures.as_ref().map(Vec::len), Some(1));
        assert_eq!(output.ratings.as_ref().map(Vec::len), Some(2));
        let report = crate::report::from_output(&output);
        assert_eq!(report.summary.complete, Some(false));
        assert_eq!(report.failed_pairs.len(), 1);
        assert!(!report.results.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selected_provider_uses_the_matching_client() {
        let openai = resolve_web_provider(
            WebProviderKind::Openai,
            " super-secret-web-key ",
            "https://ignored.example/v1",
        )
        .unwrap();
        assert_eq!(openai.api_key, "super-secret-web-key");
        assert_eq!(openai.base_url, "https://api.openai.com/v1");
        let openai_client = OpenAICompatibleProvider::new(&openai.api_key, &openai.base_url);
        assert_eq!(openai_client.base_url(), "https://api.openai.com/v1");
        assert!(!format!("{openai_client:?}").contains("super-secret-web-key"));

        let compatible = resolve_web_provider(
            WebProviderKind::Compatible,
            "key",
            " https://deepinfra.example/v1/ ",
        )
        .unwrap();
        assert_eq!(compatible.base_url, "https://deepinfra.example/v1/");
        assert_eq!(
            OpenAICompatibleProvider::new(&compatible.api_key, &compatible.base_url).base_url(),
            "https://deepinfra.example/v1/"
        );

        let claude =
            resolve_web_provider(WebProviderKind::Claude, "key", "https://nope.example").unwrap();
        assert_eq!(claude.base_url, "https://api.anthropic.com/v1");
        assert!(!WebProviderKind::Claude.supports_model_discovery());
        assert_eq!(
            AnthropicProvider::new(&claude.api_key, &claude.base_url).base_url(),
            "https://api.anthropic.com/v1"
        );

        let gemini = resolve_web_provider(WebProviderKind::Gemini, "key", "").unwrap();
        assert_eq!(
            gemini.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert!(!WebProviderKind::Gemini.supports_model_discovery());
        let gemini_client = GeminiProvider::new("super-secret-web-key", &gemini.base_url);
        assert_eq!(
            gemini_client.base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert!(!format!("{gemini_client:?}").contains("super-secret-web-key"));

        assert_eq!(
            resolve_web_provider(WebProviderKind::Compatible, "key", " ").unwrap_err(),
            "base URL is required"
        );
        assert_eq!(
            resolve_web_provider(WebProviderKind::Openai, " ", "").unwrap_err(),
            "API key is required"
        );
        assert!(WebProviderKind::Openai.supports_model_discovery());
        assert!(WebProviderKind::Compatible.supports_model_discovery());
    }

    fn sample_tasks() -> serde_json::Value {
        serde_json::json!([{ "id": "t1", "prompt": "p" }])
    }

    #[tokio::test]
    async fn load_models_rejects_providers_without_discovery() {
        let state = Arc::new(AppState::new());
        for kind in [WebProviderKind::Claude, WebProviderKind::Gemini] {
            let response = load_models(
                State(state.clone()),
                Json(ConnectRequest {
                    provider: kind,
                    base_url: String::new(),
                    api_key: "secret".into(),
                }),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }

        let missing_url = load_models(
            State(state),
            Json(ConnectRequest {
                provider: WebProviderKind::Compatible,
                base_url: "  ".into(),
                api_key: "secret".into(),
            }),
        )
        .await;
        assert_eq!(missing_url.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn start_run_rejects_invalid_provider_configuration() {
        let state = Arc::new(AppState::new());
        let missing_key = start_run(
            State(state.clone()),
            Json(StartRequest {
                provider: WebProviderKind::Openai,
                base_url: "https://evil.example/v1".into(),
                api_key: " ".into(),
                models: vec!["m0".into(), "m1".into()],
                judge: None,
                tasks: sample_tasks(),
                seed: "0".into(),
                tournament: TournamentFormat::RoundRobin,
                best_of: 1,
                opening_matchups: None,
            }),
        )
        .await;
        assert_eq!(missing_key.status(), StatusCode::BAD_REQUEST);

        let missing_url = start_run(
            State(state.clone()),
            Json(StartRequest {
                provider: WebProviderKind::Compatible,
                base_url: String::new(),
                api_key: "secret".into(),
                models: vec!["m0".into(), "m1".into()],
                judge: None,
                tasks: sample_tasks(),
                seed: "0".into(),
                tournament: TournamentFormat::RoundRobin,
                best_of: 1,
                opening_matchups: None,
            }),
        )
        .await;
        assert_eq!(missing_url.status(), StatusCode::BAD_REQUEST);

        let invalid_models = start_run(
            State(state),
            Json(StartRequest {
                provider: WebProviderKind::Claude,
                base_url: String::new(),
                api_key: "secret".into(),
                models: vec!["judge".into()],
                judge: Some("judge".into()),
                tasks: sample_tasks(),
                seed: "0".into(),
                tournament: TournamentFormat::RoundRobin,
                best_of: 1,
                opening_matchups: None,
            }),
        )
        .await;
        assert_eq!(invalid_models.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn seed_decimal_string_preserves_values_above_the_js_safe_integer() {
        let request: StartRequest = serde_json::from_str(
            r#"{"provider":"openai","models":["m0","m1"],"tasks":[{"id":"t1","prompt":"p"}],"seed":"9007199254740993"}"#,
        )
        .unwrap();
        let seed = parse_seed(&request.seed).unwrap();
        assert_eq!(request.tournament, TournamentFormat::RoundRobin);
        assert_eq!(request.best_of, 1);
        assert_eq!(seed, 9_007_199_254_740_993);
        assert_eq!(parse_seed(" 9007199254740993 ").unwrap(), seed);
        assert_eq!(parse_seed("").unwrap(), 0);
        assert_eq!(parse_seed("  ").unwrap(), 0);

        let tasks = task::parse(&serde_json::to_string(&request.tasks).unwrap()).unwrap();
        let config = build_exec_config(
            request.models,
            None,
            tasks,
            "https://example.test/v1".into(),
            seed,
        )
        .unwrap();
        assert_eq!(config.seed, 9_007_199_254_740_993);
    }

    #[test]
    fn parse_seed_rejects_values_that_are_not_u64() {
        assert!(parse_seed("nope").is_err());
        assert!(parse_seed("-1").is_err());
        assert!(parse_seed("1.5").is_err());
        assert!(parse_seed("9007199254740993.0").is_err());
        assert!(parse_seed("18446744073709551616").is_err());
    }

    #[tokio::test]
    async fn start_run_rejects_seed_strings_that_are_not_u64() {
        let state = Arc::new(AppState::new());
        let response = start_run(
            State(state),
            Json(StartRequest {
                provider: WebProviderKind::Openai,
                base_url: "https://example.test/v1".into(),
                api_key: "secret".into(),
                models: vec!["m0".into(), "m1".into()],
                judge: None,
                tasks: sample_tasks(),
                seed: "9007199254740993.0".into(),
                tournament: TournamentFormat::RoundRobin,
                best_of: 1,
                opening_matchups: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn dashboard_pair_preview_follows_the_tournament_format() {
        let round_robin = build_exec_config(
            vec!["m0".into(), "m1".into(), "m2".into(), "m3".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            0,
        )
        .unwrap();
        let live = LiveRun::from_config("rr".into(), &round_robin);
        {
            let state = live.inner.try_lock().expect("lock");
            assert_eq!(state.tournament_format, "round-robin");
            assert_eq!(state.expected_pairs, 6);
            assert_eq!(state.pairs.len(), 6);
        }

        let elimination = build_exec_config_with_format(
            vec!["m0".into(), "m1".into(), "m2".into(), "m3".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            None,
            0,
            TournamentFormat::SingleElimination,
            tournament::DEFAULT_BEST_OF,
            None,
        )
        .unwrap();
        let live = LiveRun::from_config("se".into(), &elimination);
        let state = live.inner.try_lock().expect("lock");
        assert_eq!(state.tournament_format, "single-elimination");
        assert_eq!(state.expected_pairs, 3);
        assert_eq!(state.pairs.len(), 2);
        assert_eq!(state.pairs[0].model_a, "m0");
        assert_eq!(state.pairs[0].model_b, "m1");
        assert_eq!(state.pairs[1].model_a, "m2");
        assert_eq!(state.pairs[1].model_b, "m3");
    }

    #[test]
    fn king_of_the_hill_preview_uses_the_first_ordered_matchup() {
        let config = build_exec_config_with_format(
            vec!["m2".into(), "m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            None,
            0,
            TournamentFormat::KingOfTheHill,
            tournament::DEFAULT_BEST_OF,
            None,
        )
        .unwrap();
        let live = LiveRun::from_config("koth".into(), &config);
        let state = live.inner.try_lock().expect("lock");
        assert_eq!(state.tournament_format, "king-of-the-hill");
        assert_eq!(state.expected_pairs, 2);
        assert_eq!(state.pairs.len(), 1);
        assert_eq!(state.pairs[0].model_a, "m2");
        assert_eq!(state.pairs[0].model_b, "m0");
    }

    #[test]
    fn custom_opening_matchups_preview_the_requested_pairs() {
        let config = build_exec_config_with_format(
            vec!["m0".into(), "m1".into(), "m2".into(), "m3".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            None,
            0,
            TournamentFormat::SingleElimination,
            tournament::DEFAULT_BEST_OF,
            Some(vec![
                OpeningMatchup {
                    model_a: ModelId::new("m0"),
                    model_b: ModelId::new("m2"),
                },
                OpeningMatchup {
                    model_a: ModelId::new("m1"),
                    model_b: ModelId::new("m3"),
                },
            ]),
        )
        .unwrap();
        let live = LiveRun::from_config("custom".into(), &config);
        let state = live.inner.try_lock().expect("lock");
        assert_eq!(state.pairs.len(), 2);
        assert_eq!(state.pairs[0].model_a, "m0");
        assert_eq!(state.pairs[0].model_b, "m2");
        assert_eq!(state.pairs[1].model_a, "m1");
        assert_eq!(state.pairs[1].model_b, "m3");
        assert!(config.opening_matchups.is_some());
    }

    fn elim_match(
        round: u32,
        model_a: &str,
        model_b: &str,
        winner: Option<&str>,
        outcome: tournament::MatchOutcome,
        games: Vec<tournament::SeriesGame>,
        seeded_fallback: bool,
    ) -> tournament::TournamentMatch {
        tournament::TournamentMatch {
            round,
            model_a: ModelId::new(model_a),
            model_b: ModelId::new(model_b),
            winner: winner.map(ModelId::new),
            outcome,
            games,
            seeded_fallback,
        }
    }

    fn round_title(matches: usize) -> String {
        match matches {
            1 => "Final".to_string(),
            2 => "Semifinals".to_string(),
            4 => "Quarterfinals".to_string(),
            other => format!("Round of {}", other * 2),
        }
    }

    fn card_detail(row: &tournament::TournamentMatch) -> String {
        let mut text = match row.outcome {
            tournament::MatchOutcome::Winner => match &row.winner {
                Some(winner) => format!("{winner} advances"),
                None => "—".to_string(),
            },
            tournament::MatchOutcome::Draw => "Series draw".to_string(),
            tournament::MatchOutcome::JudgmentFailed => "Judgment failed".to_string(),
            tournament::MatchOutcome::Incomplete => "Incomplete".to_string(),
        };
        if row.games.is_empty() && !row.seeded_fallback {
            return text;
        }
        if row.games.is_empty() {
            return format!("{text} (seeded fallback)");
        }
        let games = row.games.len();
        let game_label = if games == 1 { "game" } else { "games" };
        text.push_str(&format!(" ({games} {game_label}"));
        let tiebreaks = row.games.iter().filter(|game| game.tiebreak).count();
        if tiebreaks > 0 {
            let tie_label = if tiebreaks == 1 {
                "tie-break"
            } else {
                "tie-breaks"
            };
            text.push_str(&format!(", {tiebreaks} {tie_label}"));
        }
        if row.seeded_fallback {
            text.push_str(", seeded fallback");
        }
        text.push(')');
        text
    }

    fn board_text(tournament: &Tournament) -> String {
        let mut text = String::new();
        for task in &tournament.tasks {
            let mut rounds: Vec<(u32, Vec<&tournament::TournamentMatch>)> = Vec::new();
            for row in &task.matches {
                if let Some((_, matches)) = rounds.iter_mut().find(|(round, _)| *round == row.round)
                {
                    matches.push(row);
                } else {
                    rounds.push((row.round, vec![row]));
                }
            }
            for (_, matches) in &rounds {
                text.push_str(&round_title(matches.len()));
                text.push('\n');
                for row in matches {
                    text.push_str(&format!(
                        "{} vs {}\n{}\n",
                        row.model_a,
                        row.model_b,
                        card_detail(row)
                    ));
                }
            }
            if let Some(winner) = &task.winner {
                text.push_str(&format!("Champion {winner}\n"));
            }
        }
        text
    }

    #[test]
    fn elimination_board_shows_opening_rounds_advancement_and_fallback() {
        let workbench = PAGE.split("<div id=\"dashboard\"").next().unwrap();
        assert!(!workbench.contains("elim_board"));
        assert!(PAGE.contains("id=\"elim_board\""));
        assert!(PAGE.contains("id=\"elim_section\""));
        assert!(PAGE.contains("id=\"hill_board\""));
        assert!(PAGE.contains("id=\"hill_section\""));
        assert!(PAGE.contains("id=\"order_board\""));
        assert!(PAGE.contains("king_of_the_hill"));
        assert!(PAGE.contains("King of the Hill"));
        assert!(PAGE.contains("return \"Final\""));
        assert!(PAGE.contains("return \"Semifinals\""));
        assert!(PAGE.contains("return \"Quarterfinals\""));
        assert!(PAGE.contains("Champion"));
        assert!(PAGE.contains("Hill holder"));
        assert!(PAGE.contains("Challenger"));
        assert!(PAGE.contains("Pending"));
        assert!(PAGE.contains("Judging"));
        assert!(PAGE.contains("seeded fallback"));
        assert!(PAGE.contains("tie-break"));
        assert!(PAGE.contains("remains"));

        let models = [
            ModelId::new("m0"),
            ModelId::new("m1"),
            ModelId::new("m2"),
            ModelId::new("m3"),
        ];
        let automatic = waiting_pairs(
            &[task()],
            &models,
            TournamentFormat::SingleElimination,
            None,
        );
        assert_eq!(automatic[0].model_a, "m0");
        assert_eq!(automatic[0].model_b, "m1");
        assert_eq!(automatic[1].model_a, "m2");
        assert_eq!(automatic[1].model_b, "m3");
        let automatic_board = Tournament {
            format: TournamentFormat::SingleElimination,
            candidates: models.to_vec(),
            status: tournament::TournamentStatus::Complete,
            best_of: 1,
            tasks: vec![tournament::TaskBracket {
                task_id: "t1".into(),
                status: tournament::TournamentStatus::Complete,
                winner: Some(ModelId::new("m0")),
                matches: vec![
                    elim_match(
                        1,
                        "m0",
                        "m1",
                        Some("m0"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                    elim_match(
                        1,
                        "m2",
                        "m3",
                        Some("m2"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                    elim_match(
                        2,
                        "m0",
                        "m2",
                        Some("m0"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                ],
            }],
            opening_matchups: None,
        };
        let automatic_text = board_text(&automatic_board);
        assert!(
            automatic_text.contains("Semifinals\nm0 vs m1\nm0 advances\nm2 vs m3\nm2 advances\n")
        );
        assert!(automatic_text.contains("Final\nm0 vs m2\nm0 advances\nChampion m0\n"));

        let custom = waiting_pairs(
            &[task()],
            &models,
            TournamentFormat::SingleElimination,
            Some(&[
                OpeningMatchup {
                    model_a: ModelId::new("m0"),
                    model_b: ModelId::new("m2"),
                },
                OpeningMatchup {
                    model_a: ModelId::new("m1"),
                    model_b: ModelId::new("m3"),
                },
            ]),
        );
        assert_eq!(custom[0].model_a, "m0");
        assert_eq!(custom[0].model_b, "m2");
        assert_eq!(custom[1].model_a, "m1");
        assert_eq!(custom[1].model_b, "m3");
        let drawn_game = |tiebreak| tournament::SeriesGame {
            winner: None,
            outcome: tournament::MatchOutcome::Draw,
            tiebreak,
        };
        let custom_board = Tournament {
            format: TournamentFormat::SingleElimination,
            candidates: models.to_vec(),
            status: tournament::TournamentStatus::Complete,
            best_of: 1,
            tasks: vec![tournament::TaskBracket {
                task_id: "t1".into(),
                status: tournament::TournamentStatus::Complete,
                winner: Some(ModelId::new("m0")),
                matches: vec![
                    elim_match(
                        1,
                        "m0",
                        "m2",
                        Some("m0"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                    elim_match(
                        1,
                        "m1",
                        "m3",
                        Some("m1"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                    elim_match(
                        2,
                        "m0",
                        "m1",
                        Some("m0"),
                        tournament::MatchOutcome::Winner,
                        vec![
                            drawn_game(false),
                            drawn_game(true),
                            drawn_game(true),
                            drawn_game(true),
                        ],
                        true,
                    ),
                ],
            }],
            opening_matchups: Some(vec![
                OpeningMatchup {
                    model_a: ModelId::new("m0"),
                    model_b: ModelId::new("m2"),
                },
                OpeningMatchup {
                    model_a: ModelId::new("m1"),
                    model_b: ModelId::new("m3"),
                },
            ]),
        };
        let custom_text = board_text(&custom_board);
        assert!(custom_text.contains("Semifinals\nm0 vs m2\nm0 advances\nm1 vs m3\nm1 advances\n"));
        assert!(custom_text.contains(
            "Final\nm0 vs m1\nm0 advances (4 games, 3 tie-breaks, seeded fallback)\nChampion m0\n"
        ));
        let json = serde_json::to_value(&custom_board).unwrap();
        assert_eq!(json["tasks"][0]["matches"][0]["model_b"], "m2");
        assert_eq!(json["tasks"][0]["matches"][2]["round"], 2);
        assert_eq!(json["tasks"][0]["matches"][2]["seeded_fallback"], true);
        assert_eq!(json["tasks"][0]["matches"][2]["games"][1]["tiebreak"], true);
        assert!(
            json["tasks"][0]["matches"][2]["games"][0]
                .get("tiebreak")
                .is_none()
        );
        assert_eq!(json["tasks"][0]["winner"], "m0");

        let drawn = Tournament {
            format: TournamentFormat::SingleElimination,
            candidates: vec![ModelId::new("m0"), ModelId::new("m1")],
            status: tournament::TournamentStatus::Draw,
            best_of: 1,
            tasks: vec![tournament::TaskBracket {
                task_id: "t1".into(),
                status: tournament::TournamentStatus::Draw,
                winner: None,
                matches: vec![elim_match(
                    1,
                    "m0",
                    "m1",
                    None,
                    tournament::MatchOutcome::Draw,
                    Vec::new(),
                    false,
                )],
            }],
            opening_matchups: None,
        };
        let drawn_text = board_text(&drawn);
        assert!(drawn_text.contains("Final\nm0 vs m1\nSeries draw\n"));
        assert!(!drawn_text.contains("Champion"));
        assert!(!drawn_text.contains("advances"));

        let stopped = Tournament {
            format: TournamentFormat::SingleElimination,
            candidates: vec![ModelId::new("m0"), ModelId::new("m1")],
            status: tournament::TournamentStatus::Incomplete,
            best_of: 1,
            tasks: vec![tournament::TaskBracket {
                task_id: "t1".into(),
                status: tournament::TournamentStatus::Incomplete,
                winner: None,
                matches: vec![elim_match(
                    1,
                    "m0",
                    "m1",
                    None,
                    tournament::MatchOutcome::Incomplete,
                    Vec::new(),
                    false,
                )],
            }],
            opening_matchups: None,
        };
        let stopped_text = board_text(&stopped);
        assert!(stopped_text.contains("Incomplete"));
        assert!(!stopped_text.contains("Champion"));
        assert!(!stopped_text.contains("advances"));
    }

    fn hill_board_text(tournament: &Tournament) -> String {
        let mut text = String::new();
        for task in &tournament.tasks {
            let mut holder = tournament
                .candidates
                .first()
                .map(ToString::to_string)
                .unwrap_or_default();
            let mut next = 1usize;
            for row in &task.matches {
                let challenger = tournament
                    .candidates
                    .get(next)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "Pending".into());
                let detail = match row.outcome {
                    tournament::MatchOutcome::Winner => match &row.winner {
                        Some(winner) => format!("{winner} remains"),
                        None => "—".into(),
                    },
                    tournament::MatchOutcome::Draw => "Series draw".into(),
                    tournament::MatchOutcome::JudgmentFailed => "Judgment failed".into(),
                    tournament::MatchOutcome::Incomplete => "Incomplete".into(),
                };
                text.push_str(&format!(
                    "Hill holder {holder}\nChallenger {challenger}\n{detail}\n"
                ));
                if row.outcome == tournament::MatchOutcome::Winner
                    && let Some(winner) = &row.winner
                {
                    holder = winner.to_string();
                    next = next.saturating_add(1);
                    continue;
                }
                break;
            }
            if let Some(winner) = &task.winner {
                text.push_str(&format!("Champion {winner}\n"));
            } else if next >= tournament.candidates.len() && !holder.is_empty() {
                text.push_str(&format!("Champion {holder}\n"));
            }
        }
        text
    }

    #[test]
    fn king_of_the_hill_board_shows_holder_challenger_and_champion() {
        let models = [ModelId::new("m0"), ModelId::new("m1"), ModelId::new("m2")];
        let preview = waiting_pairs(&[task()], &models, TournamentFormat::KingOfTheHill, None);
        assert_eq!(preview.len(), 1);
        assert_eq!(preview[0].model_a, "m0");
        assert_eq!(preview[0].model_b, "m1");

        let board = Tournament {
            format: TournamentFormat::KingOfTheHill,
            candidates: models.to_vec(),
            status: tournament::TournamentStatus::Complete,
            best_of: 1,
            tasks: vec![tournament::TaskBracket {
                task_id: "t1".into(),
                status: tournament::TournamentStatus::Complete,
                winner: Some(ModelId::new("m1")),
                matches: vec![
                    elim_match(
                        1,
                        "m0",
                        "m1",
                        Some("m1"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                    elim_match(
                        2,
                        "m1",
                        "m2",
                        Some("m1"),
                        tournament::MatchOutcome::Winner,
                        Vec::new(),
                        false,
                    ),
                ],
            }],
            opening_matchups: None,
        };
        let text = hill_board_text(&board);
        assert!(text.contains("Hill holder m0\nChallenger m1\nm1 remains\n"));
        assert!(text.contains("Hill holder m1\nChallenger m2\nm1 remains\n"));
        assert!(text.contains("Champion m1\n"));
        assert!(!text.contains("Semifinals"));
        assert!(!text.contains("advances"));
    }

    #[tokio::test]
    async fn start_run_rejects_single_elimination_without_a_power_of_two_field() {
        let state = Arc::new(AppState::new());
        let response = start_run(
            State(state),
            Json(StartRequest {
                provider: WebProviderKind::Openai,
                base_url: String::new(),
                api_key: "secret".into(),
                models: vec!["m0".into(), "m1".into(), "m2".into()],
                judge: Some("judge".into()),
                tasks: sample_tasks(),
                seed: "0".into(),
                tournament: TournamentFormat::SingleElimination,
                best_of: 1,
                opening_matchups: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn king_of_the_hill_accepts_a_non_power_of_two_field() {
        let config = build_exec_config_with_format(
            vec!["m0".into(), "m1".into(), "m2".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            None,
            0,
            TournamentFormat::KingOfTheHill,
            tournament::DEFAULT_BEST_OF,
            None,
        );
        assert!(config.is_ok());
        let rejected = build_exec_config_with_format(
            vec!["m0".into(), "m1".into(), "m2".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            None,
            0,
            TournamentFormat::SingleElimination,
            tournament::DEFAULT_BEST_OF,
            None,
        );
        assert!(rejected.is_err());
    }

    #[tokio::test]
    async fn start_run_rejects_an_even_best_of() {
        let state = Arc::new(AppState::new());
        let response = start_run(
            State(state),
            Json(StartRequest {
                provider: WebProviderKind::Openai,
                base_url: String::new(),
                api_key: "secret".into(),
                models: vec!["m0".into(), "m1".into()],
                judge: Some("judge".into()),
                tasks: sample_tasks(),
                seed: "0".into(),
                tournament: TournamentFormat::RoundRobin,
                best_of: 2,
                opening_matchups: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn start_run_rejects_duplicate_opening_matchups() {
        let state = Arc::new(AppState::new());
        let response = start_run(
            State(state),
            Json(StartRequest {
                provider: WebProviderKind::Openai,
                base_url: String::new(),
                api_key: "secret".into(),
                models: vec!["m0".into(), "m1".into(), "m2".into(), "m3".into()],
                judge: Some("judge".into()),
                tasks: sample_tasks(),
                seed: "0".into(),
                tournament: TournamentFormat::SingleElimination,
                best_of: 1,
                opening_matchups: Some(vec![
                    OpeningMatchup {
                        model_a: ModelId::new("m0"),
                        model_b: ModelId::new("m1"),
                    },
                    OpeningMatchup {
                        model_a: ModelId::new("m0"),
                        model_b: ModelId::new("m2"),
                    },
                ]),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn best_of_scales_the_dashboard_game_budget() {
        let config = build_exec_config_with_format(
            vec!["m0".into(), "m1".into()],
            Some("judge".into()),
            vec![task()],
            "https://example.test/v1".into(),
            None,
            0,
            TournamentFormat::RoundRobin,
            3,
            None,
        )
        .unwrap();
        let live = LiveRun::from_config("bo3".into(), &config);
        let state = live.inner.try_lock().expect("lock");
        assert_eq!(state.best_of, 3);
        assert_eq!(state.expected_pairs, 3);
        assert_eq!(state.pairs.len(), 1);
    }

    fn live_judgment(
        task_id: &str,
        model_a: &str,
        model_b: &str,
        winner: JudgeDecision,
    ) -> Judgment {
        Judgment {
            task_id: task_id.into(),
            model_a: ModelId::new(model_a),
            model_b: ModelId::new(model_b),
            judge_model: ModelId::new("judge"),
            winner,
            reason: "ok".into(),
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

    fn apply_live_game(
        pairs: &mut Vec<PairRow>,
        model_a: &str,
        model_b: &str,
        winner: JudgeDecision,
        best_of: u32,
        format: &str,
        seed: u64,
    ) -> PairRow {
        apply_resolved_game(
            pairs,
            "t1",
            &ModelId::new(model_a),
            &ModelId::new(model_b),
            live_judgment("t1", model_a, model_b, winner),
            best_of,
            format,
            seed,
        )
    }

    #[test]
    fn live_best_of_three_two_zero_resolves_the_series() {
        let mut pairs = Vec::new();
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "single-elimination",
            0,
        );
        let row = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "single-elimination",
            0,
        );
        assert_eq!(row.games_resolved, 2);
        assert_eq!((row.wins_a, row.wins_b), (2, 0));
        assert_eq!(row.series_winner.as_deref(), Some("m0"));
        assert!(!row.awaiting_tiebreak);
        assert!(!row.seeded_fallback);
        assert!(row.series_complete());

        let late = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::B,
            3,
            "single-elimination",
            0,
        );
        assert_eq!(late.games_resolved, 2);
        assert_eq!((late.wins_a, late.wins_b), (2, 0));
        assert_eq!(late.series_winner.as_deref(), Some("m0"));
    }

    #[test]
    fn live_best_of_three_two_one_resolves_the_series() {
        let mut pairs = Vec::new();
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "king-of-the-hill",
            0,
        );
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::B,
            3,
            "king-of-the-hill",
            0,
        );
        let row = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "king-of-the-hill",
            0,
        );
        assert_eq!(row.games_resolved, 3);
        assert_eq!((row.wins_a, row.wins_b), (2, 1));
        assert_eq!(row.series_winner.as_deref(), Some("m0"));
        assert!(!row.awaiting_tiebreak);
        assert!(!row.series_draw);
    }

    #[test]
    fn live_best_of_three_two_regulation_draws_enter_elimination_tiebreak() {
        for format in ["single-elimination", "king-of-the-hill"] {
            let mut pairs = Vec::new();
            apply_live_game(&mut pairs, "m0", "m1", JudgeDecision::Draw, 3, format, 0);
            let row = apply_live_game(&mut pairs, "m0", "m1", JudgeDecision::Draw, 3, format, 0);
            assert_eq!(row.games_resolved, 2);
            assert_eq!((row.wins_a, row.wins_b), (0, 0));
            assert!(row.awaiting_tiebreak, "{format}");
            assert!(!row.series_draw, "{format}");
            assert!(row.series_winner.is_none(), "{format}");
            assert!(!row.series_complete(), "{format}");
        }
    }

    #[test]
    fn live_decisive_tiebreak_resolves_the_series() {
        let mut pairs = Vec::new();
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::Draw,
            3,
            "single-elimination",
            0,
        );
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::Draw,
            3,
            "single-elimination",
            0,
        );
        let row = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::B,
            3,
            "single-elimination",
            0,
        );
        assert_eq!(row.games_resolved, 3);
        assert_eq!(row.tiebreak_games, 1);
        assert_eq!(row.series_winner.as_deref(), Some("m1"));
        assert!(!row.awaiting_tiebreak);
        assert!(!row.seeded_fallback);
        assert!(row.series_complete());
    }

    #[test]
    fn live_three_drawn_tiebreaks_trigger_seeded_fallback() {
        let seed = 11u64;
        let mut pairs = Vec::new();
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::Draw,
            1,
            "single-elimination",
            seed,
        );
        assert!(pairs[0].awaiting_tiebreak);
        for _ in 0..tournament::MAX_TIEBREAKS {
            apply_live_game(
                &mut pairs,
                "m0",
                "m1",
                JudgeDecision::Draw,
                1,
                "single-elimination",
                seed,
            );
        }
        let row = &pairs[0];
        let expected =
            tournament::seeded_fallback_winner(seed, &ModelId::new("m0"), &ModelId::new("m1"));
        assert_eq!(row.games_resolved, 1 + tournament::MAX_TIEBREAKS);
        assert_eq!(row.tiebreak_games, tournament::MAX_TIEBREAKS);
        assert!(row.seeded_fallback);
        assert!(!row.awaiting_tiebreak);
        assert_eq!(row.series_winner, Some(expected.to_string()));
        assert!(row.series_complete());
    }

    #[test]
    fn live_round_robin_series_draw_does_not_enter_tiebreaks() {
        let mut pairs = Vec::new();
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::Draw,
            3,
            "round-robin",
            0,
        );
        let row = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::Draw,
            3,
            "round-robin",
            0,
        );
        assert_eq!(row.games_resolved, 2);
        assert!(row.series_draw);
        assert!(!row.awaiting_tiebreak);
        assert!(row.series_winner.is_none());
        assert!(!row.seeded_fallback);
        assert!(row.series_complete());

        let late = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "round-robin",
            0,
        );
        assert_eq!(late.games_resolved, 2);
        assert!(late.series_draw);
        assert!(late.series_winner.is_none());
    }

    #[test]
    fn live_king_of_the_hill_advances_using_the_series_winner() {
        let mut pairs = Vec::new();
        apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "king-of-the-hill",
            0,
        );
        let first = apply_live_game(
            &mut pairs,
            "m0",
            "m1",
            JudgeDecision::A,
            3,
            "king-of-the-hill",
            0,
        );
        assert_eq!(first.series_winner.as_deref(), Some("m0"));

        apply_live_game(
            &mut pairs,
            "m0",
            "m2",
            JudgeDecision::B,
            3,
            "king-of-the-hill",
            0,
        );
        let second = apply_live_game(
            &mut pairs,
            "m0",
            "m2",
            JudgeDecision::B,
            3,
            "king-of-the-hill",
            0,
        );
        assert_eq!(second.series_winner.as_deref(), Some("m2"));
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].series_winner.as_deref(), Some("m0"));
        assert_eq!(pairs[1].series_winner.as_deref(), Some("m2"));
    }

    #[tokio::test]
    async fn port_zero_advertises_the_bound_address() {
        let listener = tokio::net::TcpListener::bind(std::net::SocketAddr::from((BIND_HOST, 0)))
            .await
            .expect("bind");
        let url = listen_url(&listener).expect("local addr");
        let addr = listener.local_addr().expect("local addr");
        assert_eq!(url, format!("http://{addr}"));
        assert_ne!(addr.port(), 0);
        assert!(addr.ip().is_loopback());
    }
}
