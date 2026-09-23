use std::convert::Infallible;
use std::path::{Path as FilePath, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
use crate::judge::{Judgment, JudgmentFailure};
use crate::persist;
use crate::provider::{
    AnthropicProvider, GeminiProvider, ModelId, ModelProvider, OpenAICompatibleProvider,
};
use crate::task::{self, Task};

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
    expected_pairs: usize,
    resolved_pairs: usize,
    failed_pairs: usize,
    pairs: Vec<PairRow>,
    output_path: Option<String>,
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
    expected_pairs: usize,
    resolved_pairs: usize,
    failed_pairs: usize,
    pairs: Vec<PairRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_path: Option<String>,
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
    },
    PairFailed {
        seq: u64,
        failure: JudgmentFailure,
    },
    RunComplete {
        seq: u64,
        expected_pairs: usize,
        resolved_pairs: usize,
        failed_pairs: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_path: Option<String>,
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
    seed: u64,
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

    fn alloc_id(&self) -> String {
        self.next_run.fetch_add(1, Ordering::Relaxed).to_string()
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
        let expected_pairs = if config.judge.is_some() {
            exec::expected_pairs(task_count, candidate_count)
        } else {
            0
        };
        let pairs = if config.judge.is_some() {
            waiting_pairs(&config.tasks, &config.models)
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
                expected_pairs,
                resolved_pairs: 0,
                failed_pairs: 0,
                pairs,
                output_path: None,
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
                expected_pairs: self.expected_pairs,
                resolved_pairs: self.resolved_pairs,
                failed_pairs: self.failed_pairs,
                pairs: self.pairs.clone(),
                output_path: self.output_path.clone(),
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
                upsert_pair(
                    &mut self.pairs,
                    &task_id,
                    &model_a,
                    &model_b,
                    PairStatus::Resolved,
                    Some(judgment.clone()),
                    None,
                );
                ClientEvent::PairResolved {
                    seq: self.seq,
                    judgment,
                }
            }
            ExperimentEvent::PairFailed {
                task_id,
                model_a,
                model_b,
                failure,
            } => {
                self.failed_pairs += 1;
                upsert_pair(
                    &mut self.pairs,
                    &task_id,
                    &model_a,
                    &model_b,
                    PairStatus::Failed,
                    None,
                    Some(failure.clone()),
                );
                ClientEvent::PairFailed {
                    seq: self.seq,
                    failure,
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
                    expected_pairs,
                    resolved_pairs,
                    failed_pairs,
                    output_path: self.output_path.clone(),
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

fn upsert_pair(
    pairs: &mut Vec<PairRow>,
    task_id: &str,
    model_a: &ModelId,
    model_b: &ModelId,
    status: PairStatus,
    judgment: Option<Judgment>,
    failure: Option<JudgmentFailure>,
) {
    let a = model_a.to_string();
    let b = model_b.to_string();
    if let Some(row) = pairs
        .iter_mut()
        .find(|row| same_unordered_pair(row, task_id, &a, &b))
    {
        row.status = status;
        row.judgment = judgment;
        row.failure = failure;
        return;
    }
    pairs.push(PairRow {
        task_id: task_id.to_string(),
        model_a: a,
        model_b: b,
        status,
        judgment,
        failure,
    });
}

fn waiting_pairs(tasks: &[Task], models: &[ModelId]) -> Vec<PairRow> {
    let mut pairs = Vec::new();
    for task in tasks {
        for i in 0..models.len() {
            for j in (i + 1)..models.len() {
                pairs.push(PairRow {
                    task_id: task.id.clone(),
                    model_a: models[i].to_string(),
                    model_b: models[j].to_string(),
                    status: PairStatus::Waiting,
                    judgment: None,
                    failure: None,
                });
            }
        }
    }
    pairs
}

pub async fn serve(port: u16) -> Result<()> {
    let addr = std::net::SocketAddr::from((BIND_HOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Arena web UI: http://127.0.0.1:{port}");
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
        .route("/api/runs", post(start_run))
        .route("/api/runs/{run_id}/events", get(run_events))
        .with_state(state)
}

async fn page() -> Html<&'static str> {
    Html(PAGE)
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

    let config = match build_exec_config(
        request.models,
        request.judge,
        tasks,
        resolved.base_url.clone(),
        request.seed,
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

fn write_experiment(path: &FilePath, output: &persist::Output) -> std::result::Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    persist::write(path, output).map_err(|error| error.to_string())
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
        let id = state.alloc_id();
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

pub(crate) fn build_exec_config(
    models: Vec<String>,
    judge: Option<String>,
    tasks: Vec<Task>,
    base_url: String,
    seed: u64,
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
    Ok(ExecConfig {
        tasks,
        models,
        judge,
        seed,
        tasks_path: None,
        started_at: persist::utc_timestamp(),
        base_url,
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
.region h2, label.region-title, .section-line h2, .experiment > h2 {
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
#seed { max-width: 8rem; font-family: var(--mono); font-size: .9rem; }
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
  gap: 12px 32px;
  margin: 0 0 24px;
}
.facts div { min-width: 0; }
.facts dt {
  margin: 0;
  color: var(--muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.facts dd {
  margin: 4px 0 0;
  color: var(--ink);
  font-family: var(--mono);
  font-size: .92rem;
  font-variant-numeric: tabular-nums;
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
.launch-action { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.launch-action .status { margin: 0; text-align: right; }
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
  .facts { gap: 12px 16px; }
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
    <div>
      <label for="seed">Bootstrap seed</label>
      <input id="seed" type="number" value="0" min="0" step="1">
    </div>
    <div class="launch-action">
      <button id="start" class="primary" type="button">Run Tournament →</button>
      <p id="run_status" class="status"></p>
    </div>
  </div>
</section>
</div>
</div>

<div id="dashboard" hidden>
  <header class="mast">
    <div class="brand">
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
      <dt>Resolved</dt>
      <dd id="ov_resolved" class="num">0</dd>
    </div>
    <div>
      <dt>Failed</dt>
      <dd id="ov_failed" class="num">0</dd>
    </div>
  </dl>
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
        <h2>Pairs</h2>
        <p id="pair_label" class="meta">0 / 0 pairs</p>
      </div>
      <div class="bar"><span id="pair_bar"></span></div>
    </div>
  </div>
  <h2 class="results-title">Pairwise evaluation</h2>
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
          <th>Outcome</th>
        </tr>
      </thead>
      <tbody id="pair_list"></tbody>
    </table>
  </div>
  </div>
</div>
</div>

<script>
const models = [];
const selected = new Set();
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
}

function applyProvider(next) {
  provider = next;
  models.length = 0;
  selected.clear();
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
  if (decision === "draw") return "Draw";
  return decision || "—";
}

function compactOutcome(row) {
  if (row.status === "resolved" && row.judgment) {
    return decisionText(row.judgment.winner, row.judgment.model_a, row.judgment.model_b);
  }
  if (row.status === "failed") {
    const first = row.failure && row.failure.orientations && row.failure.orientations[0];
    if (first && first.error) return first.kind + ": " + first.error;
    return "Failed";
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

function upsertPair(taskId, modelA, modelB, status, judgment, failure) {
  const row = findPair(taskId, modelA, modelB);
  if (row) {
    row.status = status;
    row.judgment = judgment || null;
    row.failure = failure || null;
    return;
  }
  view.pairs.push({
    task_id: taskId,
    model_a: modelA,
    model_b: modelB,
    status,
    judgment: judgment || null,
    failure: failure || null,
  });
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
  if (row.judgment) {
    const j = row.judgment;
    appendField(dl, "Final winner", decisionText(j.winner, j.model_a, j.model_b), true);
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
  }
  wrap.append(dl);
  return wrap;
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
  document.getElementById("ov_candidates").textContent = String(view.candidate_count || 0);
  document.getElementById("ov_tasks").textContent = String(view.task_count || 0);
  document.getElementById("ov_judge").textContent = view.judge || "None";
  const resolved = document.getElementById("ov_resolved");
  resolved.textContent = String(view.resolved_pairs);
  resolved.className = "num" + (view.resolved_pairs > 0 ? " hot" : "");
  const failed = document.getElementById("ov_failed");
  failed.textContent = String(view.failed_pairs);
  failed.className = "num" + (view.failed_pairs > 0 ? " hot" : "");
  renderElapsed();
  const candDone = view.candidate_completed;
  const candTotal = view.candidate_total;
  document.getElementById("cand_bar").style.width = candTotal ? (100 * candDone / candTotal) + "%" : "0%";
  document.getElementById("cand_label").textContent = candDone + " / " + candTotal;
  const pairDone = view.resolved_pairs + view.failed_pairs;
  const pairTotal = view.expected_pairs;
  document.getElementById("pair_bar").style.width = pairTotal ? (100 * pairDone / pairTotal) + "%" : "0%";
  document.getElementById("pair_label").textContent = pairDone + " / " + pairTotal + " pairs";
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
      upsertPair(j.task_id, j.model_a, j.model_b, "resolved", j, null);
      break;
    }
    case "pair_failed": {
      const f = msg.failure;
      view.failed_pairs += 1;
      upsertPair(f.task_id, f.model_a, f.model_b, "failed", null, f);
      break;
    }
    case "run_complete":
      view.status = msg.failed_pairs > 0 ? "incomplete" : "complete";
      view.expected_pairs = msg.expected_pairs;
      view.resolved_pairs = msg.resolved_pairs;
      view.failed_pairs = msg.failed_pairs;
      view.output_path = msg.output_path || null;
      view.finished_at = Date.now();
      stopElapsed();
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
    expected_pairs: 0,
    resolved_pairs: 0,
    failed_pairs: 0,
    pairs: [],
    error: null,
    output_path: null,
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

document.getElementById("start").addEventListener("click", async () => {
  status("run_status", "", true);
  let tasks;
  try {
    tasks = JSON.parse(document.getElementById("tasks").value);
  } catch (error) {
    status("run_status", "Tasks JSON is invalid", false);
    return;
  }
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider,
      base_url: provider === "compatible" ? baseUrl() : "",
      api_key: apiKey(),
      models: [...selected],
      judge: document.getElementById("judge").value || null,
      tasks,
      seed: Number(document.getElementById("seed").value || 0),
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    status("run_status", body.error || "Failed to start", false);
    return;
  }
  showDashboard(body.run_id);
});

applyProvider("openai");
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
        }
    }

    fn test_output_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("arena-web-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
        let live = state.run.lock().await.clone().unwrap();
        assert_eq!(live.id, run_id);
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
            ClientEvent::PairResolved { judgment, .. } => {
                assert_eq!(judgment.model_a, ModelId::new("m0"));
                assert_eq!(judgment.model_b, ModelId::new("m1"));
                assert_eq!(judgment.winner, JudgeDecision::A);
                assert!(judgment.agreement);
                assert_eq!(judgment.raw_ab.as_deref(), Some("raw-ab"));
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
                expected_pairs: 1,
                resolved_pairs: 0,
                failed_pairs: 0,
                pairs: Vec::new(),
                output_path: None,
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
        let pairs = waiting_pairs(&[task()], &models);
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
        let html = crate::html::render(&report);
        assert!(html.contains("m0"));
        assert!(html.contains("m1"));
        assert!(!html.contains(SECRET_API_KEY));
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
                seed: 0,
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
                seed: 0,
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
                seed: 0,
            }),
        )
        .await;
        assert_eq!(invalid_models.status(), StatusCode::BAD_REQUEST);
    }
}
