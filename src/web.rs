use std::convert::Infallible;
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
use crate::provider::{ModelId, ModelProvider, OpenAICompatibleProvider};
use crate::task::{self, Task};

const BIND_HOST: [u8; 4] = [127, 0, 0, 1];

struct AppState {
    session: Mutex<Option<ProviderSession>>,
    next_run: AtomicU64,
    run: Mutex<Option<Arc<LiveRun>>>,
}

struct ProviderSession {
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RunStatus {
    Running,
    Complete,
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

#[derive(Debug, Deserialize)]
struct ConnectRequest {
    base_url: String,
    api_key: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StartRequest {
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
                self.status = RunStatus::Complete;
                self.expected_pairs = expected_pairs;
                self.resolved_pairs = resolved_pairs;
                self.failed_pairs = failed_pairs;
                ClientEvent::RunComplete {
                    seq: self.seq,
                    expected_pairs,
                    resolved_pairs,
                    failed_pairs,
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
    let base_url = request.base_url.trim().to_string();
    let api_key = request.api_key.trim().to_string();
    if api_key.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "API key is required");
    }
    if base_url.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "base URL is required");
    }

    let provider = OpenAICompatibleProvider::new(api_key.clone(), base_url.clone());
    match provider.list_models().await {
        Ok(models) => {
            *state.session.lock().await = Some(ProviderSession { api_key, base_url });
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
    let api_key = if request.api_key.trim().is_empty() {
        session.as_ref().map(|item| item.api_key.clone())
    } else {
        Some(request.api_key.trim().to_string())
    };
    let Some(api_key) = api_key.filter(|key| !key.is_empty()) else {
        return json_error(StatusCode::BAD_REQUEST, "API key is required");
    };
    let base_url = if request.base_url.trim().is_empty() {
        session
            .as_ref()
            .map(|item| item.base_url.clone())
            .unwrap_or_default()
    } else {
        request.base_url.trim().to_string()
    };
    drop(session);
    if base_url.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "base URL is required");
    }

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
        base_url.clone(),
        request.seed,
    ) {
        Ok(config) => config,
        Err(error) => return json_error(StatusCode::BAD_REQUEST, error.to_string()),
    };

    let provider = OpenAICompatibleProvider::new(api_key, base_url);
    match start_experiment(state, config, provider).await {
        Ok(run_id) => (StatusCode::ACCEPTED, Json(StartResponse { run_id })).into_response(),
        Err(StartRunError::Busy) => {
            json_error(StatusCode::CONFLICT, "an experiment is already running")
        }
    }
}

async fn start_experiment<P>(
    state: Arc<AppState>,
    config: ExecConfig,
    provider: P,
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
    tokio::spawn(async move {
        let (tx, rx) = mpsc::unbounded_channel();
        let pump = tokio::spawn(pump_events(live.clone(), rx));
        let result = exec::collect_exec_with_events(&provider, &config, tx).await;
        let _ = pump.await;
        if let Err(error) = result {
            live.fail(error.to_string()).await;
        }
    });

    Ok(run_id)
}

async fn pump_events(live: Arc<LiveRun>, mut rx: mpsc::UnboundedReceiver<ExperimentEvent>) {
    while let Some(event) = rx.recv().await {
        live.apply(event).await;
    }
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
body { font: 16px/1.45 system-ui, sans-serif; max-width: 48rem; margin: 2rem auto; padding: 0 1rem; color: #1d1c19; }
h1 { font-size: 1.5rem; margin: 0 0 .3rem; }
h2 { font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
.meta { color: #5e5b54; margin: 0 0 1.25rem; }
label, .label { display: block; margin: .85rem 0 .3rem; font-weight: 600; }
input[type=text], input[type=password], input[type=number], textarea, select {
  width: 100%; box-sizing: border-box; padding: .4rem .5rem; font: inherit;
}
textarea { min-height: 10rem; font-family: ui-monospace, monospace; }
button { font: inherit; padding: .4rem .8rem; margin: .4rem .4rem 0 0; cursor: pointer; }
.row { display: flex; gap: .5rem; align-items: end; }
.row > * { flex: 1; }
.status { margin: .75rem 0; min-height: 1.3em; }
.status.error { color: #8b2d2d; }
.status.ok { color: #2b4b34; }
.models { border: 1px solid #d9d3c7; max-height: 16rem; overflow: auto; padding: .4rem .6rem; }
.models label { font-weight: 400; margin: .2rem 0; }
.dash-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.head-right { display: flex; align-items: baseline; gap: .75rem; }
.badge { font-size: .8rem; letter-spacing: .08em; font-weight: 700; }
.badge.running { color: #5e4a16; }
.badge.complete { color: #2b4b34; }
.badge.failed { color: #8b2d2d; }
.elapsed { color: #5e5b54; font-variant-numeric: tabular-nums; }
.overview { display: grid; grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr)); gap: .75rem 1rem; margin: 1rem 0 1.25rem; }
.overview div { margin: 0; }
.overview dt { font-size: .8rem; color: #5e5b54; }
.overview dd { margin: .15rem 0 0; font-weight: 600; }
section { margin: 1.25rem 0; }
.bar { background: #efeae1; height: .55rem; margin: .35rem 0 .2rem; }
.bar > span { display: block; height: 100%; background: #2b4b34; width: 0; }
.pair { border-bottom: 1px solid #ece6da; }
.pair > summary { display: grid; grid-template-columns: 5.5rem 1fr auto; gap: .5rem; align-items: baseline; padding: .5rem 0; cursor: pointer; list-style: none; }
.pair > summary::-webkit-details-marker { display: none; }
.pair .state { font-size: .75rem; font-weight: 700; letter-spacing: .04em; color: #5e5b54; }
.pair.resolved .state { color: #2b4b34; }
.pair.failed .state { color: #8b2d2d; }
.pair .detail { font-size: .9rem; padding: 0 0 .75rem 0; color: #3f3d39; }
.pair .detail dl { display: grid; grid-template-columns: 11rem 1fr; gap: .2rem .75rem; margin: .4rem 0; }
.pair .detail dt { color: #5e5b54; }
.pair .detail dd { margin: 0; }
.pair .detail pre { margin: .25rem 0 .6rem; padding: .45rem .55rem; background: #f6f3ee; white-space: pre-wrap; font: .85rem/1.35 ui-monospace, monospace; }
</style>
</head>
<body>
<div id="config">
<h1>Arena</h1>
<p class="meta">Configure an OpenAI-compatible provider and start a tournament.</p>

<label for="base_url">Provider base URL</label>
<input id="base_url" type="text" value="https://api.openai.com/v1" autocomplete="off">

<label for="api_key">API key</label>
<input id="api_key" type="password" autocomplete="off">

<button id="load" type="button">Load Models</button>
<p id="status" class="status"></p>

<label for="filter">Discovered models</label>
<input id="filter" type="text" placeholder="Filter models" autocomplete="off">
<div id="model_list" class="models"></div>

<div class="row">
  <div>
    <label for="manual">Add model ID</label>
    <input id="manual" type="text" placeholder="provider/model-id" autocomplete="off">
  </div>
  <div>
    <span class="label">&nbsp;</span>
    <button id="add" type="button">Add</button>
  </div>
</div>

<h2>Candidate models</h2>
<p class="meta">Select one or more candidates. IDs must be unique.</p>
<div id="candidates" class="models"></div>

<h2>Judge model</h2>
<p class="meta">Optional. Must not be a candidate.</p>
<select id="judge">
  <option value="">No judge</option>
</select>

<label for="tasks">Tasks JSON</label>
<textarea id="tasks">[
  {
    "id": "t1",
    "prompt": "Explain Rust ownership in two sentences."
  }
]</textarea>

<label for="seed">Bootstrap seed</label>
<input id="seed" type="number" value="0" min="0" step="1">

<button id="start" type="button">Start Tournament</button>
<p id="run_status" class="status"></p>
</div>

<div id="dashboard" hidden>
  <div class="dash-head">
    <h1>Arena</h1>
    <div class="head-right">
      <span id="dash_elapsed" class="elapsed">0:00</span>
      <span id="dash_status" class="badge running">Running</span>
    </div>
  </div>
  <p id="dash_error" class="status error"></p>
  <dl class="overview">
    <div><dt>Candidates</dt><dd id="ov_candidates">0</dd></div>
    <div><dt>Tasks</dt><dd id="ov_tasks">0</dd></div>
    <div><dt>Judge</dt><dd id="ov_judge">—</dd></div>
    <div><dt>Resolved</dt><dd id="ov_resolved">0</dd></div>
    <div><dt>Failed</dt><dd id="ov_failed">0</dd></div>
  </dl>
  <section>
    <h2>Candidates</h2>
    <div class="bar"><span id="cand_bar"></span></div>
    <p id="cand_label" class="meta">0 / 0</p>
  </section>
  <section>
    <h2>Pairs</h2>
    <div class="bar"><span id="pair_bar"></span></div>
    <p id="pair_label" class="meta">0 / 0 pairs</p>
    <div id="pair_list"></div>
  </section>
</div>

<script>
const models = [];
const selected = new Set();
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

function render() {
  const query = document.getElementById("filter").value.trim().toLowerCase();
  const list = document.getElementById("model_list");
  list.replaceChildren();
  models.filter((id) => !query || id.toLowerCase().includes(query)).forEach((id) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = selected.has(id);
    box.addEventListener("change", () => {
      if (box.checked) selected.add(id); else selected.delete(id);
      if (document.getElementById("judge").value === id) {
        document.getElementById("judge").value = "";
      }
      render();
    });
    label.append(box, " " + id);
    list.append(label);
  });

  const candidates = document.getElementById("candidates");
  candidates.replaceChildren();
  [...selected].forEach((id) => {
    const div = document.createElement("div");
    div.textContent = id;
    candidates.append(div);
  });

  const judge = document.getElementById("judge");
  const current = judge.value;
  judge.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "No judge";
  judge.append(none);
  models.filter((id) => !selected.has(id)).forEach((id) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    judge.append(option);
  });
  judge.value = selected.has(current) ? "" : current;
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

function appendField(dl, label, value) {
  if (value == null || value === "") return;
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
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
  appendField(dl, "Task", row.task_id);
  appendField(dl, "Model A", row.model_a);
  appendField(dl, "Model B", row.model_b);
  if (row.judgment) {
    const j = row.judgment;
    appendField(dl, "Final winner", decisionText(j.winner, j.model_a, j.model_b));
    appendField(dl, "Agreement", j.agreement ? "agree" : "disagree");
    appendField(dl, "Duration", j.duration_ms + " ms");
    appendField(dl, "Judge", j.judge_model);
    appendField(dl, "AB winner (mapped A/B frame)", j.orientation_ab == null ? null : decisionText(j.orientation_ab, j.model_a, j.model_b));
    appendField(dl, "BA winner (mapped A/B frame)", j.orientation_ba == null ? null : decisionText(j.orientation_ba, j.model_a, j.model_b));
    appendField(dl, "Reason AB", j.reason_ab);
    appendField(dl, "Reason BA", j.reason_ba);
    appendPre(dl, "Raw AB (prompt frame)", j.raw_ab);
    appendPre(dl, "Raw BA (prompt frame)", j.raw_ba);
  }
  if (row.failure) {
    appendField(dl, "Judge", row.failure.judge_model);
    (row.failure.orientations || []).forEach((item) => {
      appendField(
        dl,
        item.orientation.toUpperCase() + " " + item.kind,
        item.error + " (attempts: " + item.attempts + ")"
      );
    });
  }
  wrap.append(dl);
  return wrap;
}

function renderDash() {
  if (!view) return;
  const statusEl = document.getElementById("dash_status");
  statusEl.textContent = statusLabel(view.status);
  statusEl.className = "badge " + (view.status === "complete" ? "complete" : view.status === "failed" ? "failed" : "running");
  document.getElementById("dash_error").textContent = view.error || "";
  document.getElementById("ov_candidates").textContent = String(view.candidate_count || 0);
  document.getElementById("ov_tasks").textContent = String(view.task_count || 0);
  document.getElementById("ov_judge").textContent = view.judge || "None";
  document.getElementById("ov_resolved").textContent = String(view.resolved_pairs);
  document.getElementById("ov_failed").textContent = String(view.failed_pairs);
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
    const details = document.createElement("details");
    details.className = "pair " + row.status;
    const key = pairKey(row);
    details.open = openPairs.has(key);
    details.addEventListener("toggle", () => {
      if (details.open) openPairs.add(key); else openPairs.delete(key);
    });
    const summary = document.createElement("summary");
    const state = document.createElement("span");
    state.className = "state";
    state.textContent = row.status === "resolved" ? "Resolved" : row.status === "failed" ? "Failed" : row.status === "judging" ? "Judging" : "Waiting";
    const vs = document.createElement("span");
    vs.textContent = row.model_a + " vs " + row.model_b + " · " + row.task_id;
    const out = document.createElement("span");
    out.textContent = compactOutcome(row);
    summary.append(state, vs, out);
    details.append(summary);
    if (row.status === "resolved" || row.status === "failed") {
      details.append(pairDetail(row));
    }
    list.append(details);
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
      view.status = "complete";
      view.expected_pairs = msg.expected_pairs;
      view.resolved_pairs = msg.resolved_pairs;
      view.failed_pairs = msg.failed_pairs;
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

document.getElementById("filter").addEventListener("input", render);
document.getElementById("add").addEventListener("click", () => {
  addModel(document.getElementById("manual").value);
  document.getElementById("manual").value = "";
});

document.getElementById("load").addEventListener("click", async () => {
  status("status", "", true);
  const response = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_url: baseUrl(), api_key: apiKey() }),
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
      base_url: baseUrl(),
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

render();
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
        let run_id = start_experiment(state.clone(), config, OkProvider)
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
        start_experiment(state.clone(), config, HangProvider)
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
            start_experiment(state, config, HangProvider).await,
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
        start_experiment(state.clone(), config, OkProvider)
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
        start_experiment(state.clone(), config, FailProvider)
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
        start_experiment(state, config, OkProvider).await.unwrap();
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
}
