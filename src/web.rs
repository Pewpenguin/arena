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
use crate::judge::JudgeDecision;
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
    outcome: PairOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    agreement: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PairOutcome {
    Waiting,
    A,
    B,
    Draw,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
struct RunSnapshot {
    seq: u64,
    run_id: String,
    status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
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
        task_id: String,
        model_a: String,
        model_b: String,
        winner: JudgeDecision,
        agreement: bool,
    },
    PairFailed {
        seq: u64,
        task_id: String,
        model_a: String,
        model_b: String,
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
    fn new(id: String, candidate_total: usize, expected_pairs: usize, pairs: Vec<PairRow>) -> Self {
        let (events, _) = broadcast::channel(64);
        Self {
            id,
            finished: AtomicBool::new(false),
            inner: Mutex::new(RunState {
                seq: 0,
                status: RunStatus::Running,
                error: None,
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
                let outcome = match judgment.winner {
                    JudgeDecision::A => PairOutcome::A,
                    JudgeDecision::B => PairOutcome::B,
                    JudgeDecision::Draw => PairOutcome::Draw,
                };
                upsert_pair(
                    &mut self.pairs,
                    &task_id,
                    &model_a,
                    &model_b,
                    outcome,
                    Some(judgment.agreement),
                );
                ClientEvent::PairResolved {
                    seq: self.seq,
                    task_id,
                    model_a: model_a.to_string(),
                    model_b: model_b.to_string(),
                    winner: judgment.winner,
                    agreement: judgment.agreement,
                }
            }
            ExperimentEvent::PairFailed {
                task_id,
                model_a,
                model_b,
                ..
            } => {
                self.failed_pairs += 1;
                upsert_pair(
                    &mut self.pairs,
                    &task_id,
                    &model_a,
                    &model_b,
                    PairOutcome::Failed,
                    None,
                );
                ClientEvent::PairFailed {
                    seq: self.seq,
                    task_id,
                    model_a: model_a.to_string(),
                    model_b: model_b.to_string(),
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
}

fn upsert_pair(
    pairs: &mut Vec<PairRow>,
    task_id: &str,
    model_a: &ModelId,
    model_b: &ModelId,
    outcome: PairOutcome,
    agreement: Option<bool>,
) {
    let a = model_a.to_string();
    let b = model_b.to_string();
    if let Some(row) = pairs
        .iter_mut()
        .find(|row| row.task_id == task_id && row.model_a == a && row.model_b == b)
    {
        row.outcome = outcome;
        row.agreement = agreement;
        return;
    }
    pairs.push(PairRow {
        task_id: task_id.to_string(),
        model_a: a,
        model_b: b,
        outcome,
        agreement,
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
                    outcome: PairOutcome::Waiting,
                    agreement: None,
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
    let candidate_total = config.tasks.len().saturating_mul(config.models.len());
    let expected_pairs = if config.judge.is_some() {
        exec::expected_pairs(config.tasks.len(), config.models.len())
    } else {
        0
    };
    let pairs = if config.judge.is_some() {
        waiting_pairs(&config.tasks, &config.models)
    } else {
        Vec::new()
    };

    let live = {
        let mut slot = state.run.lock().await;
        if slot.as_ref().is_some_and(|run| run.is_running()) {
            return Err(StartRunError::Busy);
        }
        let id = state.alloc_id();
        let live = Arc::new(LiveRun::new(
            id.clone(),
            candidate_total,
            expected_pairs,
            pairs,
        ));
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
.badge { font-size: .8rem; letter-spacing: .08em; font-weight: 700; }
.bar { background: #efeae1; height: .55rem; margin: .35rem 0 .2rem; }
.bar > span { display: block; height: 100%; background: #2b4b34; width: 0; }
.pair { display: grid; grid-template-columns: 1.2rem 1fr auto; gap: .5rem; padding: .4rem 0; border-bottom: 1px solid #ece6da; }
.pair .mark { color: #5e5b54; }
.pair.ok .mark { color: #2b4b34; }
.pair.fail .mark { color: #8b2d2d; }
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
    <span id="dash_status" class="badge">RUNNING</span>
  </div>
  <p id="dash_error" class="status error"></p>
  <h2>Candidates</h2>
  <div class="bar"><span id="cand_bar"></span></div>
  <p id="cand_label" class="meta">0 / 0</p>
  <h2>Tournament</h2>
  <div class="bar"><span id="pair_bar"></span></div>
  <p id="pair_label" class="meta">0 / 0 pairs</p>
  <h2>Pairs</h2>
  <div id="pair_list"></div>
</div>

<script>
const models = [];
const selected = new Set();
let view = null;
let seenSeq = 0;

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

function outcomeText(row) {
  if (row.outcome === "a") return row.model_a + " wins";
  if (row.outcome === "b") return row.model_b + " wins";
  if (row.outcome === "draw") return "Draw";
  if (row.outcome === "failed") return "Failed";
  return "Waiting";
}

function pairMark(outcome) {
  if (outcome === "waiting") return "⟳";
  if (outcome === "failed") return "✗";
  return "✓";
}

function upsertPair(taskId, modelA, modelB, outcome, agreement) {
  const row = view.pairs.find((item) =>
    item.task_id === taskId && item.model_a === modelA && item.model_b === modelB
  );
  if (row) {
    row.outcome = outcome;
    row.agreement = agreement;
    return;
  }
  view.pairs.push({
    task_id: taskId,
    model_a: modelA,
    model_b: modelB,
    outcome,
    agreement,
  });
}

function renderDash() {
  if (!view) return;
  const statusEl = document.getElementById("dash_status");
  statusEl.textContent = view.status === "complete"
    ? "COMPLETE"
    : view.status === "failed"
      ? "FAILED"
      : "RUNNING";
  document.getElementById("dash_error").textContent = view.error || "";
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
    const div = document.createElement("div");
    div.className = "pair" + (row.outcome === "failed" ? " fail" : row.outcome === "waiting" ? "" : " ok");
    const mark = document.createElement("span");
    mark.className = "mark";
    mark.textContent = pairMark(row.outcome);
    const vs = document.createElement("span");
    vs.textContent = row.model_a + "  vs  " + row.model_b + "  ·  " + row.task_id;
    const out = document.createElement("span");
    out.textContent = outcomeText(row);
    div.append(mark, vs, out);
    list.append(div);
  });
}

function applyEvent(msg) {
  if (msg.type !== "snapshot" && msg.seq && msg.seq <= seenSeq) return;
  if (msg.seq) seenSeq = msg.seq;
  switch (msg.type) {
    case "snapshot":
      view = msg;
      seenSeq = msg.seq || 0;
      break;
    case "candidate_finished":
      view.candidate_completed += 1;
      break;
    case "pair_resolved":
      view.resolved_pairs += 1;
      upsertPair(msg.task_id, msg.model_a, msg.model_b, msg.winner, msg.agreement);
      break;
    case "pair_failed":
      view.failed_pairs += 1;
      upsertPair(msg.task_id, msg.model_a, msg.model_b, "failed", null);
      break;
    case "run_complete":
      view.status = "complete";
      view.expected_pairs = msg.expected_pairs;
      view.resolved_pairs = msg.resolved_pairs;
      view.failed_pairs = msg.failed_pairs;
      break;
    case "run_failed":
      view.status = "failed";
      view.error = msg.error;
      break;
    default:
      return;
  }
  renderDash();
}

function showDashboard(runId) {
  document.getElementById("config").hidden = true;
  document.getElementById("dashboard").hidden = false;
  view = {
    run_id: runId,
    status: "running",
    seq: 0,
    candidate_completed: 0,
    candidate_total: 0,
    expected_pairs: 0,
    resolved_pairs: 0,
    failed_pairs: 0,
    pairs: [],
    error: null,
  };
  seenSeq = 0;
  renderDash();
  const source = new EventSource("/api/runs/" + encodeURIComponent(runId) + "/events");
  source.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    applyEvent(msg);
    if (msg.type === "run_complete" || msg.type === "run_failed" ||
        (msg.type === "snapshot" && msg.status && msg.status !== "running")) {
      source.close();
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
    use crate::judge::{Judgment, JudgmentFailure, JudgmentFailureKind, OrientationFailure};
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
        let live = Arc::new(LiveRun::new(
            "1".into(),
            2,
            1,
            waiting_pairs(&[task()], &[ModelId::new("m0"), ModelId::new("m1")]),
        ));
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
                model_a,
                model_b,
                winner,
                agreement,
                ..
            } => {
                assert_eq!(model_a, "m0");
                assert_eq!(model_b, "m1");
                assert_eq!(winner, JudgeDecision::A);
                assert!(agreement);
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
        let live = Arc::new(LiveRun::new(
            "9".into(),
            2,
            1,
            waiting_pairs(&[task()], &[ModelId::new("m0"), ModelId::new("m1")]),
        ));
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
            failure: JudgmentFailure {
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
            },
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
                assert_eq!(snapshot.pairs[0].outcome, PairOutcome::Failed);
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
                assert_eq!(snapshot.pairs[0].outcome, PairOutcome::Draw);
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
        let live = Arc::new(LiveRun::new("2".into(), 1, 0, Vec::new()));
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
    }
}
