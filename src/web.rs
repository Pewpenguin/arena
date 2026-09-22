use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};

use crate::error::{Error, Result};
use crate::exec::{self, ExecConfig};
use crate::persist;
use crate::provider::{ModelId, OpenAICompatibleProvider};
use crate::task::{self, Task};

const BIND_HOST: [u8; 4] = [127, 0, 0, 1];

struct AppState {
    session: Mutex<Option<ProviderSession>>,
    running: Mutex<bool>,
}

struct ProviderSession {
    api_key: String,
    base_url: String,
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
    started: bool,
}

pub async fn serve(port: u16) -> Result<()> {
    let addr = std::net::SocketAddr::from((BIND_HOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Arena web UI: http://127.0.0.1:{port}");
    axum::serve(listener, router()).await?;
    Ok(())
}

fn router() -> Router {
    let state = Arc::new(AppState {
        session: Mutex::new(None),
        running: Mutex::new(false),
    });
    Router::new()
        .route("/", get(page))
        .route("/api/models", post(load_models))
        .route("/api/runs", post(start_run))
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

    {
        let mut running = state.running.lock().await;
        if *running {
            return json_error(StatusCode::CONFLICT, "an experiment is already running");
        }
        *running = true;
    }

    let provider = OpenAICompatibleProvider::new(api_key, base_url);
    let running = state.clone();
    tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let result = exec::collect_exec_with_events(&provider, &config, tx).await;
        let _ = drain.await;
        *running.running.lock().await = false;
        if let Err(error) = result {
            eprintln!("experiment failed: {error}");
        }
    });

    (StatusCode::ACCEPTED, Json(StartResponse { started: true })).into_response()
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
</style>
</head>
<body>
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

<script>
const models = [];
const selected = new Set();

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
  status("run_status", "Tournament started", true);
});

render();
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::ExperimentEvent;
    use crate::exec::collect_exec_with_events;
    use crate::provider::{CompletionRequest, CompletionResponse, ModelProvider, ProviderError};

    fn task() -> Task {
        Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
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
}
