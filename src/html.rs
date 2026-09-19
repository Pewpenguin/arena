use crate::bootstrap::BootstrapUnavailable;
use crate::judge::{JudgeDecision, OrientationFailure};
use crate::rating::UnavailableReason;
use crate::report::{
    BootstrapReport, CandidateRow, FailedPairRow, ModelReport, PairRow, Report, ReportSummary,
    RunConfig,
};

pub fn render(report: &Report<'_>) -> String {
    let mut html = String::new();
    html.push_str("<!doctype html>\n");
    html.push_str("<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"utf-8\">\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    html.push_str("<title>Arena experiment report</title>\n");
    html.push_str("<style>\n");
    html.push_str(STYLE);
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<main>\n");
    push_header(
        &mut html,
        &report.summary,
        &report.config,
        report.bootstrap.as_ref(),
    );
    push_models(&mut html, &report.models, report.summary.judge_used);
    if report.summary.judge_used {
        push_pairs(&mut html, &report.pairs);
        if !report.failed_pairs.is_empty() {
            push_failures(&mut html, &report.failed_pairs);
        }
        if let Some(bootstrap) = &report.bootstrap {
            push_bootstrap(&mut html, bootstrap);
        }
    }
    push_results(&mut html, &report.results);
    if report.summary.judge_used {
        push_audit(&mut html, &report.pairs);
    }
    push_config(&mut html, &report.config);
    html.push_str("</main>\n</body>\n</html>\n");
    html
}

const STYLE: &str = r#"
:root {
  --bg: #f4f1ea;
  --fg: #1d1c19;
  --muted: #5e5b54;
  --card: #fffcf7;
  --line: #d9d3c7;
  --agree-bg: #e7efe8;
  --agree-fg: #2b4b34;
  --disagree-bg: #f3eadc;
  --disagree-fg: #5c4520;
  --fail-bg: #f7eceb;
  --fail-line: #b56a64;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--fg);
  background: var(--bg);
  font: 16px/1.45 "Segoe UI", system-ui, sans-serif;
}
main {
  max-width: 72rem;
  margin: 0 auto;
  padding: 1.5rem 1.25rem 3rem;
}
h1, h2, h3 { font-weight: 650; letter-spacing: -0.02em; }
h1 { font-size: 1.7rem; margin: 0 0 .35rem; }
h2 { font-size: 1.15rem; margin: 0 0 .75rem; }
h3 { font-size: 1rem; margin: 0; }
.eyebrow { margin: 0; color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; }
.meta { margin: .15rem 0 .6rem; color: var(--muted); }
.status {
  display: inline-block;
  margin: 0 0 1rem;
  padding: .15rem .55rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  font-size: .78rem;
  letter-spacing: .04em;
}
.metrics { display: flex; flex-wrap: wrap; gap: .75rem; margin: 0 0 1.5rem; }
.metric {
  flex: 1 1 10rem;
  min-width: 9rem;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: .4rem;
  padding: .7rem .85rem;
}
.metric .label { display: block; color: var(--muted); font-size: .78rem; }
.metric .value { display: block; font-size: 1.15rem; font-variant-numeric: tabular-nums; }
section { margin: 0 0 1.75rem; }
.note { margin: 0 0 .7rem; color: var(--muted); font-size: .88rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border: 1px solid var(--line); padding: .45rem .6rem; text-align: left; vertical-align: top; }
th { background: #efeae1; font-weight: 600; }
.model-id, .wrap { overflow-wrap: anywhere; word-break: break-word; }
.pair {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: .45rem;
  padding: .9rem 1rem;
  margin: 0 0 .75rem;
}
.pair-top { display: flex; flex-wrap: wrap; gap: .4rem .8rem; align-items: baseline; margin-bottom: .55rem; }
.models { margin: 0 0 .7rem; color: var(--muted); overflow-wrap: anywhere; }
.outcomes { display: flex; flex-wrap: wrap; gap: .5rem; }
.outcome {
  min-width: 6.5rem;
  border: 1px solid var(--line);
  border-radius: .35rem;
  padding: .4rem .55rem;
}
.outcome .k { display: block; color: var(--muted); font-size: .75rem; }
.outcome .v { font-weight: 650; }
.pill {
  font-size: .78rem;
  padding: .1rem .5rem;
  border-radius: 999px;
}
.pill.agree { background: var(--agree-bg); color: var(--agree-fg); }
.pill.disagree { background: var(--disagree-bg); color: var(--disagree-fg); }
.muted { color: var(--muted); }
.failures { background: var(--fail-bg); border: 1px solid var(--fail-line); border-radius: .45rem; padding: .9rem 1rem; }
.failures article { margin: 0 0 .75rem; }
.failures article:last-child { margin: 0; }
.secondary details { background: var(--card); border: 1px solid var(--line); border-radius: .4rem; padding: .55rem .8rem; margin: 0 0 .5rem; }
details { margin: .4rem 0 0; }
summary { cursor: pointer; }
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f1ede4;
  border: 1px solid var(--line);
  padding: .7rem .8rem;
  font-size: .86rem;
  margin: .4rem 0 0;
}
dl { display: grid; grid-template-columns: max-content 1fr; gap: .2rem 1rem; margin: .4rem 0 0; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
@media (max-width: 640px) {
  main { padding: 1rem .8rem 2rem; }
  dl { grid-template-columns: 1fr; }
}
"#;

fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn decision(value: &JudgeDecision) -> &'static str {
    match value {
        JudgeDecision::A => "A",
        JudgeDecision::B => "B",
        JudgeDecision::Draw => "Draw",
    }
}

fn opt_decision(value: Option<&JudgeDecision>) -> String {
    value.map(decision).unwrap_or("").to_string()
}

fn opt_text(value: Option<&str>) -> String {
    value.map(escape).unwrap_or_default()
}

fn format_rating(value: f64) -> String {
    format!("{value:.2}")
}

fn format_percent(rate: f64) -> String {
    let percent = rate * 100.0;
    if (percent - percent.round()).abs() < 0.05 {
        format!("{:.0}%", percent.round())
    } else {
        format!("{percent:.1}%")
    }
}

fn rating_cell(value: Option<f64>) -> String {
    match value {
        Some(value) => format_rating(value),
        None => "unavailable".into(),
    }
}

fn unavailable(reason: UnavailableReason) -> &'static str {
    match reason {
        UnavailableReason::NoComparisons => "no_comparisons",
        UnavailableReason::Disconnected => "disconnected",
        UnavailableReason::Separated => "separated",
        UnavailableReason::Nonconvergence => "nonconvergence",
        UnavailableReason::NonfiniteResult => "nonfinite_result",
    }
}

fn bootstrap_unavailable(reason: BootstrapUnavailable) -> &'static str {
    match reason {
        BootstrapUnavailable::TooFewTasks => "too_few_tasks",
        BootstrapUnavailable::OriginalUnrated => "original_unrated",
        BootstrapUnavailable::InvalidReplicates => "invalid_replicates",
    }
}

fn agreement_label(agreement: bool) -> &'static str {
    if agreement {
        "orientation agreement"
    } else {
        "orientation disagreement"
    }
}

fn dt_dd(html: &mut String, term: &str, value: &str) {
    html.push_str("<dt>");
    html.push_str(&escape(term));
    html.push_str("</dt><dd>");
    html.push_str(value);
    html.push_str("</dd>\n");
}

fn metric(html: &mut String, label: &str, value: &str) {
    html.push_str("<div class=\"metric\"><span class=\"label\">");
    html.push_str(&escape(label));
    html.push_str("</span><span class=\"value\">");
    html.push_str(value);
    html.push_str("</span></div>\n");
}

fn push_header(
    html: &mut String,
    summary: &ReportSummary,
    config: &RunConfig<'_>,
    bootstrap: Option<&BootstrapReport>,
) {
    html.push_str("<header>\n<p class=\"eyebrow\">Arena Experiment</p>\n");
    html.push_str("<h1>");
    if let Some(judge) = config.judge {
        html.push_str("Judge: ");
        html.push_str(&escape(&judge.to_string()));
    } else {
        html.push_str("No judge");
    }
    html.push_str("</h1>\n<p class=\"meta\">");
    html.push_str(&summary.task_count.to_string());
    html.push_str(if summary.task_count == 1 {
        " task · "
    } else {
        " tasks · "
    });
    html.push_str(&summary.candidate_count.to_string());
    html.push_str(if summary.candidate_count == 1 {
        " model"
    } else {
        " models"
    });
    if summary.judge_used {
        html.push_str(" · judge run");
    } else {
        html.push_str(" · no-judge run");
    }
    html.push_str("</p>\n");
    if let Some(complete) = summary.complete {
        html.push_str("<p class=\"status\">");
        html.push_str(if complete { "COMPLETE" } else { "INCOMPLETE" });
        html.push_str("</p>\n");
    }
    if summary.judge_used {
        html.push_str("<div class=\"metrics\">\n");
        let pairs = match (summary.resolved_pairs, summary.expected_pairs) {
            (Some(resolved), Some(expected)) => format!("{resolved} / {expected}"),
            _ => String::new(),
        };
        metric(html, "Pairs", &pairs);
        if let Some(failed) = summary.failed_pairs {
            metric(html, "Failed pairs", &failed.to_string());
        }
        let agreement = match (summary.orientation_agreeing_pairs, summary.resolved_pairs) {
            (Some(agreeing), Some(resolved)) => format!("{agreeing} / {resolved}"),
            _ => String::new(),
        };
        metric(html, "Orientation agreement", &agreement);
        metric(
            html,
            "Agreement rate",
            &summary
                .agreement_rate
                .map(format_percent)
                .unwrap_or_default(),
        );
        if let Some(bootstrap) = bootstrap {
            let valid = match bootstrap.valid {
                Some(valid) => format!("{valid} / {} valid", bootstrap.replicates),
                None => format!("{} requested", bootstrap.replicates),
            };
            metric(html, "Bootstrap", &valid);
        }
        html.push_str("</div>\n");
    }
    html.push_str("</header>\n");
}

fn push_models(html: &mut String, models: &[ModelReport<'_>], judge_used: bool) {
    html.push_str("<section>\n<h2>Models</h2>\n");
    if judge_used {
        html.push_str(
            "<p class=\"note\">Bradley–Terry rating · derived from resolved pairwise judgments</p>\n",
        );
    }
    html.push_str("<div class=\"scroll\"><table>\n<thead><tr>");
    html.push_str(
        "<th>Model</th><th>W</th><th>L</th><th>D</th><th>Total</th><th>Rating</th><th>Lower bound</th><th>Upper bound</th><th>Unavailable reason</th>",
    );
    html.push_str("</tr></thead>\n<tbody>\n");
    for model in models {
        html.push_str("<tr><td class=\"model-id\">");
        html.push_str(&escape(&model.model.to_string()));
        html.push_str("</td><td>");
        html.push_str(&model.wins.to_string());
        html.push_str("</td><td>");
        html.push_str(&model.losses.to_string());
        html.push_str("</td><td>");
        html.push_str(&model.draws.to_string());
        html.push_str("</td><td>");
        html.push_str(&model.total.to_string());
        html.push_str("</td><td>");
        html.push_str(&rating_cell(model.rating));
        html.push_str("</td><td>");
        html.push_str(&rating_cell(model.rating_lower));
        html.push_str("</td><td>");
        html.push_str(&rating_cell(model.rating_upper));
        html.push_str("</td><td>");
        html.push_str(model.unavailable.map(unavailable).unwrap_or(""));
        html.push_str("</td></tr>\n");
    }
    html.push_str("</tbody></table></div>\n</section>\n");
}

fn push_results(html: &mut String, results: &[CandidateRow<'_>]) {
    if results.is_empty() {
        return;
    }
    html.push_str("<section class=\"secondary\">\n<h2>Candidate responses</h2>\n");
    html.push_str(
        "<p class=\"note\">Task-level model outputs. Collapsed by default; content is stored text, not rendered markup.</p>\n",
    );
    for result in results {
        html.push_str("<details><summary>");
        html.push_str(&escape(result.task_id));
        html.push_str(" — ");
        html.push_str(&escape(&result.model.to_string()));
        html.push_str("</summary>\n<dl>\n");
        dt_dd(html, "Task", &escape(result.task_id));
        dt_dd(html, "Model", &escape(&result.model.to_string()));
        dt_dd(html, "Duration", &format!("{} ms", result.duration_ms));
        if let Some(score) = result.score {
            dt_dd(html, "Score", &score.to_string());
        }
        html.push_str("</dl>\n<pre>");
        html.push_str(&escape(result.response));
        html.push_str("</pre>\n</details>\n");
    }
    html.push_str("</section>\n");
}

fn push_pairs(html: &mut String, pairs: &[PairRow<'_>]) {
    html.push_str("<section>\n<h2>Pairwise results</h2>\n");
    html.push_str(
        "<p class=\"note\">Resolved judgments are the primary observations. Orientation disagreement is recorded, not interpreted as position bias.</p>\n",
    );
    if pairs.is_empty() {
        html.push_str("<p>No resolved judgments.</p>\n</section>\n");
        return;
    }
    for pair in pairs {
        let pill = if pair.agreement { "agree" } else { "disagree" };
        html.push_str("<article class=\"pair\">\n<div class=\"pair-top\"><h3>");
        html.push_str(&escape(pair.task_id));
        html.push_str("</h3><span class=\"pill ");
        html.push_str(pill);
        html.push_str("\">");
        html.push_str(agreement_label(pair.agreement));
        html.push_str("</span><span class=\"muted\">");
        html.push_str(&pair.duration_ms.to_string());
        html.push_str(" ms</span></div>\n<p class=\"models\">A: ");
        html.push_str(&escape(&pair.model_a.to_string()));
        html.push_str(" · B: ");
        html.push_str(&escape(&pair.model_b.to_string()));
        html.push_str("</p>\n<div class=\"outcomes\">");
        html.push_str("<div class=\"outcome\"><span class=\"k\">AB</span><span class=\"v\">");
        html.push_str(&opt_decision(pair.orientation_ab.as_ref()));
        html.push_str("</span></div>");
        html.push_str("<div class=\"outcome\"><span class=\"k\">BA</span><span class=\"v\">");
        html.push_str(&opt_decision(pair.orientation_ba.as_ref()));
        html.push_str("</span></div>");
        html.push_str("<div class=\"outcome\"><span class=\"k\">Final</span><span class=\"v\">");
        html.push_str(decision(&pair.winner));
        html.push_str("</span></div></div>\n");
        html.push_str("<details><summary>Reasons</summary>\n<p><strong>AB</strong><br>");
        html.push_str(&opt_text(pair.reason_ab));
        html.push_str("</p>\n<p><strong>BA</strong><br>");
        html.push_str(&opt_text(pair.reason_ba));
        html.push_str("</p>\n</details>\n</article>\n");
    }
    html.push_str("</section>\n");
}

fn push_audit(html: &mut String, pairs: &[PairRow<'_>]) {
    if pairs.is_empty() {
        return;
    }
    html.push_str("<section class=\"secondary\">\n<h2>Audit details</h2>\n");
    html.push_str(
        "<p class=\"note\">Raw judge completions and orientation records. Secondary to the pairwise outcomes above.</p>\n",
    );
    for pair in pairs {
        html.push_str("<details>\n<summary>");
        html.push_str(&escape(pair.task_id));
        html.push_str(" — ");
        html.push_str(&escape(&pair.model_a.to_string()));
        html.push_str(" vs ");
        html.push_str(&escape(&pair.model_b.to_string()));
        html.push_str("</summary>\n<dl>\n");
        dt_dd(html, "Task ID", &escape(pair.task_id));
        dt_dd(html, "Model A", &escape(&pair.model_a.to_string()));
        dt_dd(html, "Model B", &escape(&pair.model_b.to_string()));
        dt_dd(html, "Final winner", decision(&pair.winner));
        dt_dd(
            html,
            "Orientation agreement",
            agreement_label(pair.agreement),
        );
        dt_dd(
            html,
            "AB winner",
            &opt_decision(pair.orientation_ab.as_ref()),
        );
        dt_dd(
            html,
            "BA winner",
            &opt_decision(pair.orientation_ba.as_ref()),
        );
        dt_dd(html, "Reason AB", &opt_text(pair.reason_ab));
        dt_dd(html, "Reason BA", &opt_text(pair.reason_ba));
        dt_dd(html, "Duration", &format!("{} ms", pair.duration_ms));
        html.push_str("<dt>Raw AB completion</dt><dd><pre>");
        html.push_str(&opt_text(pair.raw_ab));
        html.push_str("</pre></dd>\n");
        html.push_str("<dt>Raw BA completion</dt><dd><pre>");
        html.push_str(&opt_text(pair.raw_ba));
        html.push_str("</pre></dd>\n");
        html.push_str("</dl>\n</details>\n");
    }
    html.push_str("</section>\n");
}

fn push_failures(html: &mut String, failures: &[FailedPairRow<'_>]) {
    html.push_str("<section class=\"failures\">\n<h2>Failed judgments</h2>\n");
    html.push_str(
        "<p class=\"note\">These pairs did not produce a resolved judgment and are not mixed into the pairwise results above.</p>\n",
    );
    for failure in failures {
        html.push_str("<article>\n<dl>\n");
        dt_dd(html, "Task ID", &escape(failure.task_id));
        dt_dd(html, "Model A", &escape(&failure.model_a.to_string()));
        dt_dd(html, "Model B", &escape(&failure.model_b.to_string()));
        html.push_str("<dt>Failure information</dt><dd><ul>\n");
        for item in failure.orientations {
            html.push_str("<li>");
            html.push_str(&escape(&failure_line(item)));
            html.push_str("</li>\n");
        }
        html.push_str("</ul></dd>\n</dl>\n</article>\n");
    }
    html.push_str("</section>\n");
}

fn failure_line(item: &OrientationFailure) -> String {
    format!(
        "{} {}: {} (attempts: {})",
        item.orientation.as_str(),
        item.kind.as_str(),
        item.error,
        item.attempts
    )
}

fn push_bootstrap(html: &mut String, bootstrap: &BootstrapReport) {
    html.push_str("<section>\n<h2>Bootstrap</h2>\n");
    html.push_str(
        "<p class=\"note\">Task-clustered resampling of the observed-pair rating estimator. Point ratings and interval availability are distinct.</p>\n",
    );
    html.push_str("<div class=\"metrics\">\n");
    metric(
        html,
        "Requested replicates",
        &bootstrap.replicates.to_string(),
    );
    metric(
        html,
        "Valid replicates",
        &bootstrap
            .valid
            .map(|value| value.to_string())
            .unwrap_or_default(),
    );
    metric(html, "Task clusters", &bootstrap.clusters.to_string());
    metric(html, "Ran", if bootstrap.ran { "yes" } else { "no" });
    metric(
        html,
        "Rating intervals",
        if bootstrap.bounds_present {
            "present"
        } else {
            "unavailable"
        },
    );
    html.push_str("</div>\n<dl>\n");
    dt_dd(html, "Seed", &bootstrap.seed.to_string());
    dt_dd(
        html,
        "Unavailable reason",
        bootstrap
            .unavailable
            .map(bootstrap_unavailable)
            .unwrap_or(""),
    );
    html.push_str("</dl>\n</section>\n");
}

fn push_config(html: &mut String, config: &RunConfig<'_>) {
    html.push_str(
        "<section class=\"secondary\">\n<details>\n<summary>Run configuration</summary>\n<dl>\n",
    );
    let models = config
        .models
        .iter()
        .map(|model| escape(&model.to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    dt_dd(html, "Models", &models);
    dt_dd(
        html,
        "Judge",
        &config
            .judge
            .map(|id| escape(&id.to_string()))
            .unwrap_or_else(|| "not used".into()),
    );
    dt_dd(
        html,
        "Task path",
        &config
            .tasks
            .map(|path| escape(&path.display().to_string()))
            .unwrap_or_default(),
    );
    dt_dd(html, "Base URL", &escape(config.base_url));
    dt_dd(html, "Started at", &escape(config.started_at));
    dt_dd(
        html,
        "Provider concurrency",
        &config.provider_concurrency.to_string(),
    );
    dt_dd(
        html,
        "Request timeout",
        &format!("{} s", config.request_timeout_secs),
    );
    dt_dd(
        html,
        "Connect timeout",
        &format!("{} s", config.connect_timeout_secs),
    );
    dt_dd(
        html,
        "Candidate attempts",
        &config.candidate_attempts.to_string(),
    );
    dt_dd(html, "Judge attempts", &config.judge_attempts.to_string());
    dt_dd(
        html,
        "Candidate max tokens",
        &config.candidate_max_tokens.to_string(),
    );
    if let Some(decoding) = config.judge_decoding {
        dt_dd(
            html,
            "Judge decoding",
            &format!(
                "temperature {}, max_tokens {}",
                decoding.temperature, decoding.max_tokens
            ),
        );
    }
    html.push_str("</dl>\n</details>\n</section>\n");
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::bootstrap::{BootstrapMeta, BootstrapUnavailable};
    use crate::evaluate::EvaluatedResult;
    use crate::judge::{
        JudgeOrientation, Judgment, JudgmentFailure, JudgmentFailureKind, OrientationFailure,
    };
    use crate::persist::{JudgeDecoding, Output, RunMetadata};
    use crate::provider::{CompletionResponse, ModelId};
    use crate::rating::{ModelRating, UnavailableReason};
    use crate::report;
    use crate::stats::{self, PairAgreement};
    use crate::task::Task;

    fn run_meta(models: &[&str], judge: Option<&str>) -> RunMetadata {
        RunMetadata::new(
            models.iter().map(|id| ModelId::new(*id)).collect(),
            judge.map(ModelId::new),
            Some(PathBuf::from("tasks.json")),
            "2026-01-02T03:04:05Z".into(),
            "https://example.test/v1",
        )
    }

    fn task(id: &str) -> Task {
        Task {
            id: id.into(),
            prompt: "p".into(),
            evaluation: None,
        }
    }

    fn judgment() -> Judgment {
        Judgment {
            task_id: "t1".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            judge_model: ModelId::new("judge"),
            winner: JudgeDecision::Draw,
            reason: String::new(),
            duration_ms: 12,
            agreement: false,
            orientation_ab: Some(JudgeDecision::A),
            orientation_ba: Some(JudgeDecision::B),
            reason_ab: Some("a < b & c".into()),
            reason_ba: Some("ba".into()),
            raw_ab: Some("</pre><script>alert(1)</script>".into()),
            raw_ba: Some(r#"{"winner":"b","reason":"ba"}"#.into()),
            raw: None,
        }
    }

    fn output(
        run: RunMetadata,
        tasks: Vec<Task>,
        results: Vec<EvaluatedResult>,
        judgments: Option<Vec<Judgment>>,
        judgment_failures: Option<Vec<JudgmentFailure>>,
        statistics: Option<Vec<crate::stats::ModelStats>>,
        ratings: Option<Vec<ModelRating>>,
    ) -> Output {
        Output {
            run,
            tasks,
            results,
            comparisons: vec![],
            judgments,
            judgment_failures,
            statistics,
            ratings,
        }
    }

    fn render_output(data: &Output) -> String {
        render(&report::from_output(data))
    }

    #[test]
    fn complete_report_renders_coverage_and_agreement() {
        let judgments = vec![judgment()];
        let models = vec![ModelId::new("m0"), ModelId::new("m1")];
        let data = output(
            run_meta(&["m0", "m1"], Some("judge"))
                .with_judge_coverage(1, 1, 0)
                .with_orientation_agreement(stats::pair_agreement(&judgments))
                .with_judge_decoding(JudgeDecoding::arena_default())
                .with_bootstrap(&BootstrapMeta {
                    seed: 0,
                    replicates: 1000,
                    valid: Some(1000),
                    clusters: 2,
                    ran: true,
                    unavailable: None,
                }),
            vec![task("t1")],
            vec![],
            Some(judgments),
            Some(vec![]),
            Some(stats::aggregate(&[], &models)),
            Some(vec![
                ModelRating {
                    model: ModelId::new("m0"),
                    rating: Some(1600.0),
                    rating_lower: Some(1500.0),
                    rating_upper: Some(1700.0),
                    unavailable: None,
                },
                ModelRating {
                    model: ModelId::new("m1"),
                    rating: Some(1400.0),
                    rating_lower: Some(1300.0),
                    rating_upper: Some(1500.0),
                    unavailable: None,
                },
            ]),
        );
        let html = render_output(&data);
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<html"));
        assert!(html.contains("<head>"));
        assert!(html.contains("<style>"));
        assert!(html.contains("<body>"));
        assert!(html.contains("</html>"));
        assert!(html.contains("Arena Experiment"));
        assert!(html.contains("COMPLETE"));
        assert!(html.contains("1 / 1"));
        assert!(html.contains("Orientation agreement"));
        assert!(html.contains("100%"));
        assert!(html.contains("Pairwise results"));
        assert!(html.contains("m0"));
        assert!(html.contains("m1"));
        assert!(!html.contains("https://cdn"));
    }

    #[test]
    fn no_judge_report_does_not_invent_pairwise_coverage() {
        let data = output(
            run_meta(&["m0"], None),
            vec![task("t1")],
            vec![EvaluatedResult {
                task_id: "t1".into(),
                model: ModelId::new("m0"),
                response: CompletionResponse {
                    text: "hello <world>".into(),
                },
                evaluation: None,
                duration_ms: 5,
            }],
            None,
            None,
            None,
            None,
        );
        let html = render_output(&data);
        assert!(html.contains("No judge"));
        assert!(html.contains("no-judge run"));
        assert!(html.contains("m0"));
        assert!(html.contains("hello &lt;world&gt;"));
        assert!(html.contains("Candidate responses"));
        assert!(html.contains("Run configuration"));
        assert!(!html.contains("Orientation agreement"));
        assert!(!html.contains("Pairwise results"));
        assert!(!html.contains("Failed judgments"));
        assert!(!html.contains("Bootstrap"));
        assert!(!html.contains("no_comparisons"));
    }

    #[test]
    fn incomplete_report_keeps_failed_judgments_separate() {
        let failure = JudgmentFailure {
            task_id: "t2".into(),
            model_a: ModelId::new("m0"),
            model_b: ModelId::new("m1"),
            judge_model: ModelId::new("judge"),
            orientations: vec![OrientationFailure {
                orientation: JudgeOrientation::Ab,
                kind: JudgmentFailureKind::InvalidJson,
                error: "no valid judgment JSON found".into(),
                attempts: 3,
            }],
        };
        let data = output(
            run_meta(&["m0", "m1"], Some("judge"))
                .with_judge_coverage(1, 0, 1)
                .with_orientation_agreement(PairAgreement {
                    resolved_pairs: 0,
                    orientation_agreeing_pairs: 0,
                    orientation_disagreeing_pairs: 0,
                    agreement_rate: 0.0,
                })
                .with_judge_decoding(JudgeDecoding::arena_default()),
            vec![task("t2")],
            vec![],
            Some(vec![]),
            Some(vec![failure]),
            None,
            None,
        );
        let html = render_output(&data);
        assert!(html.contains("INCOMPLETE"));
        assert!(html.contains("Failed judgments"));
        assert!(html.contains("t2"));
        assert!(html.contains("invalid_json"));
        assert!(html.contains("class=\"failures\""));
        let pairwise = html.find("Pairwise results").unwrap();
        let failed = html.find("Failed judgments").unwrap();
        assert!(pairwise < failed);
    }

    #[test]
    fn pairwise_and_audit_keep_orientation_and_escape_raw_text() {
        let data = output(
            run_meta(&["m0", "m1"], Some("judge"))
                .with_judge_coverage(1, 1, 0)
                .with_orientation_agreement(stats::pair_agreement(&[judgment()])),
            vec![task("t1")],
            vec![],
            Some(vec![judgment()]),
            Some(vec![]),
            None,
            None,
        );
        let html = render_output(&data);
        assert!(html.contains("AB"));
        assert!(html.contains("BA"));
        assert!(html.contains("Final"));
        assert!(html.contains("Draw"));
        assert!(html.contains("orientation disagreement"));
        assert!(html.contains("a &lt; b &amp; c"));
        assert!(html.contains("&lt;/pre&gt;&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(!html.contains("</pre><script>alert(1)</script>"));
        assert!(html.contains("Raw AB completion"));
        assert!(
            html.contains("{&quot;winner&quot;:&quot;b&quot;,&quot;reason&quot;:&quot;ba&quot;}")
        );
        assert!(!html.contains("{\"winner\":\"b\",\"reason\":\"ba\"}"));
    }

    #[test]
    fn bootstrap_invalid_replicates_is_rendered_without_reinterpretation() {
        let data = output(
            run_meta(&["m0"], Some("judge"))
                .with_judge_coverage(0, 0, 0)
                .with_orientation_agreement(PairAgreement {
                    resolved_pairs: 0,
                    orientation_agreeing_pairs: 0,
                    orientation_disagreeing_pairs: 0,
                    agreement_rate: 0.0,
                })
                .with_bootstrap(&BootstrapMeta {
                    seed: 7,
                    replicates: 1000,
                    valid: Some(744),
                    clusters: 2,
                    ran: true,
                    unavailable: Some(BootstrapUnavailable::InvalidReplicates),
                }),
            vec![task("t1"), task("t2")],
            vec![],
            Some(vec![]),
            Some(vec![]),
            None,
            Some(vec![ModelRating {
                model: ModelId::new("m0"),
                rating: Some(1500.0),
                rating_lower: None,
                rating_upper: None,
                unavailable: None,
            }]),
        );
        let html = render_output(&data);
        assert!(html.contains("744 / 1000 valid"));
        assert!(html.contains("1000"));
        assert!(html.contains("744"));
        assert!(html.contains("invalid_replicates"));
        assert!(html.contains("Rating intervals"));
        assert!(html.contains("unavailable"));
        assert!(html.contains(">yes<"));
        assert!(!html.contains("implementation failed"));
        assert!(!html.contains("bootstrap failed"));
    }

    #[test]
    fn unavailable_ratings_are_explicit_and_requested_models_remain_present() {
        let data = output(
            run_meta(&["m0", "m1", "m2"], Some("judge")).with_judge_coverage(3, 0, 3),
            vec![task("t1")],
            vec![],
            Some(vec![]),
            Some(vec![]),
            Some(stats::aggregate(
                &[],
                &[ModelId::new("m0"), ModelId::new("m1"), ModelId::new("m2")],
            )),
            Some(vec![
                ModelRating {
                    model: ModelId::new("m0"),
                    rating: Some(1595.4242509439325),
                    rating_lower: None,
                    rating_upper: None,
                    unavailable: None,
                },
                ModelRating {
                    model: ModelId::new("m1"),
                    rating: None,
                    rating_lower: None,
                    rating_upper: None,
                    unavailable: Some(UnavailableReason::NoComparisons),
                },
                ModelRating {
                    model: ModelId::new("m2"),
                    rating: None,
                    rating_lower: None,
                    rating_upper: None,
                    unavailable: Some(UnavailableReason::NoComparisons),
                },
            ]),
        );
        let report = report::from_output(&data);
        assert_eq!(report.models[0].rating, Some(1595.4242509439325));
        let html = render(&report);
        assert!(html.contains("m0"));
        assert!(html.contains("m1"));
        assert!(html.contains("m2"));
        assert!(html.contains("1595.42"));
        assert!(!html.contains("1595.4242509439325"));
        assert!(html.contains("unavailable"));
        assert!(html.contains("no_comparisons"));
        assert!(html.contains("Bradley–Terry rating · derived from resolved pairwise judgments"));
    }
}
