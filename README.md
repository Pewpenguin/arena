# Arena

Model-vs-model evaluation for LLMs.

Arena runs the same tasks against multiple models, supports exact-match evaluation, and can compare model outputs using an LLM judge.

The core uses a `ModelProvider` trait, with an OpenAI-compatible adapter and native OpenRouter, Anthropic, and Gemini providers.

## Features

- Run a single model against a prompt
- Run multiple models against the same task set
- Tournament schedules: round robin, single elimination, and king of the hill
- Best-of-N series for each scheduled contest
- Concurrent candidate and judge requests with bounded provider concurrency
- Exact-match evaluation
- Pairwise LLM judging with both response orders
- Agreement tracking for orientation disagreement
- Win/loss/draw statistics
- Bradley–Terry ratings with task-clustered bootstrap uncertainty
- Request duration and run provenance
- JSON output

## Setup

Requires Rust 1.88 or newer and Cargo.

The `run` and `exec` commands default to an OpenAI-compatible API endpoint. Set the API key in `.env`:

    cp .env.example .env

Then set:

    ARENA_API_KEY=your_api_key

`ARENA_BASE_URL` is optional and defaults to `https://api.openai.com/v1`. Set it to any service exposing the OpenAI-compatible API subset Arena requires, and pass that service's key through `ARENA_API_KEY`. OpenRouter, Anthropic, and Gemini use their own API keys and do not read `ARENA_BASE_URL`. See [Providers](#providers).

Build the project:

    cargo build

## Usage

Run one model:

    cargo run -- run \
      --model MODEL \
      --prompt "Explain Rust ownership in two sentences."

Run multiple models against a task file:

    cargo run -- exec \
      --tasks tasks.json \
      --model MODEL_A \
      --model MODEL_B \
      --judge MODEL_JUDGE \
      --tournament round-robin \
      --best-of 1 \
      --output results.json

`--model` can be specified multiple times. Model IDs must be unique. `--judge` must not match any `--model`. `--judge` and `--output` are optional. `--tournament` selects `round-robin` (the default), `single-elimination`, or `king-of-the-hill`. `--best-of` sets the series length and must be an odd positive integer; the default is `1`. `--seed` defaults to `0` and, when a judge is used, seeds bootstrap resampling and elimination seeded fallback; it does not control ordinary scheduling or provider completions. `--provider` selects `openai` (the default), `openrouter`, `anthropic`, or `gemini`.

Without `--output`, the JSON result is written to stdout. Each candidate provider request is tried up to three times (a 1s backoff, then 2s) for provider errors and invalid provider responses. An exhausted candidate failure still aborts the run without producing output. A permanently failed judged game is saved in the output with `run.complete` set to false, and the process then exits non-zero.

Render a persisted experiment JSON file as a self-contained HTML audit report:

    cargo run -- report \
      --input results.json \
      --output report.html

Start the local web UI:

    cargo run -- web

It listens on `127.0.0.1:3030`. `--port` selects another port:

    cargo run -- web --port 8080

The page asks for an API key directly. It does not read `.env` or `ARENA_API_KEY`.

The web workbench exposes the same tournament formats and Best-of-N controls as `exec`. Custom first-round opening matchups for single elimination are web-only; CLI `exec` has no opening-matchup flag and always pairs the first round automatically from candidate order. King-of-the-hill challenge order is the candidate list order in both places (`--model` order on the CLI; the ordered candidate list on the web start form).

A completed or incomplete web experiment is written to `arena-web/{run_id}.json`. Run IDs are unique across process restarts. A candidate execution failure does not write a file. `arena-web/` is gitignored. Pass a saved web experiment to `arena report`:

    cargo run -- report \
      --input arena-web/RUN_ID.json \
      --output report.html

## Tasks

Tasks are defined as a JSON array. Task IDs must be unique. Unknown fields are rejected.

    [
      {
        "id": "t1",
        "prompt": "Explain Rust ownership in two sentences."
      },
      {
        "id": "t2",
        "prompt": "What is the capital of France? Answer with only the city name.",
        "evaluation": {
          "type": "exact",
          "expected": "Paris"
        }
      }
    ]

`evaluation` is optional. The only supported evaluation type is `exact`.

Exact evaluation trims the candidate response and compares it with `expected`. A match scores `1.0`; otherwise it scores `0.0`. The comparison is case-sensitive.

Exact evaluation and LLM judging are separate. Exact scores produce `comparisons`. `--judge` always produces `judgments`, `judgment_failures`, `statistics`, `ratings`, tournament metadata, and game-coverage metadata, including when those collections are empty. Those judge fields are omitted only when `--judge` is not used. Without a judge, `tournament` is still present with status `not_judged` and no per-task matches.

## Tournaments

When a judge is configured, Arena schedules contests according to `--tournament` (CLI) or the web Tournament control. Each scheduled contest is a **series** between two candidates. A series consists of one or more **games**. Each game is one judged evaluation: both response orientations, resolved as today.

| Term | Meaning |
| --- | --- |
| Pairing | A model-vs-model relationship |
| Series | One tournament contest between two models |
| Game | One judged evaluation (AB and BA orientations combined into one resolved judgment or failure) |

### Formats

**Round robin** (`round-robin`, default)

- Schedules every unordered candidate pairing once per task.
- Useful when you want full pairwise evidence for statistics and ratings.
- Each pairing plays one Best-of-N series.
- A drawn series stays a draw. Round robin does not run elimination tie-breaks or seeded fallback.

**Single elimination** (`single-elimination`)

- Requires a power-of-two candidate count (2, 4, 8, …).
- Winners advance through rounds until one champion remains, or the bracket stops without a champion (for example after a judgment failure).
- First-round pairing follows candidate order by default (adjacent pairs).
- Custom opening matchups are available in the web workbench only. CLI `exec` has no opening-matchup option.
- Uses Best-of-N series. A drawn series may play elimination tie-breaks (see below).

**King of the hill** (`king-of-the-hill`)

- Any candidate count of at least two is allowed.
- Candidates challenge in order: the current hill holder faces the next challenger.
- The series winner remains on the hill; the loser does not return.
- Candidate order matters. On the CLI, that order is the `--model` list. In the web UI, it is the ordered candidate list on the start form.
- The final hill holder is the champion when the challenge sequence completes.
- Uses the same Best-of-N series and elimination tie-break / seeded-fallback rules as single elimination.

### Best-of-N

`--best-of` / the web Best of control:

- Must be an odd positive integer (`1`, `3`, `5`, …).
- Default is `1` (a single judged game per series unless elimination tie-breaks apply).
- A series stops early once one candidate reaches the majority needed to win (`best_of / 2 + 1` wins).
- A judged game draw does not count as a win for either side.
- If neither side can still reach that majority with the remaining regulation games, the series is a draw.
- For single elimination and king of the hill only: a drawn series then plays up to three additional tie-break games, one at a time. The first decisive tie-break advances that candidate.
- If those three tie-breaks are also all draws, Arena applies a **seeded fallback**: it chooses an advancing candidate from `--seed` and the two model IDs. Seeded fallback is deterministic for a given seed and pair, and it is **not** a judged win. The match is marked with `seeded_fallback` in the tournament record.

Round robin never enters the tie-break / seeded-fallback path.

### Coverage and progress

Persisted judge coverage counts **games**, not distinct pairings and not series:

- `run.expected_pairs` — games submitted to the judge in this run (`tournament.judged_match_count()`)
- `run.resolved_pairs` — resolved judgments (one per successful game)
- `run.failed_pairs` — permanently failed games

The JSON field names still say `pairs` for compatibility. For a finished judge run, `resolved_pairs + failed_pairs == expected_pairs`. A mismatch is reported as an error rather than rewritten in the JSON.

`run.complete` is `true` only when `resolved_pairs == expected_pairs` and `failed_pairs` is 0.

Planned or progress totals can differ from those final game counts because:

- Best-of-N can stop a series early
- elimination tie-breaks can add games beyond Best-of-N
- a judgment failure can stop further games in a series or bracket

CLI and web progress bars estimate work from the format's scheduled series count × tasks × `--best-of`. That estimate is not the same number as the final `expected_pairs` game count in the saved JSON.

### `--seed`

With `--judge`, the same `--seed` value (default `0`) is used for:

1. **Bootstrap resampling** of ratings (see [Ratings](#ratings))
2. **Seeded fallback** after three drawn elimination tie-breaks in single elimination or king of the hill

It does not:

- choose which pairs are scheduled (except indirectly: seeded fallback only decides who advances after exhausted tie-breaks)
- reorder candidates
- affect provider sampling or retry timing
- make candidate or judge completions deterministic

Without `--judge`, `--seed` has no effect on the run. Reproducing bootstrap intervals from a saved file requires the same seed and the same persisted judgments; it does not replay providers.

## Judging

For each game, Arena runs the judge twice:

    (A, B)
    (B, A)

The second result is mapped back to the original model order.

If both orientations agree, the judgment is kept with `agreement: true`.

If they disagree, Arena resolves the game as a draw with `agreement: false`. Orientation disagreement is not automatically evidence of position bias: it can also come from remaining judge noise, formatting, or other order effects.

Both orientation winners, reasons, and raw judge completions are persisted, mapped back to the original model pair. Statistics and Bradley–Terry ratings use only the resolved winner. The two orientations are retained for diagnostics and are not counted as independent ranking observations.

Each orientation is tried up to three times (a 1s backoff, then 2s) for provider errors and malformed/truncated JSON. The parser stays strict: the entire judge completion must be one JSON object in the required schema. Surrounding text, extra objects, or JSON copied from a candidate reply are rejected rather than scanned for a last match. A retry repeats the same judge request rather than repairing invalid output. Candidate replies are isolated in `<response_a>` / `<response_b>` tags in the judge prompt; `<` in that quoted text is escaped so it cannot close the fence.

Judge requests set `temperature` to `0`. That is persisted as `run.judge_decoding.temperature` and applies only to judge calls, not candidate generation. Temperature 0 asks the provider for deterministic decoding; it does not guarantee identical completions across providers or models.

Candidate and judge requests set `max_tokens` to `4096`. That value is the requested output budget, not a guarantee that prompt tokens plus output tokens fit every provider's context window. It is persisted as `run.candidate_max_tokens` and, when a judge is used, `run.judge_decoding.max_tokens`.

Both orientations must succeed to produce a resolved judgment. A permanently failed game is omitted from `judgments` (it is not a draw) and recorded in `judgment_failures`. The run still writes its output, then exits non-zero.

`run.orientation_agreement` counts each resolved game once: `orientation_agreeing_pairs`, `orientation_disagreeing_pairs`, and `agreement_rate` (`orientation_agreeing_pairs / resolved_pairs`, or `0.0` when there are no resolved games). That summary is the game-level agreement metric. Per-model `agreement_count` / `disagreement_count` attribute the same game to both endpoints; they are not additional independent observations, and they are not a position-bias estimate.

Game-coverage fields (`expected_pairs`, `resolved_pairs`, `failed_pairs`, `complete`) are described under [Tournaments](#tournaments).

## Output

`arena exec` produces one JSON object with these fields:

- `run` — version, models, judge, task path, resolved provider `base_url`, start time, provider concurrency, request/connect timeouts, candidate/judge attempt counts, and `candidate_max_tokens`. Judge-only fields below are omitted when `--judge` is not used.
- `tasks` — the loaded task definitions (id, prompt, optional exact evaluation)
- `results` — model responses, evaluations, and durations
- `comparisons` — pairwise comparisons from exact scores
- `judgments` — resolved LLM-judge results, both orientation winners, both orientation reasons, and both raw judge completions
- `judgment_failures` — judged games that never produced both orientations
- `statistics` — per-model wins, losses, draws, and per-model orientation-agreement counts among that model's resolved games
- `ratings` — one record per requested candidate: a full-data Bradley–Terry point estimate from resolved judgments, on a 400-point scale centered at 1500, optional 95% percentile bounds, or `rating: null` with an `unavailable` reason
- `tournament` — format, candidates, status, Best-of-N, optional opening matchups, and per-task series/game results. Present for judge and no-judge runs; without a judge the status is `not_judged` and `tasks` is empty.

When `--judge` is used, `judgments`, `judgment_failures`, `statistics`, and `ratings` are always present. Empty arrays mean the judge ran and that collection has no entries. Absence of those keys means no judge was configured. `results` and `comparisons` keep their existing schema.

Judge-run `run` fields:

- `expected_pairs`, `resolved_pairs`, `failed_pairs` — game coverage (field names retained for compatibility). See [Tournaments](#tournaments). `complete` is true only when `resolved_pairs == expected_pairs` and `failed_pairs == 0`.
- `orientation_agreement` — game-level orientation agreement: `resolved_pairs`, `orientation_agreeing_pairs`, `orientation_disagreeing_pairs`, `agreement_rate`. Each resolved game is counted once. `agreement_rate` is agreeing / resolved, or `0.0` when `resolved_pairs` is 0. This is not a measure of position bias.
- `judge_decoding` — judge request decoding (`temperature` 0 and `max_tokens` 4096)
- `bootstrap_seed`, `bootstrap_replicates`, `bootstrap_clusters`, `bootstrap_ran` — bootstrap request and whether resampling ran
- `bootstrap_valid` — finite replicate count; present only when the resampling loop ran
- `bootstrap_unavailable` — why interval bounds were not produced (`too_few_tasks`, `original_unrated`, or `invalid_replicates`)

The process still exits non-zero on incomplete judging; the JSON is the record of completeness, not the exit code. Requested candidates remain in `ratings` even with no resolved judgments: `rating` is `null`, bounds are omitted, and `unavailable` is set (typically `no_comparisons`). Missing judgments can disconnect or separate the comparison graph, and the gaps need not be random. Statistics and ratings from an incomplete run describe only the observed games and should not be read as if every scheduled contest had been observed.

`results[].duration_ms` is the total candidate execution duration after it acquires a provider permit, including retries and retry backoff.

`judgments[].duration_ms` is the wall-clock time from when the first of a game's two judge orientations begins work after acquiring a provider permit until both orientations have completed. It excludes time the game spends queued behind other provider calls, and it is not the sum of the two orientation request times.

Results retain task-file and CLI model order. Statistics and ratings are ordered by model ID.

The saved run is an audit of one execution: models, judge, task file path, the loaded tasks, the provider base URL, Arena-controlled concurrency, timeout, retry-attempt, and output-token settings, and, when a judge is used, game coverage, pair-level orientation agreement, the judge decoding configuration, rating availability, bootstrap metadata, and the tournament record. It does not store API keys. Candidate completions omit temperature so the provider default applies; they are not deterministic. Judge calls request `temperature` 0, but providers and models may still vary. `--seed` controls bootstrap resampling and elimination seeded fallback only; it does not make model completions deterministic. When intervals were produced, the same seed reproduces them from the persisted judgments. Raw judge completions are kept so a later audit can see what was parsed. `max_tokens` bounds the requested completion length only.

## Reporting

The `report` module derives an experiment/audit view from persisted `Output`. It does not write a second schema, recompute coverage, or fit a new rating. Judged games remain the primary observations; win/loss totals, Bradley–Terry ratings, and bootstrap intervals are derived statistics on the observed games. Tournament brackets and series outcomes are taken from the saved `tournament` object when present. No-judge runs are reported as no-judge rather than filling in judge coverage.

`arena report` turns that view into a self-contained HTML audit document. The page is a compact experiment dashboard: header and coverage first, tournament results when present, pairwise judgments as the primary evidence, then collapsed candidate responses, raw judge completions, bootstrap metadata, and run configuration. Ratings are shown as derived statistics. The HTML is generated from `report::Report`, embeds its CSS, and has no external assets, JavaScript, or frontend dependencies. It is an inspectable rendering of one persisted experiment, not the final Arena interface.

## Ratings

Ratings are relative Bradley–Terry strengths fitted to resolved pairwise judgments on this run's observed games. They are not a general-purpose leaderboard and are not comparable across different task files, judges, missingness patterns, or tournament configurations. Round robin, single elimination, and king of the hill produce different numbers and structures of judgments for the same candidate set; do not compare ratings casually across those schedules or across different Best-of-N settings.

Draws enter the likelihood as half-wins: a draw between A and B contributes 0.5 to each model's win total. Arena does not fit a separate draw parameter. A finite maximum-likelihood rating exists only when the directed graph with an edge `i → j` whenever `i` has a positive win mass against `j` (including 0.5 from a draw) is strongly connected, and the comparison graph is a single comparable component. Isolated models are `no_comparisons`. Two or more comparable components are `disconnected` and are not placed on one scale. Complete separation is `separated`. If the Hunter MM iteration does not meet tolerance, the reason is `nonconvergence`. If it meets tolerance but the resulting strengths or Elo-like transform are not finite and positive, the reason is `nonfinite_result`. Unavailable models have `rating: null` and that reason; they are not assigned 1500.

The 1500 center and 400-point scale are a convention on this relative log-strength scale: 400 points is a 10× strength ratio. Exact-score `comparisons` are not used. Seeded-fallback advances are tournament outcomes only; they are not judged games and do not enter the rating likelihood.

Uncertainty, when present, is a task-clustered percentile bootstrap of that same estimator. Each unique `task_id` among resolved judgments is one cluster; resampling a cluster carries all of that task's resolved games together. The two judge orientations are not independent observations and are not resampled. The default is 1,000 replicates, seed from `--seed` (default `0`), and 95% Hyndman–Fan type-7 percentile bounds (`rating_lower`, `rating_upper`). `bootstrap_clusters` is the number of those task IDs, not the number of rows in the task file. Point estimates always come from the full observed sample; bounds are additional.

Bounds are omitted when they cannot be computed. `bootstrap_ran` is whether the resampling loop executed. `bootstrap_valid` is the number of finite replicates and is omitted when the loop did not run (it is not stored as 0 to mean "skipped"). `bootstrap_unavailable` is one of:

- `too_few_tasks` — fewer than two task clusters among resolved judgments
- `original_unrated` — at least one requested model has no finite full-data rating
- `invalid_replicates` — the loop ran, but at least one replicate did not produce finite ratings for every model

The interval is task-sampling variability of this observed-judgment estimator. It does not measure judge noise, residual position effects, task-selection bias, or performance on a different task distribution. It is not a test of pairwise rating differences.

## Providers

`run` and `exec` take `--provider`. Every provider uses the same request timeout, connection timeout, and retry behavior. API keys stay in the provider client: they are omitted from debug output and are not written to run JSON or HTML reports. The web UI selects OpenAI, Claude, Gemini, or an OpenAI-compatible endpoint; the key stays in server memory and is not written into the page or the saved run.

### OpenAI-compatible

`OpenAICompatibleProvider` is the generic adapter for a Chat Completions API. It reads `ARENA_API_KEY` and optional `ARENA_BASE_URL`.

    ARENA_API_KEY=your_api_key
    ARENA_BASE_URL=https://api.openai.com/v1

    cargo run -- run \
      --provider openai \
      --model MODEL \
      --prompt "Explain Rust ownership in two sentences."

Candidate requests set `max_tokens`. Judge requests set `temperature` to `0` and `max_tokens`. That Chat Completions request shape is not compatible with OpenAI o-series and other reasoning models that reject `max_tokens` or `temperature`. Use a chat-completions model or server that accepts those fields.

### OpenRouter

`OpenRouterProvider` calls OpenRouter's chat completions API with `ARENA_OPENROUTER_API_KEY`. It has its own client and does not read `ARENA_API_KEY` or `ARENA_BASE_URL`.

    ARENA_OPENROUTER_API_KEY=your_openrouter_key

    cargo run -- run \
      --provider openrouter \
      --model vendor/model \
      --prompt "Explain Rust ownership in two sentences."

### Anthropic

`AnthropicProvider` calls Anthropic's Messages API with `ARENA_ANTHROPIC_API_KEY`. Temperature and max tokens are sent as `temperature` and `max_tokens`.

    ARENA_ANTHROPIC_API_KEY=your_anthropic_key

    cargo run -- run \
      --provider anthropic \
      --model MODEL \
      --prompt "Explain Rust ownership in two sentences."

### Gemini

`GeminiProvider` calls Gemini's `generateContent` API with `ARENA_GEMINI_API_KEY`. Temperature and max tokens are sent as `generationConfig.temperature` and `generationConfig.maxOutputTokens`.

    ARENA_GEMINI_API_KEY=your_gemini_key

    cargo run -- run \
      --provider gemini \
      --model MODEL \
      --prompt "Explain Rust ownership in two sentences."

## Limitations

- LLM judgments are not ground truth.
- Running both response orders can reveal orientation disagreement, but does not establish that a judge is unbiased, and disagreement is not automatically evidence of position bias.
- An orientation disagreement is treated as a draw for statistics and ratings.
- A missing orientation is omitted, not treated as a draw.
- Statistics and ratings from an incomplete judge run describe only the observed games.
- Exact-score comparisons are not used by the Bradley–Terry calculation.
- Ratings from an incomplete comparison graph, or with `unavailable` set, are not a complete ranking.
- Ratings should not be compared casually across tournament formats or Best-of-N settings: those schedules produce different judgment sets.
- Seeded fallback advances a candidate after exhausted elimination tie-breaks; it is not a judged win and does not enter ratings.
- Bootstrap intervals measure task-sampling variability of the observed-game estimator only. They do not account for judge noise, position effects, or task-selection bias, and they are not a test of pairwise rating differences.
- Judge `temperature` 0 does not guarantee identical outputs across providers or models.
- `max_tokens` bounds the requested output length. It does not guarantee that input plus output fit a given model's context window.
- Custom single-elimination opening matchups are web-only; CLI `exec` has no opening-matchup flag.

## Development

    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test