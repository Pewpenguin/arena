# Arena

Model-vs-model evaluation for LLMs.

Arena runs the same tasks against multiple models, supports exact-match evaluation, and can compare model outputs using an LLM judge.

The core uses a `ModelProvider` trait. Arena uses an OpenAI-compatible API endpoint.

## Features

- Run a single model against a prompt
- Run multiple models against the same task set
- Concurrent candidate and judge requests with bounded provider concurrency
- Exact-match evaluation
- Pairwise LLM judging with both response orders
- Agreement tracking for orientation disagreement
- Win/loss/draw statistics
- Bradley–Terry ratings with task-clustered bootstrap uncertainty
- Request duration and run provenance
- JSON output

## Setup

Requires Rust and Cargo.

Arena uses an OpenAI-compatible API endpoint. Set the API key in `.env`:

    cp .env.example .env

Then set:

    ARENA_API_KEY=your_api_key

`ARENA_BASE_URL` is optional and defaults to `https://api.openai.com/v1`. Set `ARENA_BASE_URL` to any service exposing the OpenAI-compatible API subset Arena requires. Use that service's API key through `ARENA_API_KEY`.

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
      --output results.json

`--model` can be specified multiple times. Model IDs must be unique. `--judge` must not match any `--model`. `--judge` and `--output` are optional. `--seed` sets the bootstrap RNG seed and defaults to `0`.

Without `--output`, the JSON result is written to stdout. Each candidate provider request is tried up to three times (a 1s backoff, then 2s) for provider errors and invalid provider responses. An exhausted candidate failure still aborts the run without producing output. A permanently failed judge pair is saved in the output with `run.complete` set to false, and the process then exits non-zero.

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

Exact evaluation and LLM judging are separate. Exact scores produce `comparisons`. `--judge` always produces `judgments`, `judgment_failures`, `statistics`, `ratings`, and pair-coverage metadata, including when those collections are empty. Those fields are omitted only when `--judge` is not used.

## Judging

For each model pair, Arena runs the judge twice:

    (A, B)
    (B, A)

The second result is mapped back to the original model order.

If both orientations agree, the judgment is kept with `agreement: true`.

If they disagree, Arena resolves the pair as a draw with `agreement: false`. Orientation disagreement is not automatically evidence of position bias: it can also come from remaining judge noise, formatting, or other order effects.

Both orientation winners, reasons, and raw judge completions are persisted, mapped back to the original model pair. Statistics and Bradley–Terry ratings use only the resolved winner. The two orientations are retained for diagnostics and are not counted as independent ranking observations.

Each orientation is tried up to three times (a 1s backoff, then 2s) for provider errors and malformed/truncated JSON. The parser stays strict: the entire judge completion must be one JSON object in the required schema. Surrounding text, extra objects, or JSON copied from a candidate reply are rejected rather than scanned for a last match. A retry repeats the same judge request rather than repairing invalid output. Candidate replies are isolated in `<response_a>` / `<response_b>` tags in the judge prompt; `<` in that quoted text is escaped so it cannot close the fence.

Judge requests set `temperature` to `0`. That is persisted as `run.judge_decoding` and applies only to judge calls, not candidate generation. Temperature 0 asks the provider for deterministic decoding; it does not guarantee identical completions across providers or models.

Both orientations must succeed to produce a resolved judgment. A permanently failed pair is omitted from `judgments` (it is not a draw) and recorded in `judgment_failures`. The run still writes its output, then exits non-zero.

Coverage is counted in unordered pairs, not orientations. `expected_pairs` is the number of unordered candidate pairs across the loaded tasks. `resolved_pairs` is the number of persisted resolved judgments. `failed_pairs` is the number of permanently failed pairs. For a finished judge run, `resolved_pairs + failed_pairs == expected_pairs`. A mismatch is reported as an error rather than rewritten in the JSON.

`run.complete` is `true` only when `resolved_pairs == expected_pairs` and `failed_pairs` is 0. That includes a complete run with zero expected pairs (for example a single candidate). It is `false` when any expected pair failed, including when every expected pair failed and `judgments` is an empty array. Missing judgments can disconnect or separate the comparison graph, and the gaps need not be random. Statistics and ratings from an incomplete run describe only the observed pairs and should not be read as if every pair had been observed.

`run.orientation_agreement` counts each resolved unordered pair once: `orientation_agreeing_pairs`, `orientation_disagreeing_pairs`, and `agreement_rate` (`orientation_agreeing_pairs / resolved_pairs`, or `0.0` when there are no resolved pairs). That summary is the pair-level agreement metric. Per-model `agreement_count` / `disagreement_count` attribute the same pair to both endpoints; they are not additional independent observations, and they are not a position-bias estimate.

## Output

`arena exec` produces one JSON object with these fields:

- `run` — version, models, judge, task path, resolved provider `base_url`, start time, provider concurrency, request/connect timeouts, and candidate/judge attempt counts. Judge-only fields below are omitted when `--judge` is not used.
- `tasks` — the loaded task definitions (id, prompt, optional exact evaluation)
- `results` — model responses, evaluations, and durations
- `comparisons` — pairwise comparisons from exact scores
- `judgments` — resolved LLM-judge results, both orientation winners, both orientation reasons, and both raw judge completions
- `judgment_failures` — pairwise judge attempts that never produced both orientations
- `statistics` — per-model wins, losses, draws, and per-model orientation-agreement counts among that model's pairs
- `ratings` — one record per requested candidate: a full-data Bradley–Terry point estimate from resolved judgments, on a 400-point scale centered at 1500, optional 95% percentile bounds, or `rating: null` with an `unavailable` reason

When `--judge` is used, `judgments`, `judgment_failures`, `statistics`, and `ratings` are always present. Empty arrays mean the judge ran and that collection has no entries. Absence of those keys means no judge was configured. `results` and `comparisons` keep their existing schema.

Judge-run `run` fields:

- `expected_pairs`, `resolved_pairs`, `failed_pairs` — unordered-pair coverage. `complete` is true only when `resolved_pairs == expected_pairs` and `failed_pairs == 0`.
- `orientation_agreement` — pair-level orientation agreement: `resolved_pairs`, `orientation_agreeing_pairs`, `orientation_disagreeing_pairs`, `agreement_rate`. Each unordered resolved pair is counted once. `agreement_rate` is agreeing / resolved, or `0.0` when `resolved_pairs` is 0. This is not a measure of position bias.
- `judge_decoding` — judge request decoding (temperature 0)
- `bootstrap_seed`, `bootstrap_replicates`, `bootstrap_clusters`, `bootstrap_ran` — bootstrap request and whether resampling ran
- `bootstrap_valid` — finite replicate count; present only when the resampling loop ran
- `bootstrap_unavailable` — why interval bounds were not produced (`too_few_tasks`, `original_unrated`, or `invalid_replicates`)

The process still exits non-zero on incomplete judging; the JSON is the record of completeness, not the exit code. Requested candidates remain in `ratings` even with no resolved judgments: `rating` is `null`, bounds are omitted, and `unavailable` is set (typically `no_comparisons`).

`results[].duration_ms` is the total candidate execution duration after it acquires a provider permit, including retries and retry backoff.

`judgments[].duration_ms` is the wall-clock time from when the first of a pair's two judge orientations begins work after acquiring a provider permit until both orientations have completed. It excludes time the pair spends queued behind other provider calls, and it is not the sum of the two orientation request times.

Results retain task-file and CLI model order. Statistics and ratings are ordered by model ID.

The saved run is an audit of one execution: models, judge, task file path, the loaded tasks, the provider base URL, Arena-controlled concurrency, timeout, and retry-attempt settings, and, when a judge is used, pairwise coverage, pair-level orientation agreement, the judge decoding configuration, rating availability, and bootstrap metadata. It does not store API keys. Candidate completions use the provider default decoding and are not deterministic. Judge calls request `temperature` 0, but providers and models may still vary. `--seed` only controls the bootstrap RNG and does not make model completions deterministic. When intervals were produced, the same seed reproduces them from the persisted judgments. Raw judge completions are kept so a later audit can see what was parsed.

## Ratings

Ratings are relative Bradley–Terry strengths fitted to resolved pairwise judgments on this run's observed pairs. They are not a general-purpose leaderboard and are not comparable across different task files, judges, or missingness patterns.

Draws enter the likelihood as half-wins: a draw between A and B contributes 0.5 to each model's win total. Arena does not fit a separate draw parameter. A finite maximum-likelihood rating exists only when the directed graph with an edge `i → j` whenever `i` has a positive win mass against `j` (including 0.5 from a draw) is strongly connected, and the comparison graph is a single comparable component. Isolated models are `no_comparisons`. Two or more comparable components are `disconnected` and are not placed on one scale. Complete separation is `separated`. If the Hunter MM iteration does not meet tolerance, the reason is `nonconvergence`. If it meets tolerance but the resulting strengths or Elo-like transform are not finite and positive, the reason is `nonfinite_result`. Unavailable models have `rating: null` and that reason; they are not assigned 1500.

The 1500 center and 400-point scale are a convention on this relative log-strength scale: 400 points is a 10× strength ratio. Exact-score `comparisons` are not used.

Uncertainty, when present, is a task-clustered percentile bootstrap of that same estimator. Each unique `task_id` among resolved judgments is one cluster; resampling a cluster carries all of that task's resolved pairs together. The two judge orientations are not independent observations and are not resampled. The default is 1,000 replicates, seed `0`, and 95% Hyndman–Fan type-7 percentile bounds (`rating_lower`, `rating_upper`). `bootstrap_clusters` is the number of those task IDs, not the number of rows in the task file. Point estimates always come from the full observed sample; bounds are additional.

Bounds are omitted when they cannot be computed. `bootstrap_ran` is whether the resampling loop executed. `bootstrap_valid` is the number of finite replicates and is omitted when the loop did not run (it is not stored as 0 to mean "skipped"). `bootstrap_unavailable` is one of:

- `too_few_tasks` — fewer than two task clusters among resolved judgments
- `original_unrated` — at least one requested model has no finite full-data rating
- `invalid_replicates` — the loop ran, but at least one replicate did not produce finite ratings for every model

The interval is task-sampling variability of this observed-judgment estimator. It does not measure judge noise, residual position effects, task-selection bias, or performance on a different task distribution. It is not a test of pairwise rating differences.

## Limitations

- LLM judgments are not ground truth.
- Running both response orders can reveal orientation disagreement, but does not establish that a judge is unbiased, and disagreement is not automatically evidence of position bias.
- An orientation disagreement is treated as a draw for statistics and ratings.
- A missing orientation is omitted, not treated as a draw.
- Statistics and ratings from an incomplete judge run describe only the observed pairs.
- Exact-score comparisons are not used by the Bradley–Terry calculation.
- Ratings from an incomplete comparison graph, or with `unavailable` set, are not a complete ranking.
- Bootstrap intervals measure task-sampling variability of the observed-pair estimator only. They do not account for judge noise, position effects, or task-selection bias, and they are not a test of pairwise rating differences.
- Judge `temperature` 0 does not guarantee identical outputs across providers or models.

## Development

    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test