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
- Agreement tracking for position-sensitive judgments
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

`--model` can be specified multiple times. Model IDs must be unique. `--judge` and `--output` are optional. `--seed` sets the bootstrap RNG seed and defaults to `0`.

Without `--output`, the JSON result is written to stdout. Each candidate provider request is tried up to three times (a 1s backoff, then 2s) for provider errors and invalid provider responses. An exhausted candidate failure still aborts the run without producing output. A permanently failed judge pair is saved in the output and the process then exits non-zero.

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

Exact evaluation and LLM judging are separate. Exact scores produce `comparisons`; `--judge` produces `judgments`, `statistics`, and `ratings`.

## Judging

For each model pair, Arena runs the judge twice:

    (A, B)
    (B, A)

The second result is mapped back to the original model order.

If both orientations agree, the judgment is kept with `agreement: true`.

If they disagree, Arena resolves the pair as a draw with `agreement: false`.

Both orientation winners are persisted, mapped back to the original model pair. Statistics and Bradley–Terry ratings use only the resolved winner. The two orientations are retained for position-bias diagnostics and are not counted as independent ranking observations.

Each orientation is tried up to three times (a 1s backoff, then 2s) for provider errors and malformed/truncated JSON. The parser stays strict: a retry repeats the same judge request rather than repairing invalid output. Both orientations must succeed to produce a resolved judgment. A permanently failed pair is omitted from `judgments` (it is not a draw) and recorded in `judgment_failures`. The run still writes its output, then exits non-zero. Missing judgments can disconnect or separate the comparison graph, and the gaps need not be random, so ratings from an incomplete run should not be read as if every pair had been observed.

## Output

`arena exec` produces one JSON object with these fields:

- `run` — version, models, judge, task path, resolved provider `base_url`, start time, and bootstrap seed/replicate counts when uncertainty was computed
- `tasks` — the loaded task definitions (id, prompt, optional exact evaluation)
- `results` — model responses, evaluations, and durations
- `comparisons` — pairwise comparisons from exact scores
- `judgments` — resolved LLM-judge results and both orientation winners
- `judgment_failures` — pairwise judge attempts that never produced both orientations
- `statistics` — per-model wins, losses, draws, and judge agreement
- `ratings` — full-data Bradley–Terry point estimates from resolved judgments, on a 400-point scale centered at 1500, with optional 95% percentile bounds

Ratings are the full-data Bradley–Terry estimates. Uncertainty uses a task-clustered percentile bootstrap: each unique task is resampled as a unit, carrying all of that task's resolved pairwise judgments. The two judge orientations are not independent observations and are not resampled. The default is 1,000 replicates, a seed of `0`, and 95% percentile bounds (`rating_lower`, `rating_upper`). Bounds are omitted when there are fewer than two distinct tasks, when the original ratings are unavailable, or when any bootstrap replicate cannot produce finite ratings. In those cases the point estimates remain. The intervals describe task-sampling variability only; they do not capture judge bias, residual position bias, task-selection effects, or a different task distribution. The 1500 center is a convention on this relative scale.

`results[].duration_ms` is the duration of that candidate's provider request after it has a permit.

`judgments[].duration_ms` is the wall-clock time from when the first of a pair's two judge orientations begins work after acquiring a provider permit until both orientations have completed. It excludes time the pair spends queued behind other provider calls, and it is not the sum of the two orientation request times.

Results retain task-file and CLI model order. Statistics and ratings are ordered by model ID.

The saved run is an audit of one execution: models, judge, task file path, the loaded tasks, and the provider base URL. It does not store API keys. Candidate and judge completions are not deterministic; `--seed` only reproduces bootstrap intervals from the persisted judgments.

## Limitations

- LLM judgments are not ground truth.
- The judge can also be one of the candidate models.
- Running both response orders can reveal orientation disagreement, but does not establish that a judge is unbiased.
- An orientation disagreement is treated as a draw for statistics and ratings.
- A missing orientation is omitted, not treated as a draw.
- Exact-score comparisons are not used by the Bradley–Terry calculation.
- Bootstrap intervals are not a test of pairwise rating differences.

## Development

    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test