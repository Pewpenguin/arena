# Arena

Model-vs-model evaluation for LLMs.

Arena runs the same tasks against multiple models, supports exact-match evaluation, and can compare model outputs using an LLM judge.

The core uses a `ModelProvider` trait. DeepInfra is the currently implemented provider.

## Features

- Run a single model against a prompt
- Run multiple models against the same task set
- Concurrent candidate and judge requests with bounded provider concurrency
- Exact-match evaluation
- Pairwise LLM judging with both response orders
- Agreement tracking for position-sensitive judgments
- Win/loss/draw statistics
- Elo ratings
- Request duration and run provenance
- JSON output

## Setup

Requires Rust and Cargo.

Set a DeepInfra API token in `.env`:

    cp .env.example .env

Then set:

    DEEPINFRA_TOKEN=your_token

Build the project:

    cargo build

Model IDs are DeepInfra model names.

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

`--model` can be specified multiple times. Model IDs must be unique. `--judge` and `--output` are optional.

Without `--output`, the JSON result is written to stdout. A provider error fails the run.

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

Only the resolved judgment is persisted and used for statistics and Elo.

## Output

`arena exec` produces one JSON object with these fields:

- `run` — version, models, judge, task path, and start time
- `results` — model responses, evaluations, and durations
- `comparisons` — pairwise comparisons from exact scores
- `judgments` — resolved LLM-judge results
- `statistics` — per-model wins, losses, draws, and judge agreement
- `ratings` — Elo ratings calculated from resolved judgments

`results[].duration_ms` is the duration of that candidate's provider request after it has a permit.

`judgments[].duration_ms` is the wall-clock time from when the first of a pair's two judge orientations begins work after acquiring a provider permit until both orientations have completed. It excludes time the pair spends queued behind other provider calls, and it is not the sum of the two orientation request times.

Results retain task-file and CLI model order. Statistics and ratings are ordered by model ID.

## Limitations

- LLM judgments are not ground truth.
- The judge can also be one of the candidate models.
- Running both response orders can reveal orientation disagreement, but does not establish that a judge is unbiased.
- An orientation disagreement is treated as a draw for statistics and Elo.
- Elo depends on the order in which judgments are processed.
- DeepInfra is the only implemented provider.
- Exact-score comparisons are not used by the Elo calculation.

## Development

    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test