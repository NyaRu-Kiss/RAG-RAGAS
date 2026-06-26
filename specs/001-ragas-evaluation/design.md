# Ragas Evaluation System Design

## Summary

This design introduces an offline evaluation subsystem for the existing LlamaIndex RAG application using `ragas`.

The subsystem is designed around four principles:

- offline and reproducible
- decoupled from the app request path
- explicit separation between retrieval and generation evaluation
- friendly to future dataset growth and experiment comparison

## References

- Ragas positions itself as an experiments-first evaluation library with datasets, metrics, and test data generation capabilities.
- Ragas supports evaluation datasets built from `SingleTurnSample` values grouped in an `EvaluationDataset`.
- Ragas exposes RAG metrics including `Context Precision`, `Context Recall`, `Response Relevancy`, and `Faithfulness`.
- Ragas documents LlamaIndex integration as an evaluation path for `QueryEngine`.

## Current Project Context

The current project:

- uses LlamaIndex for retrieval and response generation
- stores embeddings in Postgres/pgvector
- exposes a `RagService.chat()` path that returns the final answer plus citation metadata
- does not currently expose a formal evaluation runner or dataset abstraction

The current implementation is sufficient to bootstrap offline eval, but it likely needs a narrow adapter to expose retrieved contexts in a format suitable for `ragas`.

## High-Level Architecture

```text
eval dataset -> eval runner -> rag adapter -> ragas dataset builder -> ragas evaluate()
                                                    |
                                                    v
                                            raw sample artifacts
                                                    |
                                                    v
                                           report + baseline diff
```

## Proposed Directory Layout

```text
eval/
  datasets/
    rag_eval_v1.jsonl
  baselines/
  reports/
  templates/
    sample.schema.json
  __init__.py
  config.py
  dataset.py
  adapter.py
  metrics.py
  runner.py
  reporting.py
  cli.py
```

## Core Components

### 1. Dataset Layer

Purpose:

- load and validate curated evaluation samples
- normalize samples into internal Python objects
- convert samples into `ragas`-compatible data structures

Input format:

- repository-local `jsonl` for human editing and version control

Internal representation:

- project-defined dataclass or Pydantic model

Output representation:

- `ragas` `SingleTurnSample`
- `ragas` `EvaluationDataset`

Recommended sample schema:

```json
{
  "id": "resume_fact_001",
  "user_input": "How many years of .NET experience does the candidate have?",
  "reference": "6 years",
  "reference_contexts": [
    "6 years .NET software engineer"
  ],
  "tags": ["resume", "factoid"],
  "difficulty": "easy",
  "question_type": "fact"
}
```

Design choice:

- `jsonl` is preferred over YAML because it is simple to stream, diff, and append.

### 2. RAG Adapter Layer

Purpose:

- invoke the existing RAG system programmatically
- capture both answer and retrieval evidence
- adapt application outputs into a structure that `ragas` can score

Primary responsibility:

- call `RagService.chat()`
- extract `response`
- extract `citations`
- if needed, extend the service or add a dedicated evaluation method to expose raw retrieved contexts

Output contract per evaluated sample:

- `response`
- `retrieved_contexts`
- `retrieved_source_ids`
- `citations`
- optional timing and error metadata

Important note:

`ragas` RAG metrics depend on retrieved contexts. Citation snippets may be sufficient for a first version, but raw retrieved text chunks are preferred because they better represent actual model evidence.

### 3. Metric Configuration Layer

Purpose:

- define the initial metric bundle
- isolate `ragas` metric imports and provider configuration

Initial metric bundle:

- retrieval:
  - `context_precision`
  - `context_recall`
- generation:
  - `response_relevancy`
- groundedness:
  - `faithfulness`

Optional later additions:

- `noise_sensitivity`
- `factual_correctness`
- rubric-based custom metrics for domain-specific answer requirements

Design choice:

- keep metric configuration in one module so future changes do not affect the runner contract

### 4. Evaluation Runner

Purpose:

- orchestrate one full experiment run
- execute the RAG system against dataset samples
- build a `ragas` evaluation dataset
- call `ragas.evaluate()`
- persist raw outputs and summaries

Responsibilities:

- filter dataset by tags or limit
- handle per-sample failures
- assemble run manifest
- coordinate result persistence

Runner phases:

1. Load dataset
2. Filter samples
3. Execute RAG app for each sample
4. Build `SingleTurnSample` rows with:
   - `user_input`
   - `response`
   - `retrieved_contexts`
   - `reference`
   - `reference_contexts` when available
5. Build `EvaluationDataset`
6. Call `ragas.evaluate()`
7. Persist outputs
8. Optionally compute baseline diff

### 5. Reporting Layer

Purpose:

- make results consumable outside Python

Outputs:

- `manifest.json`
- `samples.jsonl`
- `summary.json`
- `summary.md`
- optional `baseline_diff.json`

Aggregate summary fields:

- run id
- dataset path
- selected metrics
- sample count
- failed sample count
- mean metric scores
- baseline deltas when available

Per-sample fields:

- sample id
- tags
- question
- reference
- model response
- retrieved contexts
- citations
- metric scores
- execution errors

## Data Flow

### Input Data

Curated evaluation sample:

```text
question + reference answer + optional reference contexts + metadata
```

### Runtime Data

Application output:

```text
question -> RAG service -> answer + retrieved evidence
```

### Ragas Input Shape

Preferred `SingleTurnSample` fields:

- `user_input`
- `response`
- `retrieved_contexts`
- `reference`
- `reference_contexts`

This aligns with `ragas` single-turn evaluation semantics.

## CLI Design

Proposed command:

```bash
python -m eval.cli run --dataset eval/datasets/rag_eval_v1.jsonl
```

Supported flags in v1:

- `--tags resume,factoid`
- `--limit 10`
- `--baseline <run_id_or_path>`
- `--output-dir <path>`

Future flags:

- `--in-ci`
- `--metric-profile`
- `--judge-model`

## Config Design

Separate app config from eval config.

Proposed eval config fields:

- `eval_dataset_path`
- `eval_output_dir`
- `eval_judge_model`
- `eval_judge_base_url`
- `eval_judge_api_key`
- `eval_embeddings_model`
- `eval_timeout_seconds`
- `eval_batch_size`
- `eval_raise_exceptions`

Design choice:

- do not overload the app's existing RAG config with eval-only settings

## Baseline Comparison Design

Baseline comparison is file-based in v1.

Flow:

1. Load prior `summary.json`
2. Compare aggregate metrics
3. Compare overlapping sample ids
4. Emit diff artifact

Delta types:

- metric improved
- metric regressed
- sample newly failed
- sample no longer failed

## Error Handling

Runner default behavior:

- continue past individual sample failures
- record exception type and message
- exclude failed rows from `ragas` scoring if necessary
- count and surface failures in summary

Configurable strict mode later:

- fail the entire run on first error

## Evolution Path

### V1

- curated single-turn dataset
- offline runner
- `ragas` core metrics
- file-based reports

### V2

- synthetic dataset generation via `ragas` testset generation
- experiment tracking backend
- richer domain-specific rubric metrics
- CI regression gates

### V3

- multi-turn eval
- production sample import
- dashboarding

## Risks

### Risk 1: Retrieved contexts are not exposed cleanly

Impact:

- `ragas` retrieval and faithfulness metrics become less trustworthy

Mitigation:

- add an evaluation-only adapter path that exposes raw source-node content before final answer synthesis

### Risk 2: Judge model instability

Impact:

- run-to-run variance in LLM-based metrics

Mitigation:

- capture judge model config in manifest
- start with deterministic or low-temperature settings
- preserve baseline slices for manual review

### Risk 3: Weak gold dataset quality

Impact:

- misleading aggregate scores

Mitigation:

- start with a small manually reviewed dataset
- require sample metadata and references
- expand only after spot-checking

## Design Decisions

### DD-1 Use curated gold data first

Reason:

- this gives a reliable benchmark before synthetic expansion

### DD-2 Keep evaluation offline

Reason:

- eval workload is slower, costlier, and operationally different from user chat

### DD-3 Use file-based artifacts in v1

Reason:

- simplest reproducible starting point

### DD-4 Separate app and eval configuration

Reason:

- reduces coupling and avoids accidental impact on serving behavior
