# Ragas Evaluation System Requirements

## Summary

Build a mature, offline-first evaluation system for the current LlamaIndex-based RAG application using `ragas`.

The evaluation system must let the team:

- measure retrieval and answer quality consistently
- compare runs across prompt, retrieval, chunking, and model changes
- inspect per-sample failures instead of relying on aggregate scores alone
- use the evaluation loop as the primary gate for future RAG iteration

This spec follows an SDD flow and defines only the evaluation system, not retrieval improvements.

## Problem Statement

The current project can answer questions over uploaded documents, but it does not have a formal evaluation workflow. Without systematic evaluation:

- changes cannot be compared reliably
- regressions are easy to miss
- retrieval failures and generation failures are mixed together
- optimization work is guided by intuition instead of evidence

## Goals

- Introduce a reproducible offline evaluation pipeline based on `ragas`
- Support both curated human-written samples and future synthetic testset expansion
- Score retrieval quality, answer quality, and groundedness separately
- Persist run artifacts for comparison and audit
- Keep the evaluation system decoupled from the web UI and request path

## Non-Goals

- Real-time online evaluation during user chat
- Automatic prompt optimization in the first version
- CI gating in the first version
- Multi-turn conversation evaluation in the first version
- Rebuilding the application around a different framework

## Users

### Primary Users

- Developers iterating on retrieval, prompt, and model configuration
- Maintainers reviewing whether a change improves the RAG system

### Secondary Users

- Future automation or CI jobs that run regression checks

## Scope

### In Scope

- Evaluation dataset format and storage
- Offline execution runner
- `ragas` metric selection and configuration
- Run result persistence
- Human-readable evaluation reports
- Baseline comparison support

### Out of Scope

- Frontend visualization dashboard
- Production telemetry ingestion
- Automatic dataset labeling from live traffic

## Functional Requirements

### FR-1 Dataset Management

The system shall provide a versioned offline dataset for evaluation samples.

Each sample shall support at minimum:

- `id`
- `user_input`
- `reference`
- `reference_contexts` when available
- `tags`
- `difficulty`
- `question_type`

The dataset format shall be stable and editable by humans in the repository.

### FR-2 Single-Turn RAG Evaluation

The first version shall evaluate single-turn RAG behavior only.

For each sample, the runner shall execute the project RAG pipeline and capture:

- user question
- model answer
- retrieved contexts
- retrieved source metadata
- reference answer

### FR-3 Ragas Metrics

The system shall support `ragas` metrics appropriate for single-turn RAG evaluation.

Initial required metric groups:

- retrieval quality
- answer relevance/correctness
- groundedness/faithfulness

The exact initial metric set shall be implementation-defined, but the design must support at least these `ragas`-style capabilities:

- context-level retrieval evaluation
- response-level answer evaluation
- hallucination or groundedness evaluation

### FR-4 Run Reproducibility

Every evaluation run shall persist a run manifest including:

- timestamp
- dataset version or file path
- evaluated sample count
- application config snapshot relevant to RAG behavior
- evaluation model config relevant to `ragas`
- selected metrics

### FR-5 Result Persistence

Every run shall persist:

- per-sample raw outputs
- per-sample metric scores
- aggregate metric scores
- execution failures or skipped samples

Results shall be saved to a repository-local results directory.

### FR-6 Baseline Comparison

The system shall support comparing a current run against a prior baseline run.

Comparison output shall include:

- metric deltas
- sample count differences
- newly failed samples
- newly improved samples when determinable

### FR-7 Failure Inspection

The system shall make it easy to inspect failure cases.

Each sample result shall include enough data to answer:

- what question was asked
- what answer the system produced
- what contexts were retrieved
- which sources were cited
- which metrics failed

### FR-8 Execution Modes

The system shall support at least:

- full dataset evaluation
- filtered evaluation by tag
- limited evaluation by sample count for local smoke runs

### FR-9 Synthetic Expansion Readiness

The first version does not need to generate synthetic datasets automatically, but the architecture shall leave room for future `ragas` testset generation.

## Quality Requirements

### QR-1 Separation of Concerns

Evaluation code shall be isolated from serving code.

### QR-2 Reproducibility

Given the same dataset, config, and dependencies, the system should make repeatable runs practical. Any non-deterministic judge behavior must be captured in run metadata.

### QR-3 Inspectability

Outputs shall be readable without opening Python objects manually.

### QR-4 Extensibility

It shall be straightforward to add:

- new metrics
- new datasets
- new report formats
- future multi-turn evaluation

### QR-5 Failure Tolerance

Individual sample failures shall not invalidate the entire run by default. Failures should be recorded and surfaced.

## Constraints

- The current application uses LlamaIndex and returns answer plus citations from the service layer.
- The evaluation system must integrate with the existing Python project layout.
- The first version should avoid introducing a database dependency for evaluation storage.
- The first version should rely on official `ragas` dataset and metric abstractions where practical.

## Assumptions

- The current RAG service can be invoked programmatically from Python without the web UI.
- Retrieved contexts can be extracted or adapted from the existing citation and source-node flow.
- The team is willing to curate an initial gold dataset manually before adding synthetic generation.

## Acceptance Criteria

### AC-1

A developer can run one command locally and evaluate the existing RAG system over a repository dataset.

### AC-2

The run produces a persisted artifact directory containing:

- manifest
- per-sample results
- aggregate summary

### AC-3

The initial evaluation pipeline reports at least one retrieval metric, one answer metric, and one groundedness metric using `ragas`.

### AC-4

A developer can compare the latest run against a baseline run and inspect deltas.

### AC-5

A developer can restrict evaluation to a subset of samples by tag or limit.

## Open Questions

- Which evaluation LLM should be used for `ragas` judging: DeepSeek via OpenAI-compatible API, OpenAI, or another provider?
- Should baseline comparison be file-based only in v1, or also tracked through an experiment store later?
- Do we want to require `reference_contexts` for all gold samples, or allow partial datasets initially?
