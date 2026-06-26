# Ragas Evaluation System Tasks

## Summary

This task list implements the SDD requirements and design for a `ragas`-based evaluation system for the current LlamaIndex RAG app.

## Phase 1: Foundation

### T1 Create evaluation module layout

- add `eval/` package and base module files
- add `eval/datasets/`, `eval/reports/`, and `eval/baselines/`
- add a sample schema or example dataset file

### T2 Add dependencies and configuration surface

- add `ragas` dependency
- add any required adapter dependencies for the selected judge/embedding providers
- define eval-specific settings separate from app runtime settings

### T3 Define dataset schema

- implement sample model for curated single-turn evaluation
- support load and validation from `jsonl`
- validate required fields and metadata

## Phase 2: RAG Integration

### T4 Build RAG evaluation adapter

- add a programmatic adapter around the current RAG service
- capture response text
- capture retrieved contexts
- capture citations and source metadata
- normalize failures into structured result objects

### T5 Expose retrieval evidence cleanly

- confirm whether current citation snippets are sufficient
- if not, add an evaluation-only path to expose raw source-node text from retrieval

## Phase 3: Ragas Evaluation Pipeline

### T6 Implement `ragas` dataset conversion

- convert evaluated rows into `SingleTurnSample`
- build `EvaluationDataset`
- map project fields to `ragas` expected fields

### T7 Implement metric registry

- create initial metric profile for:
  - `context_precision`
  - `context_recall`
  - `response_relevancy`
  - `faithfulness`
- support future metric profiles

### T8 Implement evaluation runner

- load dataset
- apply tag and limit filters
- run RAG inference over all selected samples
- call `ragas.evaluate()`
- collect aggregate and per-sample outputs

## Phase 4: Reporting

### T9 Persist run artifacts

- write `manifest.json`
- write per-sample output file
- write aggregate summary file
- write human-readable markdown summary

### T10 Implement baseline comparison

- load a prior run artifact
- compute metric deltas
- identify newly failed and newly improved rows
- persist comparison output

## Phase 5: CLI and Developer Workflow

### T11 Add CLI entrypoint

- support `run`
- support dataset path override
- support `--tags`
- support `--limit`
- support `--baseline`
- support `--output-dir`

### T12 Document developer workflow

- add README section for evaluation
- describe dataset authoring
- describe local run commands
- describe how to interpret reports

## Phase 6: Validation

### T13 Add smoke tests

- dataset parsing test
- adapter contract test
- report generation test

### T14 Add fixture-based evaluation test

- use a mocked RAG adapter or controlled fixture
- verify `ragas` pipeline integration at a minimal level

### T15 Manual validation

- run a limited local eval on a few curated samples
- verify outputs are persisted correctly
- verify baseline diff works

## Suggested Implementation Order

1. T1
2. T2
3. T3
4. T4
5. T5
6. T6
7. T7
8. T8
9. T9
10. T11
11. T10
12. T13
13. T14
14. T12
15. T15

## Definition of Done

- SDD spec is approved
- evaluation package exists
- local command can run offline evaluation against curated dataset
- run artifacts are persisted
- baseline comparison works
- tests cover core data flow
