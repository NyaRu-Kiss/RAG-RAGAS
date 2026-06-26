# RAG Evaluation

Evaluate a RAG (Retrieval Augmented Generation) system with custom metrics

## Local Offline Evaluation For This Repo

This repo also includes a local offline evaluation pipeline for the main RAG app.
It is designed for cases like HotpotQA, where you already have a raw local `json` or `jsonl`
file and want to turn it into:

- a corpus that the RAG app can ingest
- an eval dataset that `ragas` can score
- a report directory with per-sample answers and aggregate metrics

### Simple Principle

The pipeline does not send the dataset `answer` to the RAG app.
It works in four steps:

1. Read local raw samples such as HotpotQA `fullwiki/validation`.
2. Extract each sample's `question`, `answer`, `supporting_facts`, and `context`.
3. Convert `context` into local `.txt` files for ingestion, and convert each sample into the repo's eval `jsonl` format.
4. Rebuild an isolated vector index, ask the RAG app with only `question`, then let `ragas` score the generated `response` against `reference` and `reference_contexts`.

So the responsibility split is:

- RAG app input: `question`
- RAG app retrieval source: generated local corpus from `context`
- Ragas scoring reference: `answer` and `supporting_facts` derived fields

### Files Involved

- `eval/prepare.py`
  Converts raw local HotpotQA-style samples into:
  - corpus text files under `data/eval_uploads/...`
  - eval dataset `jsonl` under `eval/datasets/...`
- `eval/cli.py`
  Provides the one-command evaluation entrypoint.
- `eval/runner.py`
  Calls the RAG app, collects responses, and runs `ragas`.

### Run With Local Samples

Use the project virtualenv. Do not use bare `python3`, or you may hit a different dependency set.

Prepare only:

```bash
. .venv/bin/activate
python -m eval.prepare \
  --input eval/datasets/hotpotqa_fullwiki_validation_15.json \
  --limit 5
```

This generates:

- `eval/datasets/hotpotqa_fullwiki_validation_15.jsonl`
- `data/eval_uploads/hotpotqa_fullwiki_validation_15/`

Run the full offline pipeline:

```bash
. .venv/bin/activate
python -m eval.cli run-hotpotqa-local \
  --input eval/datasets/hotpotqa_fullwiki_validation_15.json \
  --limit 5
```

This command does all of the following:

- prepares local corpus files from the raw dataset
- prepares the eval `jsonl`
- rebuilds an isolated pgvector table
- asks the RAG app for each question
- runs `ragas`
- writes a report under `eval/reports/<run_id>/`

### Output Files

After a run, check:

- `eval/reports/<run_id>/summary.json`
  Aggregate metrics for the run.
- `eval/reports/<run_id>/summary.md`
  Human-readable summary.
- `eval/reports/<run_id>/samples.jsonl`
  Per-sample records including:
  - `user_input`
  - `reference`
  - `response`
  - `retrieved_contexts`
  - `citations`
  - `metric_scores`
- `eval/reports/<run_id>/failures.json`
  Failed samples, if any.

Current `metric_scores` only keeps numeric metrics to avoid redundant fields.

### Metrics Used In This Repo

The current offline pipeline uses these `ragas` metrics:

- `context_precision`
- `answer_relevancy`
- `faithfulness`

Interpretation:

- `context_precision`
  Whether retrieved contexts are actually relevant.
- `answer_relevancy`
  Whether the answer addresses the question.
- `faithfulness`
  Whether the answer stays grounded in the retrieved evidence.

For this repo's current single-turn text RAG flow, these three metrics are the primary ones to watch.

## Quick Start

### 1. Set Your API Key

Choose your LLM provider:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-openai-key"

# Or use Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Run the Evaluation

Using `uv`:

```bash
uv run python evals.py
```

Or using `pip`:

```bash
python evals.py
```

## Project Structure

```
rag_eval/
├── README.md           # This file
├── pyproject.toml      # Project configuration
├── rag.py              # Your RAG application code
├── evals.py            # Evaluation workflow
├── __init__.py         # Makes this a Python package
└── evals/              # Evaluation-related data
    ├── datasets/       # Test datasets
    ├── experiments/    # Experiment results
    └── logs/           # Evaluation logs and traces
```

## Customization

### Modify the LLM Provider

In `evals.py`, update the LLM configuration:

```python
from ragas.llms import llm_factory

# Use Anthropic Claude
llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic")

# Use Google Gemini
llm = llm_factory("gemini-1.5-pro", provider="google")

# Use local Ollama
llm = llm_factory("mistral", provider="ollama", base_url="http://localhost:11434")
```

### Customize Test Cases

Edit the `load_dataset()` function in `evals.py` to add or modify test cases.

### Change Evaluation Metrics

Update the `my_metric` definition in `evals.py` to use different grading criteria.

## Documentation

Visit https://docs.ragas.io for more information.
