import json
from pathlib import Path

import pytest

from eval.artifacts import file_sha256
from eval.config import EvalSettings
from eval.runner import preflight_generation


def _settings(tmp_path: Path) -> EvalSettings:
    return EvalSettings(
        EVAL_DATASET_PATH=str(tmp_path / "generation.jsonl"),
        EVAL_OUTPUT_DIR=str(tmp_path / "reports"),
        EVAL_BASELINE_DIR=str(tmp_path / "baselines"),
        EVAL_JUDGE_PROVIDER="openai_compatible",
        EVAL_JUDGE_API_KEY="test-key",
        EVAL_JUDGE_BASE_URL="https://judge.example/v1",
        EVAL_JUDGE_MODEL="judge-v1",
    )


def test_preflight_rejects_non_fixed_generation_set_before_external_checks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    dataset = Path(settings.eval_dataset_path)
    dataset.write_text(json.dumps({"id": "1", "user_input": "q", "reference": "a"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 20"):
        preflight_generation(settings, dataset_path=dataset)


def test_preflight_rejects_index_state_with_wrong_corpus_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    dataset = Path(settings.eval_dataset_path)
    rows = [
        {"id": f"id-{index}", "source_sample_id": str(index), "user_input": "q", "reference": "a"}
        for index in range(20)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"p"}\n', encoding="utf-8")
    tables = {"pg_table": "eval_vector", "docstore_table": "eval_docstore", "file_index_table": "eval_file_index"}
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"corpus_sha256": file_sha256(corpus), "generation_source_sample_ids": [str(index) for index in range(20)], "eval_tables": tables}),
        encoding="utf-8",
    )
    (tmp_path / "index_state.json").write_text(json.dumps({"corpus_sha256": "sha256:wrong"}), encoding="utf-8")
    monkeypatch.setattr("eval.runner._verify_local_embedding", lambda: {"model": "local", "config_sha256": "sha256:x"})
    monkeypatch.setattr("eval.runner.build_metric_specs", lambda **_: [])

    with pytest.raises(ValueError, match="index state"):
        preflight_generation(settings, dataset_path=dataset)
