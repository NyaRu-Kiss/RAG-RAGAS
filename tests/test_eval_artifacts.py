from pathlib import Path

import pytest

from eval.artifacts import (
    canonical_json_sha256,
    file_sha256,
    safe_config_snapshot,
    source_sample_ids_sha256,
    validate_eval_table_name,
)


def test_hashes_are_deterministic_for_equivalent_json(tmp_path: Path) -> None:
    first = {"items": ["a", "b"], "name": "eval"}
    second = {"name": "eval", "items": ["a", "b"]}
    file_path = tmp_path / "payload.txt"
    file_path.write_text("payload", encoding="utf-8")

    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert file_sha256(file_path) == file_sha256(file_path)
    assert source_sample_ids_sha256(["sample-1", "sample-2"]) != source_sample_ids_sha256(
        ["sample-2", "sample-1"]
    )


def test_safe_config_snapshot_removes_sensitive_values() -> None:
    snapshot = safe_config_snapshot(
        {
            "model": "judge-v1",
            "api_key": "secret-key",
            "nested": {"authorization": "Bearer secret", "timeout": 30},
            "tokens": ["safe", {"password": "hidden"}],
        }
    )

    assert snapshot == {
        "model": "judge-v1",
        "nested": {"timeout": 30},
        "tokens": ["safe", {}],
    }
    assert "secret" not in str(snapshot)


@pytest.mark.parametrize(
    "table_name",
    ["eval_hotpotqa", "eval_1", "eval_a_b_2"],
)
def test_validate_eval_table_name_accepts_only_eval_tables(table_name: str) -> None:
    assert validate_eval_table_name(table_name) == table_name


@pytest.mark.parametrize(
    "table_name",
    ["", "rag_eval_hotpotqa", "eval-unsafe", "eval.Schema", "eval_Upper", 'eval_"quoted'],
)
def test_validate_eval_table_name_rejects_unsafe_names(table_name: str) -> None:
    with pytest.raises(ValueError, match="eval_"):
        validate_eval_table_name(table_name)
