import json
from pathlib import Path

import pytest

from eval.dataset import EmptyDatasetError, EvalSample, filter_dataset, load_dataset


def test_load_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_load_dataset_raises_for_empty_file(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("\n", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="contains no samples"):
        load_dataset(dataset)


def test_load_dataset_rejects_duplicate_source_sample_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "duplicate.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(payload)
            for payload in [
                {"id": "local-1", "source_sample_id": "source-1", "user_input": "q1", "reference": "a1"},
                {"id": "local-2", "source_sample_id": "source-1", "user_input": "q2", "reference": "a2"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate source_sample_id: source-1"):
        load_dataset(dataset)


def test_eval_sample_uses_id_as_legacy_source_sample_id() -> None:
    sample = EvalSample(id="legacy-1", user_input="question", reference="answer")

    assert sample.source_sample_id == "legacy-1"


def test_filter_dataset_by_tag_and_limit() -> None:
    samples = [
        EvalSample(id="1", user_input="q1", reference="a1", tags=["resume"]),
        EvalSample(id="2", user_input="q2", reference="a2", tags=["paper"]),
        EvalSample(id="3", user_input="q3", reference="a3", tags=["resume", "paper"]),
    ]

    filtered = filter_dataset(samples, tags=["resume"], limit=1)

    assert [sample.id for sample in filtered] == ["1"]
