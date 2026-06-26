from pathlib import Path

from eval.dataset import EvalSample, filter_dataset, load_dataset


def test_load_dataset_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_dataset(tmp_path / "missing.jsonl") == []


def test_filter_dataset_by_tag_and_limit() -> None:
    samples = [
        EvalSample(id="1", user_input="q1", reference="a1", tags=["resume"]),
        EvalSample(id="2", user_input="q2", reference="a2", tags=["paper"]),
        EvalSample(id="3", user_input="q3", reference="a3", tags=["resume", "paper"]),
    ]

    filtered = filter_dataset(samples, tags=["resume"], limit=1)

    assert [sample.id for sample in filtered] == ["1"]
