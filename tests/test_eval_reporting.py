import json
from pathlib import Path

import pytest

from eval.reporting import AtomicRunWriter, write_summary_md


def test_write_summary_md(tmp_path: Path) -> None:
    target = tmp_path / "summary.md"
    write_summary_md(
        target,
        {
            "run_id": "run-1",
            "dataset_path": "eval/datasets/rag_eval_v1.jsonl",
            "sample_count": 0,
            "failed_sample_count": 0,
            "metrics": {"faithfulness": 0.8},
            "skipped_metrics": {"multimodal_faithfulness": "not supported"},
        },
    )

    content = target.read_text(encoding="utf-8")
    assert "run-1" in content
    assert "multimodal_faithfulness" in content


def test_atomic_run_writer_only_publishes_complete_runs(tmp_path: Path) -> None:
    writer = AtomicRunWriter(tmp_path)
    writer.write_json("summary.json", {"run_id": writer.run_id})
    writer.write_jsonl("samples.jsonl", [{"id": "sample-1"}])
    writer.write_jsonl("retrieval_traces.jsonl", [{"id": "sample-1", "trace": {}}])
    writer.write_json("failures.json", [])
    writer.write_text("summary.md", "# Summary\n")

    run_dir = writer.complete({"run_id": writer.run_id})

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run_dir.parent == tmp_path
    assert manifest["status"] == "completed"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_atomic_run_writer_preserves_failed_run_without_publishing(tmp_path: Path) -> None:
    writer = AtomicRunWriter(tmp_path)
    writer.write_json("summary.json", {"run_id": writer.run_id})

    failed_dir = writer.fail({"run_id": writer.run_id}, status="failed")

    assert not (tmp_path / writer.run_id).exists()
    assert ".failed-" in failed_dir.name
    manifest = json.loads((failed_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_atomic_run_writer_rejects_incomplete_completion(tmp_path: Path) -> None:
    writer = AtomicRunWriter(tmp_path)

    with pytest.raises(ValueError, match="missing required artifacts"):
        writer.complete({"run_id": writer.run_id})

    writer.fail({"run_id": writer.run_id}, status="partial")
