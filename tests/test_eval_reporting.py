from pathlib import Path

from eval.reporting import write_summary_md


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
