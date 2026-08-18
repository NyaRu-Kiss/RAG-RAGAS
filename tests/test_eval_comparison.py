import json
from pathlib import Path

from eval.comparison import compare_runs


def _manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "status": "completed",
        "dataset_sha256": "sha256:dataset",
        "corpus_sha256": "sha256:corpus",
        "source_sample_ids_sha256": "sha256:ids",
        "sample_count": 20,
        "metrics": ["faithfulness"],
        "app_config": {"top_k": 5},
        "eval_config": {"judge_model": "judge-a"},
    }
    manifest.update(overrides)
    return manifest


def _sample(source_id: str, score: float | None, *, status: str = "scored") -> dict[str, object]:
    metric: dict[str, object] = {"status": status}
    if score is not None:
        metric["score"] = score
    return {"source_sample_id": source_id, "status": status, "metrics": {"faithfulness": metric}}


def test_comparison_uses_only_common_finite_scores_and_reports_config_diff(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    (baseline_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [_sample("a", 0.4), _sample("b", None, status="failed")]) + "\n",
        encoding="utf-8",
    )

    result = compare_runs(
        current_manifest=_manifest(app_config={"top_k": 10}),
        current_samples=[_sample("a", 0.7), _sample("b", 0.9)],
        baseline_dir=baseline_dir,
    )

    assert result["comparable"] is True
    assert result["metrics"]["faithfulness"]["mean_delta"] == 0.29999999999999993
    assert result["metrics"]["faithfulness"]["paired_sample_count"] == 1
    assert result["status_changes"]["recovered"] == ["b"]
    assert result["config_diff"]["app_config.top_k"] == {"baseline": 5, "current": 10}


def test_comparison_refuses_aggregate_delta_for_incompatible_runs(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "manifest.json").write_text(
        json.dumps(_manifest(corpus_sha256="sha256:old")), encoding="utf-8"
    )
    (baseline_dir / "samples.jsonl").write_text(json.dumps(_sample("a", 0.4)) + "\n", encoding="utf-8")

    result = compare_runs(
        current_manifest=_manifest(), current_samples=[_sample("a", 0.7)], baseline_dir=baseline_dir
    )

    assert result["comparable"] is False
    assert "corpus_sha256" in result["reasons"]
    assert "metrics" not in result


def test_comparison_includes_pipeline_and_prompt_config_diffs(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "manifest.json").write_text(
        json.dumps(_manifest(pipeline_config={"retrieval": "vector"}, prompt_config={"hash": "old"})),
        encoding="utf-8",
    )
    (baseline_dir / "samples.jsonl").write_text(json.dumps(_sample("a", 0.4)) + "\n", encoding="utf-8")

    result = compare_runs(
        current_manifest=_manifest(pipeline_config={"retrieval": "hybrid"}, prompt_config={"hash": "new"}),
        current_samples=[_sample("a", 0.5)],
        baseline_dir=baseline_dir,
    )

    assert result["config_diff"]["pipeline_config.retrieval"]["current"] == "hybrid"
    assert result["config_diff"]["prompt_config.hash"]["baseline"] == "old"
