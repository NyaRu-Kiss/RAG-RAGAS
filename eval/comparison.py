"""Strict, artifact-only baseline comparison for generation evaluation runs."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def compare_runs(
    *,
    current_manifest: dict[str, object],
    current_samples: list[dict[str, object]],
    baseline_dir: Path,
) -> dict[str, object]:
    baseline_manifest = _load_json(baseline_dir / "manifest.json")
    if baseline_manifest.get("status") != "completed":
        raise ValueError("Baseline run must have status completed")
    baseline_samples = _load_jsonl(baseline_dir / "samples.jsonl")
    reasons = _compatibility_reasons(current_manifest, baseline_manifest)
    result: dict[str, object] = {
        "baseline_run": str(baseline_dir),
        "comparable": not reasons,
        "reasons": reasons,
        "config_diff": _config_diff(baseline_manifest, current_manifest),
    }
    if reasons:
        return result

    current_by_id = {str(row["source_sample_id"]): row for row in current_samples}
    baseline_by_id = {str(row["source_sample_id"]): row for row in baseline_samples}
    metric_keys = [str(key) for key in current_manifest["metrics"]]
    metric_result: dict[str, object] = {}
    for key in metric_keys:
        deltas: list[dict[str, object]] = []
        recovered: list[str] = []
        newly_failed: list[str] = []
        missing: list[str] = []
        for source_id in sorted(set(current_by_id) | set(baseline_by_id)):
            current = current_by_id.get(source_id)
            baseline = baseline_by_id.get(source_id)
            if current is None or baseline is None:
                missing.append(source_id)
                continue
            current_score = _finite_score(current, key)
            baseline_score = _finite_score(baseline, key)
            if current_score is not None and baseline_score is not None:
                deltas.append(
                    {
                        "source_sample_id": source_id,
                        "baseline": baseline_score,
                        "current": current_score,
                        "delta": current_score - baseline_score,
                    }
                )
            elif current_score is not None:
                recovered.append(source_id)
            elif baseline_score is not None:
                newly_failed.append(source_id)
        values = [float(item["delta"]) for item in deltas]
        metric_result[key] = {
            "paired_sample_count": len(deltas),
            "mean_delta": sum(values) / len(values) if values else None,
            "samples": deltas,
            "recovered": recovered,
            "newly_failed": newly_failed,
            "missing": missing,
        }
    result["metrics"] = metric_result
    result["status_changes"] = {
        "recovered": sorted({item for metric in metric_result.values() for item in metric["recovered"]}),
        "newly_failed": sorted({item for metric in metric_result.values() for item in metric["newly_failed"]}),
        "missing": sorted({item for metric in metric_result.values() for item in metric["missing"]}),
    }
    return result


def _compatibility_reasons(current: Mapping[str, object], baseline: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    for key in ("dataset_sha256", "corpus_sha256", "source_sample_ids_sha256", "sample_count", "metrics"):
        if current.get(key) != baseline.get(key):
            reasons.append(key)
    if current.get("sample_count") != 20:
        reasons.append("sample_count_not_20")
    return reasons


def _finite_score(row: Mapping[str, object], metric_key: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_key)
    if not isinstance(metric, Mapping) or metric.get("status") != "scored":
        return None
    score = metric.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        return None
    return float(score)


def _config_diff(baseline: Mapping[str, object], current: Mapping[str, object]) -> dict[str, dict[str, object]]:
    diff: dict[str, dict[str, object]] = {}
    for root in ("app_config", "pipeline_config", "eval_config", "embedding", "prompt_config"):
        before = baseline.get(root, {})
        after = current.get(root, {})
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            if before != after:
                diff[root] = {"baseline": before, "current": after}
            continue
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                diff[f"{root}.{key}"] = {"baseline": before.get(key), "current": after.get(key)}
    return diff


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline artifact not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
