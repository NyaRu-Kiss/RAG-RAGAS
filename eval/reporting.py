import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_REQUIRED_RUN_ARTIFACTS = {
    "summary.json",
    "samples.jsonl",
    "retrieval_traces.jsonl",
    "failures.json",
    "summary.md",
}


class AtomicRunWriter:
    """Write a run into a staging directory and publish it atomically on success."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        self.staging_dir = self.base_dir / f"{self.run_id}.tmp-{uuid4().hex[:8]}"
        self.staging_dir.mkdir()
        self._finished = False

    def path(self, artifact_name: str) -> Path:
        if Path(artifact_name).name != artifact_name:
            raise ValueError("Artifact name must not contain a directory path")
        return self.staging_dir / artifact_name

    def write_json(self, artifact_name: str, payload: object) -> None:
        write_json(self.path(artifact_name), payload)

    def write_jsonl(self, artifact_name: str, rows: list[dict[str, object]]) -> None:
        write_jsonl(self.path(artifact_name), rows)

    def write_text(self, artifact_name: str, content: str) -> None:
        self.path(artifact_name).write_text(content, encoding="utf-8")

    def write_summary_md(self, summary: dict[str, object]) -> None:
        write_summary_md(self.path("summary.md"), summary)

    def complete(self, manifest: dict[str, object]) -> Path:
        self._assert_open()
        missing = sorted(name for name in _REQUIRED_RUN_ARTIFACTS if not self.path(name).is_file())
        if missing:
            raise ValueError(f"Cannot complete run: missing required artifacts: {', '.join(missing)}")
        final_manifest = {**manifest, "status": "completed", "run_id": self.run_id}
        self.write_json("manifest.json", final_manifest)
        run_dir = self.base_dir / self.run_id
        self.staging_dir.replace(run_dir)
        self._finished = True
        return run_dir

    def fail(self, manifest: dict[str, object], *, status: str) -> Path:
        self._assert_open()
        if status not in {"failed", "partial"}:
            raise ValueError("Failed run status must be failed or partial")
        self.write_json("manifest.json", {**manifest, "status": status, "run_id": self.run_id})
        failed_dir = self.base_dir / f"{self.run_id}.{status}-{uuid4().hex[:8]}"
        self.staging_dir.replace(failed_dir)
        self._finished = True
        return failed_dir

    def _assert_open(self) -> None:
        if self._finished:
            raise RuntimeError("Run writer is already finalized")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Dataset: `{summary['dataset_path']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Failed sample count: `{summary['failed_sample_count']}`",
        "",
        "## Metrics",
        "",
    ]
    for name, value in summary["metrics"].items():
        if isinstance(value, dict):
            lines.append(
                f"- `{name}`: mean={value.get('mean')}, scored={value.get('scored')}, "
                f"p50={value.get('p50')}, p95={value.get('p95')}, "
                f"failed={value.get('failed')}, nan={value.get('nan')}"
            )
        else:
            lines.append(f"- `{name}`: {value}")
    if summary.get("skipped_metrics"):
        lines.extend(["", "## Skipped Metrics", ""])
        for name, reason in summary["skipped_metrics"].items():
            lines.append(f"- `{name}`: {reason}")
    if summary.get("comparison"):
        comparison = summary["comparison"]
        lines.extend(["", "## Baseline Comparison", ""])
        if comparison.get("comparable"):
            for name, value in comparison.get("metrics", {}).items():
                lines.append(f"- `{name}` delta: {value.get('mean_delta')} ({value.get('paired_sample_count')} paired)")
        else:
            lines.append(f"- Not comparable: {', '.join(comparison.get('reasons', []))}")
    if summary.get("failures"):
        lines.extend(["", "## Failed Samples", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- `{failure.get('source_sample_id')}` ({failure.get('stage')}): `{failure.get('trace')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
