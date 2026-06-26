import json
from datetime import datetime, timezone
from pathlib import Path


def create_run_dir(base_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
        lines.append(f"- `{name}`: {value}")
    if summary.get("skipped_metrics"):
        lines.extend(["", "## Skipped Metrics", ""])
        for name, reason in summary["skipped_metrics"].items():
            lines.append(f"- `{name}`: {reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
