import argparse
import json
import logging
from pathlib import Path

from eval.config import get_eval_settings
from eval.prepare import (
    HOT_POT_DATASET,
    build_eval_rag_service,
    dataset_dir,
    fetch_hotpotqa_raw,
    prepare_hotpotqa_dataset,
    rebuild_eval_index,
)
from eval.reporting import write_summary_md
from eval.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG generation evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(dest="data_command", required=True)
    fetch = data.add_parser("fetch")
    fetch.add_argument("--dataset", choices=[HOT_POT_DATASET], required=True)
    prepare = data.add_parser("prepare")
    prepare.add_argument("--dataset", choices=[HOT_POT_DATASET], required=True)
    prepare.add_argument("--seed", type=int, required=True)
    index = commands.add_parser("index").add_subparsers(dest="index_command", required=True)
    rebuild = index.add_parser("rebuild")
    rebuild.add_argument("--dataset", choices=[HOT_POT_DATASET], required=True)
    run = commands.add_parser("run").add_subparsers(dest="run_command", required=True)
    generation = run.add_parser("generation")
    generation.add_argument("--dataset", choices=[HOT_POT_DATASET], required=True)
    generation.add_argument("--baseline", type=Path, default=None)
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    root = Path("eval/datasets")
    if args.command == "data" and args.data_command == "fetch":
        output = Path("data/eval_raw") / args.dataset / "validation.jsonl"
        fetch_hotpotqa_raw(output_path=output)
        print(f"Downloaded raw dataset to: {output}")
        return 0
    if args.command == "data" and args.data_command == "prepare":
        raw = Path("data/eval_raw") / args.dataset / "validation.jsonl"
        prepared = prepare_hotpotqa_dataset(source_path=raw, output_dir=dataset_dir(args.dataset, root), seed=args.seed)
        print(f"Prepared shared corpus with {prepared.corpus_count} paragraphs: {prepared.dataset_dir}")
        return 0
    if args.command == "index":
        state = rebuild_eval_index(prepared_dir=dataset_dir(args.dataset, root))
        print(f"Index state written to: {state}")
        return 0
    if args.command == "run":
        settings = get_eval_settings()
        prepared = dataset_dir(args.dataset, root)
        rag_service = build_eval_rag_service(prepared_dir=prepared)
        baseline = _resolve_completed_run(args.baseline, settings.eval_output_dir) if args.baseline else None
        artifacts = run_evaluation(
            settings,
            dataset_path=prepared / "generation.jsonl",
            rag_service=rag_service,
            require_generation_preflight=True,
            baseline=baseline,
        )
        print(f"Evaluation run saved to: {artifacts.run_dir}")
        return 0
    run_dir = _resolve_completed_run(args.run, Path("eval/reports"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    write_summary_md(run_dir / "summary.md", summary)
    print(f"Report regenerated: {run_dir / 'summary.md'}")
    return 0


def _resolve_completed_run(value: Path, reports_dir: Path) -> Path:
    run_dir = value if value.is_dir() else reports_dir / value
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only completed evaluation runs can be reported or used as baselines")
    return run_dir


if __name__ == "__main__":
    raise SystemExit(main())
