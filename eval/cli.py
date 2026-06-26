import argparse
from pathlib import Path

from eval.config import get_eval_settings
from eval.prepare import _default_pg_table, run_hotpotqa_local_pipeline
from eval.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline ragas evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument("--dataset", type=Path, default=None)
    run_parser.add_argument("--tag", action="append", default=None)
    run_parser.add_argument("--limit", type=int, default=None)

    pipeline_parser = subparsers.add_parser(
        "run-hotpotqa-local",
        help="Prepare a local HotpotQA JSON/JSONL file, rebuild an isolated index, and run ragas",
    )
    pipeline_parser.add_argument("--input", type=Path, required=True, dest="input_path")
    pipeline_parser.add_argument("--dataset-out", type=Path, default=None)
    pipeline_parser.add_argument("--corpus-dir", type=Path, default=None)
    pipeline_parser.add_argument("--pg-table", default=None)
    pipeline_parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        artifacts = run_evaluation(
            get_eval_settings(),
            dataset_path=args.dataset,
            tags=args.tag,
            limit=args.limit,
        )
        print(f"Evaluation run saved to: {artifacts.run_dir}")
        return 0

    if args.command == "run-hotpotqa-local":
        dataset_path = args.dataset_out or Path("eval/datasets") / f"{args.input_path.stem}.jsonl"
        corpus_dir = args.corpus_dir or Path("data/eval_uploads") / args.input_path.stem
        pg_table = args.pg_table or _default_pg_table(args.input_path)
        prepared, artifacts = run_hotpotqa_local_pipeline(
            source_path=args.input_path,
            dataset_path=dataset_path,
            corpus_dir=corpus_dir,
            pg_table=pg_table,
            limit=args.limit,
            eval_settings=get_eval_settings(),
        )
        print(f"Prepared {prepared.sample_count} samples into: {prepared.dataset_path}")
        print(f"Wrote {prepared.document_count} corpus documents into: {prepared.corpus_dir}")
        print(f"Evaluation run saved to: {artifacts.run_dir}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
