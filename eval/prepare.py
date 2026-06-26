import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval.config import EvalSettings, get_eval_settings
from eval.dataset import EvalSample
from eval.hotpotqa import convert_hotpotqa_row
from eval.runner import EvalRunArtifacts, run_evaluation

if TYPE_CHECKING:
    from app.rag import RagService


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "dataset"


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object at line {line_no}, got {type(payload).__name__}")
                records.append(payload)
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(payload).__name__}")
    records = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at item {idx}, got {type(item).__name__}")
        records.append(item)
    return records


def _build_context_sections(row: dict[str, Any]) -> list[str]:
    context = row.get("context")
    if not isinstance(context, dict):
        return []

    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return []

    sections: list[str] = []
    for title, paragraph_sentences in zip(titles, sentences, strict=False):
        if not isinstance(title, str) or not isinstance(paragraph_sentences, list):
            continue
        normalized_sentences = [
            _normalize_text(sentence)
            for sentence in paragraph_sentences
            if isinstance(sentence, str) and _normalize_text(sentence)
        ]
        if not normalized_sentences:
            continue
        body = " ".join(normalized_sentences)
        sections.append(f"Title: {title}\n{body}")
    return sections


def _sample_document_name(sample: EvalSample, index: int) -> str:
    return f"{index:03d}_{_slugify(sample.id)}.txt"


@dataclass
class PreparedDatasetArtifacts:
    source_path: Path
    dataset_path: Path
    corpus_dir: Path
    sample_count: int
    document_count: int


def prepare_hotpotqa_dataset(
    *,
    source_path: Path,
    dataset_path: Path,
    corpus_dir: Path,
    limit: int | None = None,
) -> PreparedDatasetArtifacts:
    records = _load_json_records(source_path)
    if limit is not None:
        records = records[:limit]

    corpus_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    document_count = 0
    with dataset_path.open("w", encoding="utf-8") as dataset_handle:
        for index, row in enumerate(records, start=1):
            sample = convert_hotpotqa_row(row)
            dataset_handle.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            sample_count += 1

            sections = _build_context_sections(row)
            if not sections:
                continue
            document_path = corpus_dir / _sample_document_name(sample, index)
            document_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
            document_count += 1

    return PreparedDatasetArtifacts(
        source_path=source_path,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        sample_count=sample_count,
        document_count=document_count,
    )


def build_isolated_rag_service(
    *,
    corpus_dir: Path,
    pg_table: str,
) -> "RagService":
    from app.config import Settings, get_settings
    from app.rag import RagService

    base_settings = get_settings()
    isolated_settings: Settings = base_settings.model_copy(
        update={
            "upload_dir": corpus_dir,
            "pg_table": pg_table,
        }
    )
    isolated_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return RagService(isolated_settings)


def run_hotpotqa_local_pipeline(
    *,
    source_path: Path,
    dataset_path: Path,
    corpus_dir: Path,
    pg_table: str,
    limit: int | None = None,
    eval_settings: EvalSettings | None = None,
) -> tuple[PreparedDatasetArtifacts, EvalRunArtifacts]:
    prepared = prepare_hotpotqa_dataset(
        source_path=source_path,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        limit=limit,
    )
    rag_service = build_isolated_rag_service(corpus_dir=prepared.corpus_dir, pg_table=pg_table)
    rag_service.rebuild_index()
    artifacts = run_evaluation(
        eval_settings or get_eval_settings(),
        dataset_path=prepared.dataset_path,
        limit=limit,
        rag_service=rag_service,
    )
    return prepared, artifacts


def _default_dataset_path(source_path: Path) -> Path:
    return Path("eval/datasets") / f"{source_path.stem}.jsonl"


def _default_corpus_dir(source_path: Path) -> Path:
    return Path("data/eval_uploads") / source_path.stem


def _default_pg_table(source_path: Path) -> str:
    return f"rag_eval_{_slugify(source_path.stem)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare raw HotpotQA samples for local RAG evaluation")
    parser.add_argument("--input", type=Path, required=True, dest="input_path")
    parser.add_argument("--dataset-out", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--pg-table", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-eval", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset_out or _default_dataset_path(args.input_path)
    corpus_dir = args.corpus_dir or _default_corpus_dir(args.input_path)
    pg_table = args.pg_table or _default_pg_table(args.input_path)

    if args.run_eval:
        prepared, artifacts = run_hotpotqa_local_pipeline(
            source_path=args.input_path,
            dataset_path=dataset_path,
            corpus_dir=corpus_dir,
            pg_table=pg_table,
            limit=args.limit,
        )
        print(f"Prepared {prepared.sample_count} samples into: {prepared.dataset_path}")
        print(f"Wrote {prepared.document_count} corpus documents into: {prepared.corpus_dir}")
        print(f"Evaluation run saved to: {artifacts.run_dir}")
        return 0

    prepared = prepare_hotpotqa_dataset(
        source_path=args.input_path,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        limit=args.limit,
    )
    print(f"Prepared {prepared.sample_count} samples into: {prepared.dataset_path}")
    print(f"Wrote {prepared.document_count} corpus documents into: {prepared.corpus_dir}")
    print(f"Suggested pgvector table: {pg_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
