"""Preparation and isolated indexing for fixed offline evaluation datasets."""

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.artifacts import canonical_json_sha256, file_sha256, source_sample_ids_sha256, validate_eval_table_name
from eval.dataset import EvalSample
from eval.hotpotqa import convert_hotpotqa_row


HOT_POT_DATASET = "hotpotqa-distractor"
HOT_POT_HUB_ID = "hotpotqa/hotpot_qa"
HOT_POT_SOURCE_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
HOT_POT_CONFIG = "distractor"
HOT_POT_SPLIT = "validation"
HOT_POT_LICENSE = "CC BY-SA 4.0"
RETRIEVAL_SAMPLE_COUNT = 200
GENERATION_SAMPLE_COUNT = 20


@dataclass(frozen=True)
class PreparedDatasetArtifacts:
    dataset_dir: Path
    corpus_path: Path
    generation_path: Path
    retrieval_path: Path
    manifest_path: Path
    corpus_count: int


def dataset_dir(dataset_name: str, root: Path = Path("eval/datasets")) -> Path:
    if dataset_name != HOT_POT_DATASET:
        raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")
    return root / dataset_name


def default_eval_table(dataset_name: str) -> str:
    if dataset_name != HOT_POT_DATASET:
        raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")
    return "eval_hotpotqa_distractor"


def eval_storage_table_names(dataset_name: str) -> dict[str, str]:
    vector_table = default_eval_table(dataset_name)
    names = {
        "pg_table": vector_table,
        "docstore_table": f"{vector_table}_docstore",
        "file_index_table": f"{vector_table}_file_index",
    }
    for table_name in names.values():
        validate_eval_table_name(table_name)
    return names


def fetch_hotpotqa_raw(*, output_path: Path) -> Path:
    """Download the pinned public split and persist raw records plus provenance."""
    from datasets import load_dataset

    dataset = load_dataset(HOT_POT_HUB_ID, HOT_POT_CONFIG, split=HOT_POT_SPLIT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    metadata = {
        "dataset": HOT_POT_HUB_ID,
        "source_url": HOT_POT_SOURCE_URL,
        "config": HOT_POT_CONFIG,
        "split": HOT_POT_SPLIT,
        "version": str(dataset.info.version) if dataset.info.version is not None else None,
        "license": HOT_POT_LICENSE,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": file_sha256(output_path),
    }
    (output_path.parent / "download_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Raw HotpotQA data must be a JSON array or JSONL of objects")
    return records


def _selected_records(records: list[dict[str, Any]], *, seed: int, count: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"HotpotQA source has {len(records)} records; at least {count} are required")
    indices = sorted(random.Random(seed).sample(range(len(records)), count))
    return [records[index] for index in indices]


def _paragraphs(row: dict[str, Any]) -> list[tuple[str, int, str]]:
    context = row.get("context")
    if not isinstance(context, dict):
        return []
    titles, sentences = context.get("title"), context.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return []
    paragraphs: list[tuple[str, int, str]] = []
    for index, (title, paragraph_sentences) in enumerate(zip(titles, sentences, strict=False)):
        if not isinstance(title, str) or not isinstance(paragraph_sentences, list):
            continue
        text = " ".join(sentence.strip() for sentence in paragraph_sentences if isinstance(sentence, str) and sentence.strip())
        if text:
            paragraphs.append((title, index, text))
    return paragraphs


def prepare_hotpotqa_records(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    seed: int,
    retrieval_count: int = RETRIEVAL_SAMPLE_COUNT,
    generation_count: int = GENERATION_SAMPLE_COUNT,
    raw_sha256: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> PreparedDatasetArtifacts:
    """Create fixed query sets and one shared corpus from raw HotpotQA rows."""
    if generation_count > retrieval_count:
        raise ValueError("generation_count must not exceed retrieval_count")
    if output_dir.exists():
        raise FileExistsError(f"Prepared dataset directory already exists: {output_dir}")

    retrieval_records = _selected_records(records, seed=seed, count=retrieval_count)
    generation_records = _selected_records(retrieval_records, seed=seed + 1, count=generation_count)
    retrieval_samples = [convert_hotpotqa_row(row) for row in retrieval_records]
    generation_ids = {str(row["id"]) for row in generation_records}
    generation_samples = [sample for sample in retrieval_samples if sample.source_sample_id in generation_ids]

    corpus: dict[str, dict[str, Any]] = {}
    for row in retrieval_records:
        source_sample_id = str(row["id"])
        for title, paragraph_index, text in _paragraphs(row):
            paragraph_id = canonical_json_sha256([title, paragraph_index])
            entry = corpus.get(paragraph_id)
            if entry is None:
                corpus[paragraph_id] = {
                    "id": paragraph_id,
                    "title": title,
                    "paragraph_index": paragraph_index,
                    "text": text,
                    "source_sample_ids": [source_sample_id],
                }
            elif entry["text"] != text:
                raise ValueError(f"Conflicting content for corpus paragraph {title!r} index {paragraph_index}")
            elif source_sample_id not in entry["source_sample_ids"]:
                entry["source_sample_ids"].append(source_sample_id)

    output_dir.mkdir(parents=True)
    corpus_path = output_dir / "corpus.jsonl"
    retrieval_path = output_dir / "retrieval.jsonl"
    generation_path = output_dir / "generation.jsonl"
    _write_jsonl(corpus_path, [corpus[key] for key in sorted(corpus)])
    _write_jsonl(retrieval_path, [sample.model_dump() for sample in retrieval_samples])
    _write_jsonl(generation_path, [sample.model_dump() for sample in generation_samples])
    manifest = {
        "dataset": HOT_POT_DATASET,
        "source": {
            "hub_id": HOT_POT_HUB_ID,
            "url": HOT_POT_SOURCE_URL,
            "config": HOT_POT_CONFIG,
            "split": HOT_POT_SPLIT,
            "license": HOT_POT_LICENSE,
            "version": (source_metadata or {}).get("version"),
            "downloaded_at_utc": (source_metadata or {}).get("downloaded_at_utc"),
        },
        "conversion_version": 1,
        "sampling_seed": seed,
        "raw_sha256": raw_sha256,
        "retrieval_source_sample_ids": [sample.source_sample_id for sample in retrieval_samples],
        "generation_source_sample_ids": [sample.source_sample_id for sample in generation_samples],
        "source_sample_ids_sha256": source_sample_ids_sha256([sample.source_sample_id for sample in generation_samples]),
        "corpus_identity": "title + paragraph_index",
        "corpus_deduplication": "stable paragraph identity; conflicting text fails",
        "corpus_sha256": file_sha256(corpus_path),
        "files": {"corpus.jsonl": file_sha256(corpus_path), "retrieval.jsonl": file_sha256(retrieval_path), "generation.jsonl": file_sha256(generation_path)},
        "eval_tables": eval_storage_table_names(HOT_POT_DATASET),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PreparedDatasetArtifacts(output_dir, corpus_path, generation_path, retrieval_path, manifest_path, len(corpus))


def prepare_hotpotqa_dataset(*, source_path: Path, output_dir: Path, seed: int) -> PreparedDatasetArtifacts:
    download_manifest = source_path.parent / "download_manifest.json"
    source_metadata = json.loads(download_manifest.read_text(encoding="utf-8")) if download_manifest.is_file() else None
    return prepare_hotpotqa_records(
        load_json_records(source_path),
        output_dir=output_dir,
        seed=seed,
        raw_sha256=file_sha256(source_path),
        source_metadata=source_metadata,
    )


def rebuild_eval_index(*, prepared_dir: Path) -> Path:
    """Materialize the shared corpus and rebuild only its validated eval table."""
    manifest_path = prepared_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_names = _eval_table_names_from_manifest(manifest)
    table_name = table_names["pg_table"]
    corpus_path = prepared_dir / "corpus.jsonl"
    if file_sha256(corpus_path) != manifest["corpus_sha256"]:
        raise ValueError("Corpus hash does not match dataset manifest")
    materialized_dir = Path("data/eval_uploads") / str(manifest["dataset"])
    if materialized_dir.exists():
        shutil.rmtree(materialized_dir)
    materialized_dir.mkdir(parents=True)
    corpus_rows = load_json_records(corpus_path)
    for row in corpus_rows:
        paragraph_id = str(row["id"]).removeprefix("sha256:")
        (materialized_dir / f"{paragraph_id}.txt").write_text(
            f"Title: {row['title']}\n{row['text']}\n", encoding="utf-8"
        )

    from app.config import Settings, get_settings
    from app.rag import RagService

    base = get_settings()
    settings: Settings = base.model_copy(
        update={"upload_dir": materialized_dir, "embed_batch_size": 1, **table_names}
    )
    rag_service = RagService(settings)
    chunk_count = rag_service.rebuild_index()
    state = {
        "dataset": manifest["dataset"],
        "table_name": table_name,
        "storage_tables": table_names,
        "corpus_sha256": manifest["corpus_sha256"],
        "document_count": len(corpus_rows),
        "chunk_count": chunk_count,
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_config_sha256": canonical_json_sha256({"chunk_mode": settings.chunk_mode, "top_k": settings.top_k, "embed_model": settings.embed_model_name}),
    }
    state_path = prepared_dir / "index_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state_path


def build_eval_rag_service(*, prepared_dir: Path):
    """Build a RAG service only after its isolated shared-corpus index is verified."""
    manifest = json.loads((prepared_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    state_path = prepared_dir / "index_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Evaluation index state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    table_names = _eval_table_names_from_manifest(manifest)
    table_name = table_names["pg_table"]
    corpus_path = prepared_dir / "corpus.jsonl"
    if (
        state.get("table_name") != table_name
        or state.get("storage_tables") != table_names
        or state.get("corpus_sha256") != file_sha256(corpus_path)
    ):
        raise ValueError("Evaluation index state does not match the prepared corpus")
    materialized_dir = Path("data/eval_uploads") / str(manifest["dataset"])
    if not materialized_dir.is_dir():
        raise FileNotFoundError(f"Materialized evaluation corpus not found: {materialized_dir}")

    from app.config import Settings, get_settings
    from app.rag import RagService

    base = get_settings()
    settings: Settings = base.model_copy(
        update={"upload_dir": materialized_dir, "embed_batch_size": 1, **table_names}
    )
    return RagService(settings)


def _eval_table_names_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    table_names = manifest.get("eval_tables")
    if not isinstance(table_names, dict) or set(table_names) != {"pg_table", "docstore_table", "file_index_table"}:
        raise ValueError("Dataset manifest must define all isolated eval storage tables")
    normalized = {key: validate_eval_table_name(str(value)) for key, value in table_names.items()}
    return normalized


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
