import json
from pathlib import Path

import pytest

from eval.prepare import build_eval_rag_service, default_eval_table, eval_storage_table_names, prepare_hotpotqa_records


def _row(sample_id: str, title: str, paragraphs: list[list[str]]) -> dict[str, object]:
    return {
        "id": sample_id,
        "question": f"Question {sample_id}",
        "answer": f"Answer {sample_id}",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": [title], "sent_id": [0]},
        "context": {"title": [title] * len(paragraphs), "sentences": paragraphs},
    }


def test_prepare_creates_shared_corpus_and_generation_subset(tmp_path: Path) -> None:
    records = [
        _row("1", "Shared", [["Same paragraph."]]),
        _row("2", "Shared", [["Same paragraph."]]),
        _row("3", "Other", [["Other paragraph."]]),
    ]

    prepared = prepare_hotpotqa_records(
        records,
        output_dir=tmp_path / "hotpotqa-distractor",
        seed=7,
        retrieval_count=3,
        generation_count=2,
        raw_sha256="sha256:raw",
    )

    corpus = [json.loads(line) for line in prepared.corpus_path.read_text(encoding="utf-8").splitlines()]
    retrieval = [json.loads(line) for line in prepared.retrieval_path.read_text(encoding="utf-8").splitlines()]
    generation = [json.loads(line) for line in prepared.generation_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    shared = next(row for row in corpus if row["title"] == "Shared")

    assert len(corpus) == 2
    assert sorted(shared["source_sample_ids"]) == ["1", "2"]
    assert len(retrieval) == 3
    assert len(generation) == 2
    assert {row["source_sample_id"] for row in generation}.issubset(
        {row["source_sample_id"] for row in retrieval}
    )
    assert manifest["corpus_identity"] == "title + paragraph_index"
    assert manifest["eval_tables"] == {
        "pg_table": "eval_hotpotqa_distractor",
        "docstore_table": "eval_hotpotqa_distractor_docstore",
        "file_index_table": "eval_hotpotqa_distractor_file_index",
    }
    assert manifest["raw_sha256"] == "sha256:raw"
    assert manifest["source"]["url"] == "https://huggingface.co/datasets/hotpotqa/hotpot_qa"


def test_prepare_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    records = [_row(str(index), f"Title {index}", [[f"Paragraph {index}"]]) for index in range(8)]

    first = prepare_hotpotqa_records(records, output_dir=tmp_path / "first", seed=3, retrieval_count=5, generation_count=2)
    second = prepare_hotpotqa_records(records, output_dir=tmp_path / "second", seed=3, retrieval_count=5, generation_count=2)

    assert first.retrieval_path.read_text(encoding="utf-8") == second.retrieval_path.read_text(encoding="utf-8")
    assert first.generation_path.read_text(encoding="utf-8") == second.generation_path.read_text(encoding="utf-8")
    assert first.corpus_path.read_text(encoding="utf-8") == second.corpus_path.read_text(encoding="utf-8")


def test_prepare_rejects_conflicting_stable_paragraph_identity(tmp_path: Path) -> None:
    records = [
        _row("1", "Shared", [["First content"]]),
        _row("2", "Shared", [["Different content"]]),
    ]

    with pytest.raises(ValueError, match="Conflicting content"):
        prepare_hotpotqa_records(records, output_dir=tmp_path / "prepared", seed=1, retrieval_count=2, generation_count=1)


def test_prepare_refuses_to_overwrite_existing_dataset_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_hotpotqa_records([_row("1", "Title", [["Text"]])], output_dir=output_dir, seed=1, retrieval_count=1, generation_count=1)


def test_default_eval_table_uses_isolated_namespace() -> None:
    assert default_eval_table("hotpotqa-distractor") == "eval_hotpotqa_distractor"
    assert all(name.startswith("eval_") for name in eval_storage_table_names("hotpotqa-distractor").values())


def test_production_prepare_requires_fixed_retrieval_population(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text(json.dumps(_row("1", "Title", [["Text"]])) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 200"):
        from eval.prepare import prepare_hotpotqa_dataset

        prepare_hotpotqa_dataset(source_path=source_path, output_dir=tmp_path / "prepared", seed=1)


def test_build_eval_service_rejects_missing_index_state_before_loading_app(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "dataset_manifest.json").write_text(
        json.dumps({"dataset": "hotpotqa-distractor", "eval_tables": eval_storage_table_names("hotpotqa-distractor")}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="index state"):
        build_eval_rag_service(prepared_dir=prepared_dir)
