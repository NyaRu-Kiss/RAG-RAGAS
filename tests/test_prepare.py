import json
from pathlib import Path

from eval.prepare import prepare_hotpotqa_dataset


def test_prepare_hotpotqa_dataset_writes_jsonl_and_corpus(tmp_path: Path) -> None:
    source_path = tmp_path / "hotpotqa.json"
    dataset_path = tmp_path / "prepared.jsonl"
    corpus_dir = tmp_path / "corpus"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": "abc",
                    "question": "Who wrote the book?",
                    "answer": "Alice",
                    "type": "bridge",
                    "level": "medium",
                    "supporting_facts": {
                        "title": ["Doc A"],
                        "sent_id": [1],
                    },
                    "context": {
                        "title": ["Doc A", "Doc B"],
                        "sentences": [
                            ["Sentence A0.", "Sentence A1."],
                            ["Sentence B0."],
                        ],
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prepared = prepare_hotpotqa_dataset(
        source_path=source_path,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
    )

    assert prepared.sample_count == 1
    assert prepared.document_count == 1
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "hotpotqa_abc"
    assert payload["reference_contexts"] == ["Sentence A1."]

    corpus_files = sorted(corpus_dir.glob("*.txt"))
    assert len(corpus_files) == 1
    document_text = corpus_files[0].read_text(encoding="utf-8")
    assert "Title: Doc A" in document_text
    assert "Sentence A0. Sentence A1." in document_text
    assert "Title: Doc B" in document_text


def test_prepare_hotpotqa_dataset_honors_limit(tmp_path: Path) -> None:
    source_path = tmp_path / "hotpotqa.json"
    dataset_path = tmp_path / "prepared.jsonl"
    corpus_dir = tmp_path / "corpus"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "Q1",
                    "answer": "A1",
                    "type": "bridge",
                    "level": "easy",
                    "supporting_facts": {"title": ["Doc A"], "sent_id": [0]},
                    "context": {"title": ["Doc A"], "sentences": [["Sentence A0."]]},
                },
                {
                    "id": "2",
                    "question": "Q2",
                    "answer": "A2",
                    "type": "bridge",
                    "level": "easy",
                    "supporting_facts": {"title": ["Doc B"], "sent_id": [0]},
                    "context": {"title": ["Doc B"], "sentences": [["Sentence B0."]]},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prepared = prepare_hotpotqa_dataset(
        source_path=source_path,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        limit=1,
    )

    assert prepared.sample_count == 1
    assert prepared.document_count == 1
    assert len(dataset_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(list(corpus_dir.glob("*.txt"))) == 1
