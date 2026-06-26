from eval.hotpotqa import convert_hotpotqa_row


def test_convert_hotpotqa_row_extracts_supporting_sentences() -> None:
    row = {
        "id": "123",
        "question": "Who wrote the book?",
        "answer": "Alice",
        "level": "medium",
        "type": "bridge",
        "context": {
            "title": ["Doc A", "Doc B"],
            "sentences": [
                ["Sentence A0.", "Sentence A1."],
                ["Sentence B0.", "Sentence B1."],
            ],
        },
        "supporting_facts": {
            "title": ["Doc A", "Doc B"],
            "sent_id": [1, 0],
        },
    }

    sample = convert_hotpotqa_row(row)

    assert sample.id == "hotpotqa_123"
    assert sample.reference == "Alice"
    assert sample.reference_contexts == ["Sentence A1.", "Sentence B0."]
    assert sample.tags == ["hotpotqa", "distractor"]
