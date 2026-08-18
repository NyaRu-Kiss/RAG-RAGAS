import argparse
import json
from pathlib import Path

from datasets import load_dataset

from eval.dataset import EvalSample


def _normalize_paragraph(text: str) -> str:
    return " ".join(text.split())


def _extract_reference_contexts(row: dict[str, object]) -> list[str]:
    context = row.get("context")
    supporting_facts = row.get("supporting_facts")
    if not isinstance(context, dict) or not isinstance(supporting_facts, dict):
        return []

    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    support_titles = supporting_facts.get("title", [])
    support_sentence_ids = supporting_facts.get("sent_id", [])
    if not all(isinstance(value, list) for value in [titles, sentences, support_titles, support_sentence_ids]):
        return []

    title_to_sentences: dict[str, list[str]] = {}
    for title, paragraph_sentences in zip(titles, sentences, strict=False):
        if isinstance(title, str) and isinstance(paragraph_sentences, list):
            title_to_sentences[title] = [sentence for sentence in paragraph_sentences if isinstance(sentence, str)]

    extracted: list[str] = []
    seen: set[str] = set()
    for title, sent_id in zip(support_titles, support_sentence_ids, strict=False):
        if not isinstance(title, str) or not isinstance(sent_id, int):
            continue
        paragraph = title_to_sentences.get(title, [])
        if 0 <= sent_id < len(paragraph):
            sentence = _normalize_paragraph(paragraph[sent_id])
            if sentence and sentence not in seen:
                extracted.append(sentence)
                seen.add(sentence)
    return extracted


def convert_hotpotqa_row(row: dict[str, object]) -> EvalSample:
    row_id = str(row["id"])
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    level = row.get("level")
    question_type = row.get("type")

    return EvalSample(
        id=f"hotpotqa_{row_id}",
        source_sample_id=row_id,
        user_input=question,
        reference=answer,
        reference_contexts=_extract_reference_contexts(row),
        tags=["hotpotqa", "distractor"],
        difficulty=str(level) if level is not None else None,
        question_type=str(question_type) if question_type is not None else "multi_hop",
    )


def export_hotpotqa_subset(
    *,
    output_path: Path,
    limit: int = 50,
    split: str = "validation",
    config_name: str = "distractor",
) -> int:
    dataset = load_dataset("hotpotqa/hotpot_qa", config_name, split=f"{split}[:{limit}]")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            sample = convert_hotpotqa_row(row)
            handle.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a HotpotQA subset into eval jsonl format")
    parser.add_argument("--output", type=Path, default=Path("eval/datasets/hotpotqa_50.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="distractor")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    count = export_hotpotqa_subset(
        output_path=args.output,
        limit=args.limit,
        split=args.split,
        config_name=args.config,
    )
    print(f"Exported {count} samples to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
