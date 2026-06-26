import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    reference_contexts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    question_type: str | None = None
    images: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)


def load_dataset(path: Path) -> list[EvalSample]:
    if not path.exists():
        return []

    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            samples.append(EvalSample.model_validate(payload))
    return samples


def filter_dataset(
    samples: list[EvalSample],
    *,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalSample]:
    selected = samples
    if tags:
        tag_filter = set(tags)
        selected = [sample for sample in selected if tag_filter.intersection(sample.tags)]
    if limit is not None:
        selected = selected[:limit]
    return selected
