import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class EmptyDatasetError(ValueError):
    """Raised when a readable evaluation dataset has no samples."""


class EvalSample(BaseModel):
    id: str = Field(min_length=1)
    source_sample_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    reference_contexts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    question_type: str | None = None
    images: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_source_sample_id(cls, value: object) -> object:
        if isinstance(value, dict) and "source_sample_id" not in value:
            return {**value, "source_sample_id": value.get("id")}
        return value


def load_dataset(path: Path) -> list[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

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

    if not samples:
        raise EmptyDatasetError(f"Dataset file contains no samples: {path}")

    source_sample_ids: set[str] = set()
    for sample in samples:
        if sample.source_sample_id in source_sample_ids:
            raise ValueError(f"Duplicate source_sample_id: {sample.source_sample_id}")
        source_sample_ids.add(sample.source_sample_id)
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
