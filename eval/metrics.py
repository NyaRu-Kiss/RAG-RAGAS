from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    metric: object | None
    skip_reason: str | None = None


def build_metric_specs(*, judge_llm: object | None = None, embeddings: object | None = None) -> list[MetricSpec]:
    from eval.ragas_compat import ensure_ragas_import_compat

    ensure_ragas_import_compat()
    from ragas.metrics import (
        ContextPrecision,
        Faithfulness,
        ResponseRelevancy,
    )

    return [
        MetricSpec(key="context_precision", metric=ContextPrecision(llm=judge_llm)),
        MetricSpec(
            key="response_relevancy",
            metric=ResponseRelevancy(llm=judge_llm, embeddings=embeddings),
        ),
        MetricSpec(key="faithfulness", metric=Faithfulness(llm=judge_llm)),
        MetricSpec(
            key="multimodal_faithfulness",
            metric=None,
            skip_reason="Current RAG pipeline does not expose multimodal evidence yet.",
        ),
        MetricSpec(
            key="multimodal_relevance",
            metric=None,
            skip_reason="Current RAG pipeline does not expose multimodal evidence yet.",
        ),
    ]
