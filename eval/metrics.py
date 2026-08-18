from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    metric: object | None
    skip_reason: str | None = None
    result_key: str | None = None

    @property
    def score_key(self) -> str:
        return self.result_key or self.key


def build_metric_specs(*, judge_llm: object | None = None, embeddings: object | None = None) -> list[MetricSpec]:
    from eval.ragas_compat import ensure_ragas_import_compat

    ensure_ragas_import_compat()
    from ragas.metrics import Faithfulness, ResponseRelevancy

    specs = [
        MetricSpec(key="faithfulness", metric=Faithfulness(llm=judge_llm)),
        MetricSpec(
            key="response_relevancy",
            metric=ResponseRelevancy(llm=judge_llm, embeddings=embeddings),
            result_key="answer_relevancy",
        ),
    ]
    try:
        from ragas.metrics import FactualCorrectness

        specs.append(
            MetricSpec(
                key="factual_correctness",
                metric=FactualCorrectness(llm=judge_llm),
                result_key="factual_correctness(mode=f1)",
            )
        )
    except Exception as exc:
        specs.append(MetricSpec(key="factual_correctness", metric=None, skip_reason=str(exc)))
    return specs
