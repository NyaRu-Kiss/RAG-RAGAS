from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag import RagService


@dataclass
class RAGRunResult:
    response: str
    retrieved_contexts: list[str]
    citations: list[dict[str, object | None]]
    retrieval_trace: dict[str, object]
    generation_request: object | None


class RagEvaluationAdapter:
    def __init__(self, rag_service: "RagService") -> None:
        self._rag_service = rag_service

    def run(self, question: str) -> RAGRunResult:
        pipeline_result = self._rag_service.evaluate_query_with_trace(question)
        return RAGRunResult(
            response=pipeline_result.answer,
            retrieved_contexts=[context.text for context in pipeline_result.retrieved_contexts],
            citations=[citation.model_dump() for citation in pipeline_result.citations],
            retrieval_trace=pipeline_result.retrieval_trace,
            generation_request=getattr(pipeline_result, "generation_request", None),
        )


def find_generation_input_leaks(
    *,
    input_sources: tuple[str, ...],
) -> list[str]:
    """Identify evaluation-only fields declared as generation input sources.

    Comparing text values is invalid here: an honest retriever can return the
    same corpus evidence used as the reference context.  Provenance therefore
    guards the data flow instead of rejecting matching content.
    """
    forbidden_sources = {"reference", "reference_contexts", "reference_images"}
    return sorted(forbidden_sources.intersection(input_sources))
