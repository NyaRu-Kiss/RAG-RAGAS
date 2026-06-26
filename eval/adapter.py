from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag import RagService


@dataclass
class RAGRunResult:
    response: str
    retrieved_contexts: list[str]
    citations: list[dict[str, object | None]]


class RagEvaluationAdapter:
    def __init__(self, rag_service: "RagService") -> None:
        self._rag_service = rag_service

    def run(self, question: str) -> RAGRunResult:
        answer, citations, contexts = self._rag_service.evaluate_query(question)
        return RAGRunResult(
            response=answer,
            retrieved_contexts=[context.text for context in contexts],
            citations=[citation.model_dump() for citation in citations],
        )
