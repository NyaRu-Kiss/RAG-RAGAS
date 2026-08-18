from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.rag import RagService
from app.schemas import Citation, RetrievedContext
from eval.adapter import RagEvaluationAdapter, find_generation_input_leaks


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "EMBED_MODEL_NAME": "BAAI/bge-m3",
        "EMBED_MODEL_PATH": "/nonexistent/path",
        "UPLOAD_DIR": "/tmp/uploads",
        "SYSTEM_PROMPT": "system instruction",
        "TOP_K": 2,
        "RETRIEVAL_TOP_K": 3,
    }
    values.update(overrides)
    return Settings(**values)


def _node(node_id: str, score: float, text: str, *, file_name: str = "source.txt") -> MagicMock:
    node = MagicMock()
    node.node_id = node_id
    node.score = score
    node.node.node_id = node_id
    node.node.metadata = {"file_name": file_name, "file_path": f"/data/{file_name}", "page_label": "2"}
    node.node.get_content.return_value = text
    return node


def _service(**overrides: object) -> RagService:
    service = object.__new__(RagService)
    service.settings = _settings(**overrides)
    service._reranker = None
    return service


def test_pipeline_trace_without_reranker_keeps_initial_and_final_nodes() -> None:
    service = _service()
    initial_nodes = [_node("node-1", 0.9, "First chunk"), _node("node-2", 0.8, "Second chunk")]
    service._ensure_index = MagicMock(return_value=MagicMock())
    service._retrieve_nodes = MagicMock(return_value=initial_nodes)
    service._generate_with_gemini = MagicMock(return_value="answer")

    result = service.evaluate_query_with_trace("What happened?")

    trace = result.retrieval_trace
    assert result.answer == "answer"
    assert trace["retrieved_nodes"] == trace["final_contexts"]
    assert trace["candidate_count"] == 2
    assert trace["reranker_enabled"] is False
    assert trace["rerank_input_count"] == 2
    assert trace["rerank_output_count"] == 2
    assert trace["retrieved_nodes"][0]["node_id"] == "node-1"
    assert trace["retrieved_nodes"][0]["text"] == "First chunk"
    assert trace["generation_input"]["serialized_context"] == "[1] source.txt (page 2)\nFirst chunk\n\n[2] source.txt (page 2)\nSecond chunk"
    assert isinstance(trace["generation_input"]["context_token_count"], int)
    assert isinstance(trace["generation_input"]["request_token_count"], int)
    assert trace["generation_input"]["context_operations"] == ["normalize_whitespace", "preserve_final_order", "serialize"]
    service._retrieve_nodes.assert_called_once()
    service._generate_with_gemini.assert_called_once()


def test_deepseek_client_uses_configured_timeout() -> None:
    with patch("app.rag.OpenAI") as openai, patch.object(RagService, "_configure_llama_index"):
        RagService(_settings(LLM_PROVIDER="deepseek", DEEPSEEK_API_KEY="test-key", LLM_TIMEOUT_SECONDS=37))

    assert openai.call_args.kwargs["timeout"] == 37


def test_deepseek_generation_disables_thinking_mode() -> None:
    service = _service(LLM_PROVIDER="deepseek")
    service._deepseek_client = MagicMock()
    service._deepseek_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )

    assert service._generate_with_deepseek("question") == "answer"
    assert service._deepseek_client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_pipeline_trace_records_rerank_before_and_after_nodes() -> None:
    service = _service(RERANKER_ENABLED=True)
    initial_nodes = [_node("node-1", 0.3, "Initial one"), _node("node-2", 0.2, "Initial two"), _node("node-3", 0.1, "Initial three")]
    final_nodes = [initial_nodes[1], initial_nodes[0]]
    service._ensure_index = MagicMock(return_value=MagicMock())
    service._retrieve_nodes = MagicMock(return_value=initial_nodes)
    service._reranker = MagicMock()
    service._reranker.postprocess_nodes.return_value = final_nodes
    service._generate_with_gemini = MagicMock(return_value="answer")

    result = service.evaluate_query_with_trace("question")

    trace = result.retrieval_trace
    assert trace["reranker_enabled"] is True
    assert trace["rerank_input_count"] == 3
    assert trace["rerank_output_count"] == 2
    assert [node["node_id"] for node in trace["retrieved_nodes"]] == ["node-1", "node-2", "node-3"]
    assert [node["node_id"] for node in trace["final_contexts"]] == ["node-2", "node-1"]
    service._reranker.postprocess_nodes.assert_called_once_with(initial_nodes, query_str="question")


def test_generation_trace_uses_the_prompt_sent_to_provider() -> None:
    service = _service(LLM_PROVIDER="deepseek")
    node = _node("node-1", 0.9, "Chunk text")
    service._ensure_index = MagicMock(return_value=MagicMock())
    service._retrieve_nodes = MagicMock(return_value=[node])
    service._generate_with_deepseek = MagicMock(return_value="answer")

    result = service.evaluate_query_with_trace("question")

    generation_input = result.retrieval_trace["generation_input"]
    sent_prompt = service._generate_with_deepseek.call_args.args[0]
    assert generation_input["serialized_context"] in sent_prompt
    assert generation_input["system_prompt_hash"].startswith("sha256:")
    assert generation_input["user_prompt_hash"].startswith("sha256:")
    assert result.generation_request.system_prompt == "system instruction"
    assert result.generation_request.user_prompt == sent_prompt


def test_existing_public_methods_keep_their_return_contracts() -> None:
    service = _service()
    pipeline_result = SimpleNamespace(
        answer="answer",
        citations=[Citation(file_name="f.txt", snippet="snippet")],
        retrieved_contexts=[RetrievedContext(file_name="f.txt", text="chunk")],
    )
    service.evaluate_query_with_trace = MagicMock(return_value=pipeline_result)

    assert service.evaluate_query("question") == (
        "answer",
        pipeline_result.citations,
        pipeline_result.retrieved_contexts,
    )
    assert service.chat("question") == ("answer", pipeline_result.citations)
    assert service.evaluate_query_with_trace.call_count == 2


def test_adapter_uses_pipeline_result_without_a_second_rag_call() -> None:
    trace = {"query": "question", "final_contexts": []}
    pipeline_result = SimpleNamespace(
        answer="answer",
        citations=[Citation(file_name="f.txt", snippet="snippet")],
        retrieved_contexts=[RetrievedContext(file_name="f.txt", text="chunk")],
        retrieval_trace=trace,
    )
    rag_service = MagicMock()
    rag_service.evaluate_query_with_trace.return_value = pipeline_result
    adapter = RagEvaluationAdapter(rag_service)

    result = adapter.run("question")

    assert result.response == "answer"
    assert result.retrieved_contexts == ["chunk"]
    assert result.retrieval_trace is trace
    rag_service.evaluate_query_with_trace.assert_called_once_with("question")
    rag_service.evaluate_query.assert_not_called()


def test_generation_input_leak_detection_checks_input_provenance() -> None:
    no_leaks = find_generation_input_leaks(
        input_sources=("system_prompt", "user_input", "retrieved_contexts"),
    )
    reference_leak = find_generation_input_leaks(
        input_sources=("system_prompt", "user_input", "reference"),
    )
    context_leak = find_generation_input_leaks(
        input_sources=("retrieved_contexts", "reference_contexts", "reference_images"),
    )

    assert no_leaks == []
    assert reference_leak == ["reference"]
    assert context_leak == ["reference_contexts", "reference_images"]
