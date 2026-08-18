"""Unit tests for OPT-01-04: Reranker, Hybrid Search, Query Transformation."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings


# ---------------------------------------------------------------------------
# Helpers: build a minimal Settings object without touching the filesystem
# ---------------------------------------------------------------------------

def _settings(**overrides) -> Settings:
    base = dict(
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="test-key",
        EMBED_MODEL_NAME="BAAI/bge-m3",
        EMBED_MODEL_PATH="/nonexistent/path",
        UPLOAD_DIR="/tmp/uploads",
        SYSTEM_PROMPT="test",
    )
    base.update(overrides)
    return Settings(**{k: v for k, v in base.items()})


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

class TestNewConfigDefaults:
    def test_reranker_disabled_by_default(self):
        s = _settings()
        assert s.reranker_enabled is False

    def test_reranker_model_name_default(self):
        s = _settings()
        assert s.reranker_model_name == "BAAI/bge-reranker-v2-m3"

    def test_reranker_model_path_none_by_default(self):
        s = _settings()
        assert s.reranker_model_path is None

    def test_retrieval_top_k_default(self):
        s = _settings()
        assert s.retrieval_top_k == 20

    def test_embedding_batch_size_default(self):
        s = _settings()
        assert s.embed_batch_size == 10

    def test_hybrid_search_disabled_by_default(self):
        s = _settings()
        assert s.hybrid_search_enabled is False

    def test_query_transform_mode_none_by_default(self):
        s = _settings()
        assert s.query_transform_mode == "none"

    def test_num_queries_default(self):
        s = _settings()
        assert s.num_queries == 4

    def test_reranker_enabled_via_env(self):
        s = _settings(RERANKER_ENABLED=True)
        assert s.reranker_enabled is True

    def test_hybrid_search_enabled_via_env(self):
        s = _settings(HYBRID_SEARCH_ENABLED=True)
        assert s.hybrid_search_enabled is True

    def test_query_transform_multi_query(self):
        s = _settings(QUERY_TRANSFORM_MODE="multi_query")
        assert s.query_transform_mode == "multi_query"

    def test_query_transform_hyde(self):
        s = _settings(QUERY_TRANSFORM_MODE="hyde")
        assert s.query_transform_mode == "hyde"


# ---------------------------------------------------------------------------
# 2. RagService._fetch_k
# ---------------------------------------------------------------------------

class TestFetchK:
    def _make_service(self, **overrides):
        s = _settings(**overrides)
        with patch("app.rag.HuggingFaceEmbedding"), \
             patch("app.rag.LlamaSettings"):
            from app.rag import RagService
            svc = object.__new__(RagService)
            svc.settings = s
            svc._reranker = None
            svc._bm25_nodes = None
            return svc

    def test_fetch_k_without_reranker_returns_top_k(self):
        svc = self._make_service(TOP_K=5)
        assert svc._fetch_k() == 5

    def test_fetch_k_with_reranker_returns_retrieval_top_k(self):
        svc = self._make_service(RERANKER_ENABLED=True, TOP_K=5, RETRIEVAL_TOP_K=20)
        svc.settings.reranker_enabled = True
        assert svc._fetch_k() == 20


# ---------------------------------------------------------------------------
# 3. RagService._build_snippet / _build_context (no truncation for context)
# ---------------------------------------------------------------------------

class TestSnippetAndContext:
    def _make_service(self):
        s = _settings()
        with patch("app.rag.HuggingFaceEmbedding"), \
             patch("app.rag.LlamaSettings"):
            from app.rag import RagService
            svc = object.__new__(RagService)
            svc.settings = s
            return svc

    def test_build_snippet_truncates_long_text(self):
        svc = self._make_service()
        long_text = "word " * 200
        result = svc._build_snippet(long_text)
        assert result.endswith("...")
        assert len(result) <= 220 + 3  # 220 chars + "..."

    def test_build_snippet_preserves_short_text(self):
        svc = self._make_service()
        short = "hello world"
        assert svc._build_snippet(short) == "hello world"

    def test_build_context_does_not_truncate(self):
        svc = self._make_service()
        long_content = "word " * 300  # well over 220 chars
        node = MagicMock()
        node.node.metadata = {"file_name": "test.pdf", "page_label": "1"}
        node.node.get_content.return_value = long_content

        response = MagicMock()
        response.source_nodes = [node]

        context = svc._build_context(response)
        # Context must contain all words — not truncated at 220
        assert "..." not in context
        assert len(context) > 220

    def test_build_context_normalises_whitespace(self):
        svc = self._make_service()
        node = MagicMock()
        node.node.metadata = {"file_name": "f.txt"}
        node.node.get_content.return_value = "hello   \n\n  world"

        response = MagicMock()
        response.source_nodes = [node]

        context = svc._build_context(response)
        assert "hello world" in context
        assert "\n\n" not in context.split("[1]")[1]


# ---------------------------------------------------------------------------
# 4. _build_retriever: vector-only vs hybrid path
# ---------------------------------------------------------------------------

class TestBuildRetriever:
    def _make_service(self, **overrides):
        s = _settings(**overrides)
        with patch("app.rag.HuggingFaceEmbedding"), \
             patch("app.rag.LlamaSettings"):
            from app.rag import RagService
            svc = object.__new__(RagService)
            svc.settings = s
            svc._bm25_nodes = None
            svc._reranker = None
            return svc

    def test_vector_only_when_hybrid_disabled(self):
        svc = self._make_service()
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        result = svc._build_retriever(mock_index)

        mock_index.as_retriever.assert_called_once_with(similarity_top_k=svc._fetch_k())
        assert result is mock_retriever

    def test_vector_only_when_hybrid_enabled_but_no_bm25_nodes(self):
        svc = self._make_service(HYBRID_SEARCH_ENABLED=True)
        svc._bm25_nodes = None
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = MagicMock()

        result = svc._build_retriever(mock_index)

        # Falls back to plain vector because _bm25_nodes is None
        mock_index.as_retriever.assert_called_once()

    def test_hybrid_retriever_when_bm25_nodes_present(self):
        svc = self._make_service(HYBRID_SEARCH_ENABLED=True)
        svc._bm25_nodes = [MagicMock()]  # non-empty

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = MagicMock()

        mock_bm25_cls = MagicMock()
        mock_fusion_instance = MagicMock()
        mock_fusion_cls = MagicMock(return_value=mock_fusion_instance)

        # Patch the lazy imports inside _build_retriever via sys.modules so
        # the test works regardless of whether the optional packages are installed.
        with patch.dict("sys.modules", {
            "llama_index.retrievers.bm25": MagicMock(BM25Retriever=mock_bm25_cls),
            "llama_index.core.retrievers": MagicMock(QueryFusionRetriever=mock_fusion_cls),
        }):
            result = svc._build_retriever(mock_index)

        assert result is mock_fusion_instance
        # QueryFusionRetriever must be called with num_queries=1 (pure fusion, no
        # extra query generation at this layer).
        call_kwargs = mock_fusion_cls.call_args
        assert call_kwargs.kwargs.get("num_queries") == 1


# ---------------------------------------------------------------------------
# 5. _retrieve_nodes dispatch
# ---------------------------------------------------------------------------

class TestRetrieveNodesDispatch:
    def _make_service(self, **overrides):
        s = _settings(**overrides)
        with patch("app.rag.HuggingFaceEmbedding"), \
             patch("app.rag.LlamaSettings"):
            from app.rag import RagService
            svc = object.__new__(RagService)
            svc.settings = s
            svc._bm25_nodes = None
            svc._reranker = None
            return svc

    def test_no_transform_calls_retriever_directly(self):
        svc = self._make_service(QUERY_TRANSFORM_MODE="none")
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_index.as_retriever.return_value = mock_retriever

        with patch.object(svc, "_build_retriever", return_value=mock_retriever):
            result = svc._retrieve_nodes("test query", mock_index)

        mock_retriever.retrieve.assert_called_once_with("test query")
        assert result == []

    def test_multi_query_wraps_in_fusion_retriever(self):
        svc = self._make_service(QUERY_TRANSFORM_MODE="multi_query", NUM_QUERIES=3)
        mock_index = MagicMock()
        mock_base_retriever = MagicMock()
        mock_fusion = MagicMock()
        mock_fusion.retrieve.return_value = ["node1"]

        with patch.object(svc, "_build_retriever", return_value=mock_base_retriever), \
             patch.dict("sys.modules", {
                 "llama_index.core.retrievers": MagicMock(
                     QueryFusionRetriever=MagicMock(return_value=mock_fusion)
                 )
             }):
            result = svc._retrieve_nodes("test", mock_index)

        mock_fusion.retrieve.assert_called_once_with("test")
        assert result == ["node1"]

    def test_hyde_calls_transform_then_retrieves(self):
        svc = self._make_service(QUERY_TRANSFORM_MODE="hyde")
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = ["hydrated_node"]

        mock_query_bundle = MagicMock()
        mock_hyde = MagicMock(return_value=mock_query_bundle)
        mock_hyde_cls = MagicMock(return_value=mock_hyde)

        with patch.object(svc, "_build_retriever", return_value=mock_retriever), \
             patch.dict("sys.modules", {
                 "llama_index.core.indices.query.query_transform.base": MagicMock(
                     HyDEQueryTransform=mock_hyde_cls
                 ),
                 "llama_index.core.schema": MagicMock(QueryBundle=MagicMock()),
             }):
            result = svc._retrieve_nodes("test", mock_index)

        mock_retriever.retrieve.assert_called_once_with(mock_query_bundle)
        assert result == ["hydrated_node"]
