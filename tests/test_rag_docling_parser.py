"""Unit tests for DOCLING-01: docling PDF/Office parsing + layout_aware chunking."""
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.config import Settings
from llama_index.core.schema import Document, NodeRelationship, TextNode


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


def _make_service(**overrides):
    s = _settings(**overrides)
    with patch("app.rag.HuggingFaceEmbedding"), patch("app.rag.LlamaSettings"):
        from app.rag import RagService
        svc = object.__new__(RagService)
        svc.settings = s
        svc._bm25_nodes = None
        svc._reranker = None
        return svc


# ---------------------------------------------------------------------------
# 1. Config layer
# ---------------------------------------------------------------------------

class TestNewConfigValues:
    def test_pdf_parser_docling_accepted(self):
        s = _settings(PDF_PARSER="docling")
        assert s.pdf_parser == "docling"

    def test_pdf_parser_invalid_rejected(self):
        with pytest.raises(ValidationError):
            _settings(PDF_PARSER="not-a-real-parser")

    def test_chunk_mode_layout_aware_accepted(self):
        s = _settings(CHUNK_MODE="layout_aware")
        assert s.chunk_mode == "layout_aware"

    def test_chunk_mode_invalid_rejected(self):
        with pytest.raises(ValidationError):
            _settings(CHUNK_MODE="not-a-real-mode")


# ---------------------------------------------------------------------------
# 2. RagService._build_docling_nodes
# ---------------------------------------------------------------------------

class TestBuildDoclingNodes:
    def _mock_docling_modules(self, source_documents, produced_nodes):
        mock_reader_instance = MagicMock()
        mock_reader_instance.lazy_load_data.return_value = source_documents
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)
        mock_reader_cls.ExportType = MagicMock(JSON="json")

        mock_node_parser_instance = MagicMock()
        mock_node_parser_instance.get_nodes_from_documents.return_value = produced_nodes
        mock_node_parser_cls = MagicMock(return_value=mock_node_parser_instance)

        return {
            "llama_index.readers.docling": MagicMock(DoclingReader=mock_reader_cls),
            "llama_index.node_parser.docling": MagicMock(DoclingNodeParser=mock_node_parser_cls),
        }, mock_reader_instance

    def test_all_pdf_only_calls_docling_path(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"fake pdf")
        svc = _make_service(PDF_PARSER="docling", UPLOAD_DIR=str(tmp_path))

        source_doc = Document(text="{}", metadata={"file_name": "a.pdf", "file_path": str(tmp_path / "a.pdf")})
        produced_node = TextNode(text="chunk text", metadata={"heading": "Intro"})
        produced_node.relationships[NodeRelationship.SOURCE] = source_doc.as_related_node_info()

        modules, reader_instance = self._mock_docling_modules([source_doc], [produced_node])
        mock_simple_reader = MagicMock()

        with patch("app.rag.SimpleDirectoryReader", mock_simple_reader), \
             patch.dict("sys.modules", modules):
            nodes = svc._build_docling_nodes()

        mock_simple_reader.assert_not_called()
        reader_instance.lazy_load_data.assert_called_once_with(
            file_path=str(tmp_path / "a.pdf"),
            extra_info={"file_name": "a.pdf", "file_path": str(tmp_path / "a.pdf")},
        )
        assert len(nodes) == 1
        # metadata copied back from the SOURCE relationship for citations
        assert nodes[0].metadata["file_name"] == "a.pdf"
        # docling's own chunk metadata is preserved, not overwritten
        assert nodes[0].metadata["heading"] == "Intro"
        assert svc._docling_doc_count == 1

    def test_mixed_pdf_and_txt_merges_both_paths(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"fake pdf")
        (tmp_path / "b.txt").write_text("hello world")
        svc = _make_service(PDF_PARSER="docling", UPLOAD_DIR=str(tmp_path))

        source_doc = Document(text="{}", metadata={"file_name": "a.pdf"})
        produced_node = TextNode(text="chunk", metadata={})
        produced_node.relationships[NodeRelationship.SOURCE] = source_doc.as_related_node_info()
        modules, _ = self._mock_docling_modules([source_doc], [produced_node])

        fallback_doc = Document(text="hello world", metadata={"file_name": "b.txt"})
        mock_simple_reader_instance = MagicMock()
        mock_simple_reader_instance.load_data.return_value = [fallback_doc]
        mock_simple_reader_cls = MagicMock(return_value=mock_simple_reader_instance)

        with patch("app.rag.SimpleDirectoryReader", mock_simple_reader_cls), \
             patch.dict("sys.modules", modules):
            nodes = svc._build_docling_nodes()

        mock_simple_reader_cls.assert_called_once_with(input_files=[str(tmp_path / "b.txt")])
        # 1 docling node + at least 1 fallback node (real SentenceSplitter on "hello world")
        assert len(nodes) >= 2
        assert svc._docling_doc_count == 2

    def test_empty_dir_returns_no_nodes_without_importing_docling(self, tmp_path):
        svc = _make_service(PDF_PARSER="docling", UPLOAD_DIR=str(tmp_path))

        # No sys.modules patching: if docling/DoclingReader were imported
        # unconditionally, an empty dir would still need them importable.
        nodes = svc._build_docling_nodes()

        assert nodes == []
        assert svc._docling_doc_count == 0


# ---------------------------------------------------------------------------
# 3. RagService._parse_documents_to_nodes (layout_aware dispatch)
# ---------------------------------------------------------------------------

class TestParseDocumentsToNodes:
    def test_non_layout_aware_mode_uses_build_node_parser(self):
        svc = _make_service(CHUNK_MODE="sentence")
        documents = [Document(text="hello world", metadata={"file_name": "f.txt"})]

        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = ["node"]
        with patch.object(svc, "_build_node_parser", return_value=mock_parser) as mock_build:
            result = svc._parse_documents_to_nodes(documents)

        mock_build.assert_called_once()
        mock_parser.get_nodes_from_documents.assert_called_once_with(documents)
        assert result == ["node"]

    def test_layout_aware_dispatches_markdown(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        documents = [Document(text="# Heading\n\nBody text", metadata={"file_name": "notes.md"})]

        nodes = svc._parse_documents_to_nodes(documents)

        assert len(nodes) >= 1
        assert nodes[0].metadata.get("file_name") == "notes.md"

    def test_layout_aware_dispatches_json(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        documents = [Document(text='{"a": 1, "b": {"c": 2}}', metadata={"file_name": "data.json"})]

        nodes = svc._parse_documents_to_nodes(documents)

        assert len(nodes) >= 1

    def test_layout_aware_falls_back_to_sentence_splitter_for_other_extensions(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        documents = [Document(text="Plain text content.", metadata={"file_name": "notes.txt"})]

        nodes = svc._parse_documents_to_nodes(documents)

        assert len(nodes) == 1
        assert "Plain text content." in nodes[0].get_content()

    def test_layout_aware_mixes_groups_and_merges_results(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        documents = [
            Document(text="# Title\n\nMarkdown body.", metadata={"file_name": "a.md"}),
            Document(text='{"x": 1}', metadata={"file_name": "b.json"}),
            Document(text="Some plain text.", metadata={"file_name": "c.txt"}),
        ]

        nodes = svc._parse_documents_to_nodes(documents)

        # each document produces at least one node; none are silently dropped
        assert len(nodes) >= 3


# ---------------------------------------------------------------------------
# 4. RagService.rebuild_index (dispatch across PDF_PARSER branches)
# ---------------------------------------------------------------------------

class TestRebuildIndex:
    def test_docling_branch_dispatches_to_build_docling_nodes(self):
        svc = _make_service(PDF_PARSER="docling")

        def fake_build_docling_nodes():
            svc._docling_doc_count = 3
            return ["node1", "node2"]

        with patch.object(svc, "_build_docling_nodes", side_effect=fake_build_docling_nodes) as mock_build, \
             patch.object(svc, "reset_index_storage") as mock_reset, \
             patch.object(svc, "_storage_context", return_value="ctx"), \
             patch("app.rag.SimpleDirectoryReader") as mock_reader, \
             patch("app.rag.VectorStoreIndex") as mock_index_cls:
            count = svc.rebuild_index()

        mock_build.assert_called_once()
        mock_reset.assert_called_once()
        mock_reader.assert_not_called()
        mock_index_cls.assert_called_once_with(["node1", "node2"], storage_context="ctx")
        assert count == 3
        assert svc._bm25_nodes == ["node1", "node2"]

    def test_default_branch_loads_and_parses_documents(self):
        svc = _make_service(PDF_PARSER="default", CHUNK_MODE="sentence")
        documents = [
            Document(text="a", metadata={"file_name": "a.txt"}),
            Document(text="b", metadata={"file_name": "b.txt"}),
        ]
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = documents
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)

        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls), \
             patch.object(svc, "_parse_documents_to_nodes", return_value=["n1", "n2"]) as mock_parse, \
             patch.object(svc, "reset_index_storage") as mock_reset, \
             patch.object(svc, "_storage_context", return_value="ctx"), \
             patch("app.rag.VectorStoreIndex") as mock_index_cls:
            count = svc.rebuild_index()

        mock_reader_cls.assert_called_once_with(input_dir=str(svc.settings.upload_dir), recursive=True)
        mock_parse.assert_called_once_with(documents)
        mock_reset.assert_called_once()
        mock_index_cls.assert_called_once_with(["n1", "n2"], storage_context="ctx")
        assert count == 2
        assert svc._bm25_nodes == ["n1", "n2"]

    def test_no_documents_returns_zero_and_clears_index(self):
        svc = _make_service(PDF_PARSER="default")
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = []
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)

        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls), \
             patch.object(svc, "reset_index_storage") as mock_reset, \
             patch("app.rag.VectorStoreIndex") as mock_index_cls:
            count = svc.rebuild_index()

        mock_reset.assert_called_once()
        mock_index_cls.assert_not_called()
        assert count == 0
        assert svc._index is None
        assert svc._bm25_nodes is None

    def test_pymupdf4llm_branch_converts_then_loads_and_cleans_up(self, tmp_path):
        svc = _make_service(PDF_PARSER="pymupdf4llm")
        converted_dir = tmp_path / "converted"
        converted_dir.mkdir()
        documents = [Document(text="converted md", metadata={"file_name": "a.md"})]

        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = documents
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)
        mock_convert = MagicMock(return_value=converted_dir)

        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls), \
             patch("app.pdf_utils.convert_pdfs_to_markdown_temp", mock_convert), \
             patch.object(svc, "_parse_documents_to_nodes", return_value=["n1"]) as mock_parse, \
             patch.object(svc, "reset_index_storage"), \
             patch.object(svc, "_storage_context", return_value="ctx"), \
             patch("app.rag.VectorStoreIndex") as mock_index_cls, \
             patch("app.rag.shutil.rmtree") as mock_rmtree:
            count = svc.rebuild_index()

        mock_convert.assert_called_once_with(svc.settings.upload_dir)
        mock_reader_cls.assert_called_once_with(input_dir=str(converted_dir), recursive=True)
        mock_parse.assert_called_once_with(documents)
        mock_rmtree.assert_called_once_with(converted_dir, ignore_errors=True)
        mock_index_cls.assert_called_once_with(["n1"], storage_context="ctx")
        assert count == 1
