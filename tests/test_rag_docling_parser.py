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

    def test_hierarchical_chunk_sizes_default(self):
        s = _settings()
        assert s.hierarchical_chunk_sizes == [2048, 512, 128]

    def test_hierarchical_chunk_sizes_override(self):
        s = _settings(HIERARCHICAL_CHUNK_SIZES=[1024, 256])
        assert s.hierarchical_chunk_sizes == [1024, 256]

    def test_docstore_table_default(self):
        s = _settings()
        assert s.docstore_table == "rag_docstore"


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
            leaf_nodes, all_nodes = svc._build_docling_nodes()

        mock_simple_reader.assert_not_called()
        reader_instance.lazy_load_data.assert_called_once_with(
            file_path=str(tmp_path / "a.pdf"),
            extra_info={"file_name": "a.pdf", "file_path": str(tmp_path / "a.pdf")},
        )
        assert len(leaf_nodes) == 1
        # docling's own chunks have no extra parent layer: leaf == all
        assert leaf_nodes == all_nodes
        # metadata copied back from the SOURCE relationship for citations
        assert leaf_nodes[0].metadata["file_name"] == "a.pdf"
        # docling's own chunk metadata is preserved, not overwritten
        assert leaf_nodes[0].metadata["heading"] == "Intro"
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
            leaf_nodes, all_nodes = svc._build_docling_nodes()

        mock_simple_reader_cls.assert_called_once_with(input_files=[str(tmp_path / "b.txt")])
        # 1 docling node + at least 1 fallback node (real SentenceSplitter on "hello world")
        assert len(leaf_nodes) >= 2
        # CHUNK_MODE defaults to "sentence" (flat), so fallback leaf == all too
        assert len(all_nodes) == len(leaf_nodes)
        assert svc._docling_doc_count == 2

    def test_empty_dir_returns_no_nodes_without_importing_docling(self, tmp_path):
        svc = _make_service(PDF_PARSER="docling", UPLOAD_DIR=str(tmp_path))

        # No sys.modules patching: if docling/DoclingReader were imported
        # unconditionally, an empty dir would still need them importable.
        leaf_nodes, all_nodes = svc._build_docling_nodes()

        assert leaf_nodes == []
        assert all_nodes == []
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
            leaf_nodes, all_nodes = svc._parse_documents_to_nodes(documents)

        mock_build.assert_called_once()
        mock_parser.get_nodes_from_documents.assert_called_once_with(documents)
        # flat/single-layer modes: leaf_nodes and all_nodes are the same list
        assert leaf_nodes == ["node"]
        assert leaf_nodes is all_nodes

    def test_layout_aware_builds_hierarchical_tree_with_parent_child_relationships(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        long_text = "This is a long paragraph. " * 100  # long enough for 3 real layers
        documents = [Document(text=long_text, metadata={"file_name": "big.md"})]

        leaf_nodes, all_nodes = svc._parse_documents_to_nodes(documents)

        assert len(all_nodes) > len(leaf_nodes) > 0
        assert all(node.metadata.get("file_name") == "big.md" for node in leaf_nodes)
        # every leaf has a real PARENT relationship, and its parent is in all_nodes
        assert all(NodeRelationship.PARENT in node.relationships for node in leaf_nodes)
        parent_ids = {node.relationships[NodeRelationship.PARENT].node_id for node in leaf_nodes}
        assert parent_ids.issubset({n.node_id for n in all_nodes})
        # at least one node has children (root/mid layer)
        assert any(NodeRelationship.CHILD in node.relationships for node in all_nodes)

    def test_layout_aware_short_document_still_produces_all_three_layers(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        documents = [Document(text="Plain text content.", metadata={"file_name": "notes.txt"})]

        leaf_nodes, all_nodes = svc._parse_documents_to_nodes(documents)

        assert len(leaf_nodes) == 1
        assert "Plain text content." in leaf_nodes[0].get_content()
        # even a tiny document is still recursively split into 3 layers
        # (root -> mid -> leaf), it's just that each layer has one node.
        assert len(all_nodes) == 3

    def test_layout_aware_respects_configured_chunk_sizes(self):
        documents = [Document(text="word " * 400, metadata={"file_name": "a.txt"})]

        svc_default = _make_service(CHUNK_MODE="layout_aware")
        default_leaves, _ = svc_default._parse_documents_to_nodes(documents)

        # a much smaller leaf chunk size must produce more (smaller) leaves
        svc_small = _make_service(CHUNK_MODE="layout_aware", HIERARCHICAL_CHUNK_SIZES=[512, 128, 32])
        small_leaves, _ = svc_small._parse_documents_to_nodes(documents)

        assert len(small_leaves) > len(default_leaves)

    def test_layout_aware_mixes_multiple_documents(self):
        svc = _make_service(CHUNK_MODE="layout_aware")
        documents = [
            Document(text="# Title\n\nMarkdown body.", metadata={"file_name": "a.md"}),
            Document(text='{"x": 1}', metadata={"file_name": "b.json"}),
            Document(text="Some plain text.", metadata={"file_name": "c.txt"}),
        ]

        leaf_nodes, all_nodes = svc._parse_documents_to_nodes(documents)

        # each document produces at least one leaf node; none are silently dropped
        leaf_file_names = {node.metadata.get("file_name") for node in leaf_nodes}
        assert leaf_file_names == {"a.md", "b.json", "c.txt"}


# ---------------------------------------------------------------------------
# 4. RagService.rebuild_index (dispatch across PDF_PARSER branches)
# ---------------------------------------------------------------------------

class TestRebuildIndex:
    def test_docling_branch_dispatches_to_build_docling_nodes(self):
        svc = _make_service(PDF_PARSER="docling")

        def fake_build_docling_nodes():
            svc._docling_doc_count = 3
            return ["leaf1", "leaf2"], ["leaf1", "leaf2", "mid1"]

        mock_ctx = MagicMock()
        with patch.object(svc, "_build_docling_nodes", side_effect=fake_build_docling_nodes) as mock_build, \
             patch.object(svc, "reset_index_storage") as mock_reset, \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=[]), \
             patch.object(svc, "_reset_tracked_files"), \
             patch("app.rag.SimpleDirectoryReader") as mock_reader, \
             patch("app.rag.VectorStoreIndex") as mock_index_cls:
            count = svc.rebuild_index()

        mock_build.assert_called_once()
        mock_reset.assert_called_once()
        mock_reader.assert_not_called()
        # ALL nodes (every layer) go to the docstore before the vector index
        # is built from leaf nodes only.
        mock_ctx.docstore.add_documents.assert_called_once_with(
            ["leaf1", "leaf2", "mid1"], allow_update=True
        )
        mock_index_cls.assert_called_once_with(["leaf1", "leaf2"], storage_context=mock_ctx)
        assert count == 3
        assert svc._bm25_nodes == ["leaf1", "leaf2"]

    def test_default_branch_loads_and_parses_documents(self):
        svc = _make_service(PDF_PARSER="default", CHUNK_MODE="sentence")
        documents = [
            Document(text="a", metadata={"file_name": "a.txt"}),
            Document(text="b", metadata={"file_name": "b.txt"}),
        ]
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = documents
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)

        mock_ctx = MagicMock()
        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls), \
             patch.object(svc, "_parse_documents_to_nodes", return_value=(["n1", "n2"], ["n1", "n2"])) as mock_parse, \
             patch.object(svc, "reset_index_storage") as mock_reset, \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=[]), \
             patch.object(svc, "_reset_tracked_files"), \
             patch("app.rag.VectorStoreIndex") as mock_index_cls:
            count = svc.rebuild_index()

        mock_reader_cls.assert_called_once_with(input_dir=str(svc.settings.upload_dir), recursive=True)
        mock_parse.assert_called_once_with(documents)
        mock_reset.assert_called_once()
        mock_ctx.docstore.add_documents.assert_called_once_with(["n1", "n2"], allow_update=True)
        mock_index_cls.assert_called_once_with(["n1", "n2"], storage_context=mock_ctx)
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
        mock_convert = MagicMock(return_value=(converted_dir, {}))

        mock_ctx = MagicMock()
        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls), \
             patch("app.pdf_utils.convert_pdfs_to_markdown_temp", mock_convert), \
             patch.object(svc, "_parse_documents_to_nodes", return_value=(["n1"], ["n1"])) as mock_parse, \
             patch.object(svc, "reset_index_storage"), \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=[]), \
             patch.object(svc, "_reset_tracked_files"), \
             patch("app.rag.VectorStoreIndex") as mock_index_cls, \
             patch("app.rag.shutil.rmtree") as mock_rmtree:
            count = svc.rebuild_index()

        mock_convert.assert_called_once_with(svc.settings.upload_dir, None)
        mock_reader_cls.assert_called_once_with(input_dir=str(converted_dir), recursive=True)
        mock_parse.assert_called_once_with(documents)
        mock_rmtree.assert_called_once_with(converted_dir, ignore_errors=True)
        mock_ctx.docstore.add_documents.assert_called_once_with(["n1"], allow_update=True)
        mock_index_cls.assert_called_once_with(["n1"], storage_context=mock_ctx)
        assert count == 1
