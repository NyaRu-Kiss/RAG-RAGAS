"""Unit tests for incremental index updates (update_index) and its
supporting helpers: the file-tracking table, deterministic doc ids, and
the BM25 in-memory corpus sync.
"""
from unittest.mock import MagicMock, patch

from app.config import Settings
from llama_index.core.schema import Document, NodeRelationship, RelatedNodeInfo, TextNode


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
        svc._index = None
        return svc


def _node_with_ref(doc_id: str, text: str = "chunk") -> TextNode:
    node = TextNode(text=text)
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


# ---------------------------------------------------------------------------
# 1. _doc_id_for_relpath determinism
# ---------------------------------------------------------------------------

class TestDocIdForRelpath:
    def test_same_relpath_produces_same_id(self):
        from app.rag import _doc_id_for_relpath

        assert _doc_id_for_relpath("a/b.txt") == _doc_id_for_relpath("a/b.txt")

    def test_different_relpaths_produce_different_ids(self):
        from app.rag import _doc_id_for_relpath

        assert _doc_id_for_relpath("a.txt") != _doc_id_for_relpath("b.txt")

    def test_id_is_content_independent(self):
        # Same relpath must map to the same id regardless of the file's
        # content, so update_index() can replace-on-change without
        # accumulating duplicate vectors for the same logical file.
        from app.rag import _doc_id_for_relpath

        assert _doc_id_for_relpath("a.txt") == _doc_id_for_relpath("a.txt")


# ---------------------------------------------------------------------------
# 2. RagService._stamp_document_ids
# ---------------------------------------------------------------------------

class TestStampDocumentIds:
    def test_stamps_doc_id_from_relpath(self, tmp_path):
        from app.rag import _doc_id_for_relpath

        (tmp_path / "sub").mkdir()
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        doc = Document(text="x", metadata={"file_path": str(tmp_path / "sub" / "a.txt")})

        svc._stamp_document_ids([doc])

        assert doc.doc_id == _doc_id_for_relpath("sub/a.txt")

    def test_leaves_id_untouched_when_file_path_missing(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        doc = Document(text="x", metadata={})
        original_id = doc.doc_id

        svc._stamp_document_ids([doc])

        assert doc.doc_id == original_id

    def test_leaves_id_untouched_when_outside_upload_dir(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path / "uploads"))
        doc = Document(text="x", metadata={"file_path": "/some/other/place/a.txt"})
        original_id = doc.doc_id

        svc._stamp_document_ids([doc])

        assert doc.doc_id == original_id


# ---------------------------------------------------------------------------
# 3. RagService._collect_file_records
# ---------------------------------------------------------------------------

class TestCollectFileRecords:
    def test_builds_records_with_node_counts_from_ref_doc_id(self, tmp_path):
        from app.rag import _doc_id_for_relpath

        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        doc_id = _doc_id_for_relpath("a.txt")
        node = _node_with_ref(doc_id)

        records = svc._collect_file_records([f1], [node])

        assert len(records) == 1
        rec = records[0]
        assert rec["file_relpath"] == "a.txt"
        assert rec["doc_id"] == doc_id
        assert rec["node_count"] == 1
        assert rec["content_hash"] == svc._hash_file(f1)

    def test_counts_multiple_nodes_for_same_doc(self, tmp_path):
        from app.rag import _doc_id_for_relpath

        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        doc_id = _doc_id_for_relpath("a.txt")
        nodes = [_node_with_ref(doc_id), _node_with_ref(doc_id), _node_with_ref("other-doc")]

        records = svc._collect_file_records([f1], nodes)

        assert records[0]["node_count"] == 2

    def test_uses_provided_hashes_without_rehashing(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")

        with patch.object(svc, "_hash_file") as mock_hash:
            records = svc._collect_file_records([f1], [], hashes={"a.txt": "precomputed"})

        mock_hash.assert_not_called()
        assert records[0]["content_hash"] == "precomputed"


# ---------------------------------------------------------------------------
# 4. RagService BM25 in-memory corpus helpers
# ---------------------------------------------------------------------------

class TestBm25Helpers:
    def test_drop_ref_doc_removes_matching_nodes(self):
        svc = _make_service()
        node_a = _node_with_ref("doc-a")
        node_b = _node_with_ref("doc-b")
        svc._bm25_nodes = [node_a, node_b]

        svc._bm25_drop_ref_doc("doc-a")

        assert svc._bm25_nodes == [node_b]

    def test_drop_ref_doc_is_noop_when_not_hydrated(self):
        svc = _make_service()
        svc._bm25_nodes = None

        svc._bm25_drop_ref_doc("doc-a")

        assert svc._bm25_nodes is None

    def test_add_nodes_noop_when_hybrid_disabled(self):
        svc = _make_service(HYBRID_SEARCH_ENABLED=False)
        svc._bm25_nodes = None

        svc._bm25_add_nodes([TextNode(text="x")])

        assert svc._bm25_nodes is None

    def test_add_nodes_appends_when_already_hydrated(self):
        svc = _make_service(HYBRID_SEARCH_ENABLED=True)
        svc._bm25_nodes = [TextNode(text="a")]
        new_node = TextNode(text="b")

        with patch.object(svc, "_hydrate_bm25_nodes") as mock_hydrate:
            svc._bm25_add_nodes([new_node])

        mock_hydrate.assert_not_called()
        assert svc._bm25_nodes[-1] is new_node
        assert len(svc._bm25_nodes) == 2

    def test_add_nodes_hydrates_first_when_corpus_missing(self):
        svc = _make_service(HYBRID_SEARCH_ENABLED=True)
        svc._bm25_nodes = None
        new_node = TextNode(text="b")

        def fake_hydrate():
            svc._bm25_nodes = []

        with patch.object(svc, "_hydrate_bm25_nodes", side_effect=fake_hydrate) as mock_hydrate:
            svc._bm25_add_nodes([new_node])

        mock_hydrate.assert_called_once()
        assert svc._bm25_nodes == [new_node]


# ---------------------------------------------------------------------------
# 5. RagService._load_documents_default (pymupdf4llm metadata restoration)
# ---------------------------------------------------------------------------

class TestLoadDocumentsDefault:
    def test_pymupdf4llm_restores_original_file_metadata_and_stamps_id(self, tmp_path):
        from app.rag import _doc_id_for_relpath

        svc = _make_service(PDF_PARSER="pymupdf4llm", UPLOAD_DIR=str(tmp_path))
        original_pdf = tmp_path / "report.pdf"
        original_pdf.write_bytes(b"fake pdf")

        temp_dir = tmp_path / "converted"
        temp_dir.mkdir()
        converted_doc = Document(
            text="converted",
            metadata={"file_path": str(temp_dir / "report.md"), "file_name": "report.md"},
        )
        mock_convert = MagicMock(return_value=(temp_dir, {"report.md": original_pdf}))
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [converted_doc]
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)

        with patch("app.pdf_utils.convert_pdfs_to_markdown_temp", mock_convert), \
             patch("app.rag.SimpleDirectoryReader", mock_reader_cls), \
             patch("app.rag.shutil.rmtree") as mock_rmtree:
            documents = svc._load_documents_default()

        assert documents[0].metadata["file_path"] == str(original_pdf)
        assert documents[0].metadata["file_name"] == "report.pdf"
        assert documents[0].doc_id == _doc_id_for_relpath("report.pdf")
        mock_rmtree.assert_called_once_with(temp_dir, ignore_errors=True)

    def test_default_parser_with_explicit_paths_uses_input_files(self, tmp_path):
        from app.rag import _doc_id_for_relpath

        svc = _make_service(PDF_PARSER="default", UPLOAD_DIR=str(tmp_path))
        f = tmp_path / "a.txt"
        f.write_text("hi")
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [
            Document(text="hi", metadata={"file_path": str(f)})
        ]
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)

        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls):
            documents = svc._load_documents_default([f])

        mock_reader_cls.assert_called_once_with(input_files=[str(f)])
        assert documents[0].doc_id == _doc_id_for_relpath("a.txt")

    def test_default_parser_without_paths_scans_whole_upload_dir(self, tmp_path):
        svc = _make_service(PDF_PARSER="default", UPLOAD_DIR=str(tmp_path))
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = []
        mock_reader_cls = MagicMock(return_value=mock_reader_instance)

        with patch("app.rag.SimpleDirectoryReader", mock_reader_cls):
            svc._load_documents_default()

        mock_reader_cls.assert_called_once_with(input_dir=str(tmp_path), recursive=True)


# ---------------------------------------------------------------------------
# 6. RagService.update_index (new/changed/removed/unchanged classification)
# ---------------------------------------------------------------------------

class TestUpdateIndex:
    def test_new_file_is_inserted_and_recorded(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        (tmp_path / "new.txt").write_text("hello")

        mock_index = MagicMock()
        mock_ctx = MagicMock()
        documents = [Document(text="hello", metadata={"file_name": "new.txt"})]
        fake_records = [{"file_relpath": "new.txt", "content_hash": "h", "doc_id": "d", "node_count": 1}]

        with patch.object(svc, "_load_tracked_files", return_value={}), \
             patch.object(svc, "_hash_file", return_value="hash-new"), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_load_documents_default", return_value=documents) as mock_load, \
             patch.object(svc, "_parse_documents_to_nodes", return_value=(["n1"], ["n1"])) as mock_parse, \
             patch.object(svc, "_bm25_add_nodes") as mock_bm25_add, \
             patch.object(svc, "_bm25_drop_ref_doc") as mock_bm25_drop, \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=fake_records) as mock_collect, \
             patch.object(svc, "_apply_tracked_file_changes") as mock_apply:
            stats = svc.update_index()

        assert [p.name for p in mock_load.call_args.args[0]] == ["new.txt"]
        mock_parse.assert_called_once_with(documents)
        mock_ctx.docstore.add_documents.assert_called_once_with(["n1"], allow_update=True)
        mock_index.insert_nodes.assert_called_once_with(["n1"])
        mock_bm25_add.assert_called_once_with(["n1"])
        mock_bm25_drop.assert_not_called()
        mock_index.delete_ref_doc.assert_not_called()

        collect_paths, collect_leaf_nodes = mock_collect.call_args.args[0], mock_collect.call_args.args[1]
        assert [p.name for p in collect_paths] == ["new.txt"]
        assert collect_leaf_nodes == ["n1"]
        mock_apply.assert_called_once_with(fake_records, [])

        assert stats.added == 1
        assert stats.updated == 0
        assert stats.removed == 0
        assert stats.unchanged == 0
        assert stats.total_changed == 1

    def test_changed_file_purges_old_nodes_then_reinserts(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        (tmp_path / "a.txt").write_text("new content")

        tracked = {
            "a.txt": {"file_relpath": "a.txt", "content_hash": "old-hash", "doc_id": "doc-a", "node_count": 1}
        }
        mock_index = MagicMock()
        mock_ctx = MagicMock()
        documents = [Document(text="new content", metadata={"file_name": "a.txt"})]
        fake_records = [{"file_relpath": "a.txt", "content_hash": "new-hash", "doc_id": "doc-a", "node_count": 1}]

        with patch.object(svc, "_load_tracked_files", return_value=tracked), \
             patch.object(svc, "_hash_file", return_value="new-hash"), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_load_documents_default", return_value=documents), \
             patch.object(svc, "_parse_documents_to_nodes", return_value=(["n1"], ["n1"])), \
             patch.object(svc, "_bm25_add_nodes") as mock_bm25_add, \
             patch.object(svc, "_bm25_drop_ref_doc") as mock_bm25_drop, \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=fake_records), \
             patch.object(svc, "_apply_tracked_file_changes") as mock_apply:
            stats = svc.update_index()

        mock_index.delete_ref_doc.assert_called_once_with("doc-a", delete_from_docstore=True)
        mock_bm25_drop.assert_called_once_with("doc-a")
        mock_index.insert_nodes.assert_called_once_with(["n1"])
        mock_bm25_add.assert_called_once_with(["n1"])
        mock_apply.assert_called_once_with(fake_records, [])

        assert stats.added == 0
        assert stats.updated == 1
        assert stats.removed == 0
        assert stats.unchanged == 0

    def test_removed_file_purges_nodes_without_reprocessing(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        # nothing on disk

        tracked = {
            "gone.txt": {"file_relpath": "gone.txt", "content_hash": "h", "doc_id": "doc-gone", "node_count": 1}
        }
        mock_index = MagicMock()

        with patch.object(svc, "_load_tracked_files", return_value=tracked), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_bm25_drop_ref_doc") as mock_bm25_drop, \
             patch.object(svc, "_load_documents_default") as mock_load, \
             patch.object(svc, "_apply_tracked_file_changes") as mock_apply:
            stats = svc.update_index()

        mock_index.delete_ref_doc.assert_called_once_with("doc-gone", delete_from_docstore=True)
        mock_bm25_drop.assert_called_once_with("doc-gone")
        mock_load.assert_not_called()
        mock_apply.assert_called_once_with([], ["gone.txt"])

        assert stats.added == 0
        assert stats.updated == 0
        assert stats.removed == 1
        assert stats.unchanged == 0

    def test_unchanged_file_is_left_untouched(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        (tmp_path / "same.txt").write_text("stable")

        tracked = {
            "same.txt": {"file_relpath": "same.txt", "content_hash": "stable-hash", "doc_id": "doc-same", "node_count": 1}
        }
        mock_index = MagicMock()

        with patch.object(svc, "_load_tracked_files", return_value=tracked), \
             patch.object(svc, "_hash_file", return_value="stable-hash"), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_load_documents_default") as mock_load, \
             patch.object(svc, "_bm25_drop_ref_doc") as mock_bm25_drop, \
             patch.object(svc, "_apply_tracked_file_changes") as mock_apply:
            stats = svc.update_index()

        mock_index.delete_ref_doc.assert_not_called()
        mock_bm25_drop.assert_not_called()
        mock_load.assert_not_called()
        mock_apply.assert_called_once_with([], [])

        assert stats.added == 0
        assert stats.updated == 0
        assert stats.removed == 0
        assert stats.unchanged == 1

    def test_safe_to_call_before_any_rebuild_classifies_everything_as_new(self, tmp_path):
        # No tracking table rows yet (equivalent to a brand-new deployment):
        # every on-disk file must be treated as "new", not crash.
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        mock_index = MagicMock()
        mock_ctx = MagicMock()

        with patch.object(svc, "_load_tracked_files", return_value={}), \
             patch.object(svc, "_hash_file", return_value="h"), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_load_documents_default", return_value=[]) as mock_load, \
             patch.object(svc, "_bm25_add_nodes"), \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=[]), \
             patch.object(svc, "_apply_tracked_file_changes") as mock_apply:
            stats = svc.update_index()

        assert stats.added == 2
        assert stats.unchanged == 0
        processed_names = sorted(p.name for p in mock_load.call_args.args[0])
        assert processed_names == ["a.txt", "b.txt"]
        mock_apply.assert_called_once_with([], [])

    def test_mixed_new_changed_removed_unchanged(self, tmp_path):
        svc = _make_service(UPLOAD_DIR=str(tmp_path))
        (tmp_path / "new.txt").write_text("new")
        (tmp_path / "changed.txt").write_text("changed-content")
        (tmp_path / "same.txt").write_text("same-content")
        # "gone.txt" is tracked but no longer on disk.

        tracked = {
            "changed.txt": {"file_relpath": "changed.txt", "content_hash": "old-hash", "doc_id": "doc-changed", "node_count": 1},
            "same.txt": {"file_relpath": "same.txt", "content_hash": "same-hash", "doc_id": "doc-same", "node_count": 1},
            "gone.txt": {"file_relpath": "gone.txt", "content_hash": "h", "doc_id": "doc-gone", "node_count": 1},
        }
        mock_index = MagicMock()
        mock_ctx = MagicMock()

        def fake_hash(path):
            return {"new.txt": "new-hash", "changed.txt": "new-hash-2", "same.txt": "same-hash"}[path.name]

        documents = [Document(text="x", metadata={"file_name": "x"})]

        with patch.object(svc, "_load_tracked_files", return_value=tracked), \
             patch.object(svc, "_hash_file", side_effect=fake_hash), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_load_documents_default", return_value=documents) as mock_load, \
             patch.object(svc, "_parse_documents_to_nodes", return_value=(["n1"], ["n1"])), \
             patch.object(svc, "_bm25_add_nodes"), \
             patch.object(svc, "_bm25_drop_ref_doc") as mock_bm25_drop, \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=[]), \
             patch.object(svc, "_apply_tracked_file_changes") as mock_apply:
            stats = svc.update_index()

        assert stats.added == 1
        assert stats.updated == 1
        assert stats.removed == 1
        assert stats.unchanged == 1

        processed_names = sorted(p.name for p in mock_load.call_args.args[0])
        assert processed_names == ["changed.txt", "new.txt"]

        dropped_doc_ids = sorted(call.args[0] for call in mock_bm25_drop.call_args_list)
        assert dropped_doc_ids == ["doc-changed", "doc-gone"]

        delete_calls = sorted(call.args[0] for call in mock_index.delete_ref_doc.call_args_list)
        assert delete_calls == ["doc-changed", "doc-gone"]

        _, removed_arg = mock_apply.call_args.args
        assert removed_arg == ["gone.txt"]

    def test_docling_pdf_parser_dispatches_to_build_docling_nodes(self, tmp_path):
        svc = _make_service(PDF_PARSER="docling", UPLOAD_DIR=str(tmp_path))
        (tmp_path / "new.pdf").write_bytes(b"fake")

        mock_index = MagicMock()
        mock_ctx = MagicMock()

        with patch.object(svc, "_load_tracked_files", return_value={}), \
             patch.object(svc, "_hash_file", return_value="h"), \
             patch.object(svc, "_ensure_index", return_value=mock_index), \
             patch.object(svc, "_build_docling_nodes", return_value=(["n1"], ["n1"])) as mock_build, \
             patch.object(svc, "_load_documents_default") as mock_load_default, \
             patch.object(svc, "_bm25_add_nodes"), \
             patch.object(svc, "_storage_context", return_value=mock_ctx), \
             patch.object(svc, "_collect_file_records", return_value=[]), \
             patch.object(svc, "_apply_tracked_file_changes"):
            svc.update_index()

        mock_load_default.assert_not_called()
        called_paths = mock_build.call_args.args[0]
        assert [p.name for p in called_paths] == ["new.pdf"]
