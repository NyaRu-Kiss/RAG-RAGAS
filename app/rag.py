import hashlib
import logging
import shutil
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from openai import OpenAI
from sqlalchemy import create_engine, text
from google import genai

from app.config import Settings, get_settings
from app.schemas import Citation, RetrievedContext

logger = logging.getLogger(__name__)


EMBED_DIM = 1024
SNIPPET_LENGTH = 220
DOCLING_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}


def _doc_id_for_relpath(relpath: str) -> str:
    """Deterministic ref_doc_id derived from a file's path relative to
    upload_dir (NOT its content), so re-indexing the same file after a
    content change reuses the same id. That lets update_index() cleanly
    replace a file's nodes via index.delete_ref_doc() + reinsert instead of
    accumulating duplicate/orphaned vectors on every edit.
    """
    return f"file:{hashlib.sha256(relpath.encode('utf-8')).hexdigest()}"


@dataclass
class IndexUpdateStats:
    """Summary of an incremental update_index() run."""

    added: int
    updated: int
    removed: int
    unchanged: int

    @property
    def total_changed(self) -> int:
        return self.added + self.updated + self.removed


@dataclass(frozen=True)
class GenerationRequest:
    """The exact prompts and context passed to the configured LLM provider."""

    system_prompt: str
    user_prompt: str
    serialized_context: str
    input_sources: tuple[str, ...] = ("system_prompt", "user_input", "retrieved_contexts")


@dataclass
class RAGPipelineResult:
    """One RAG execution plus the trace required by offline evaluation."""

    answer: str
    citations: list[Citation]
    retrieved_contexts: list[RetrievedContext]
    retrieval_trace: dict[str, Any]
    generation_request: GenerationRequest

ANSWER_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the retrieved context below.

Rules:
1. Cite your sources inline using bracketed numbers, e.g. [1], [2].
2. If multiple sources support a point, cite all relevant ones, e.g. [1][3].
3. Think step by step: first identify which context sections are relevant, then compose your answer.
4. If the retrieved context does NOT contain enough information to answer, respond with exactly:
   "根据检索到的资料，无法回答该问题。"

{history}Question:
{question}

Retrieved context:
{context}"""


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._index: VectorStoreIndex | None = None
        self._bm25_nodes: list | None = None  # in-memory nodes for BM25 retriever
        self._reranker = None
        self._retrieval_trace_local = threading.local()

        self._gemini_client: genai.Client | None = None
        self._deepseek_client: OpenAI | None = None
        if self.settings.llm_provider == "gemini":
            self._gemini_client = genai.Client(
                api_key=self.settings.gemini_api_key,
                http_options={"base_url": self.settings.gemini_base_url},
            )
        else:
            self._deepseek_client = OpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                timeout=self.settings.llm_timeout_seconds,
            )

        self._configure_llama_index()

        if self.settings.reranker_enabled:
            self._reranker = self._build_reranker()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _configure_llama_index(self) -> None:
        model_path = self.settings.embed_model_path
        if not model_path.exists():
            model_path = Path(self.settings.embed_model_name)

        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=str(model_path),
            trust_remote_code=True,
            embed_batch_size=self.settings.embed_batch_size,
        )
        # Use a real LLM for query transformation features; otherwise MockLLM
        # keeps startup fast and avoids unnecessary API calls.
        if self.settings.query_transform_mode != "none":
            LlamaSettings.llm = self._build_llamaindex_llm()
        else:
            LlamaSettings.llm = MockLLM()

    def _build_llamaindex_llm(self):
        """OpenAI-compatible LLM wrapper used by query-transform features."""
        from llama_index.llms.openai_like import OpenAILike

        if self.settings.llm_provider == "deepseek":
            return OpenAILike(
                model=self.settings.deepseek_model,
                api_base=self.settings.deepseek_base_url,
                api_key=self.settings.deepseek_api_key,
                is_chat_model=True,
                max_tokens=512,
            )
        return OpenAILike(
            model=self.settings.gemini_model,
            api_base=self.settings.gemini_base_url,
            api_key=self.settings.gemini_api_key,
            is_chat_model=True,
            max_tokens=512,
        )

    def _build_reranker(self):
        """Instantiate cross-encoder reranker from local path or HF model name."""
        from llama_index.postprocessor.sentence_transformer_rerank import SentenceTransformerRerank

        model_path = self.settings.reranker_model_path
        model_name = (
            str(model_path)
            if (model_path is not None and model_path.exists())
            else self.settings.reranker_model_name
        )
        return SentenceTransformerRerank(model=model_name, top_n=self.settings.top_k)

    # ------------------------------------------------------------------
    # Vector store / index
    # ------------------------------------------------------------------

    def _vector_store(self) -> PGVectorStore:
        return PGVectorStore.from_params(
            host=self.settings.pg_host,
            port=str(self.settings.pg_port),
            database=self.settings.pg_database,
            user=self.settings.pg_user,
            password=self.settings.pg_password,
            table_name=self.settings.pg_table,
            embed_dim=EMBED_DIM,
            perform_setup=True,
            use_jsonb=True,
        )

    def _docstore(self):
        """Postgres-backed docstore holding ALL hierarchical layers (root/mid/leaf)
        for CHUNK_MODE=layout_aware, so AutoMergingRetriever can look up parent
        nodes by id at query time. Only leaf nodes are embedded into the vector
        store, but the docstore needs every layer.
        """
        from llama_index.storage.docstore.postgres import PostgresDocumentStore

        return PostgresDocumentStore.from_params(
            host=self.settings.pg_host,
            port=str(self.settings.pg_port),
            database=self.settings.pg_database,
            user=self.settings.pg_user,
            password=self.settings.pg_password,
            table_name=self.settings.docstore_table,
            perform_setup=True,
            use_jsonb=True,
        )

    def _storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(
            vector_store=self._vector_store(),
            docstore=self._docstore(),
        )

    def _ensure_index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store(),
            )
        self._hydrate_bm25_nodes()
        return self._index

    def reset_index_storage(self) -> None:
        engine = create_engine(self.settings.postgres_dsn)
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "data_{self.settings.pg_table}"'))
            conn.execute(text(f'DROP TABLE IF EXISTS "data_{self.settings.docstore_table}"'))
            conn.execute(text(f'DROP TABLE IF EXISTS "{self.settings.file_index_table}"'))

    # ------------------------------------------------------------------
    # BM25 in-memory corpus (hybrid search)
    # ------------------------------------------------------------------

    def _hydrate_bm25_nodes(self) -> None:
        """Populate the in-memory BM25 corpus from the Postgres docstore if
        it isn't already loaded. rebuild_index()/update_index() keep
        self._bm25_nodes current while the process is alive, but it starts
        out None after a restart — without this, hybrid search would
        silently run against whatever partial corpus update_index() has
        added so far instead of the full indexed set.
        """
        if not self.settings.hybrid_search_enabled or self._bm25_nodes is not None:
            return
        docstore = self._docstore()
        ref_doc_infos = docstore.get_all_ref_doc_info() or {}
        node_ids = [node_id for info in ref_doc_infos.values() for node_id in info.node_ids]
        if not node_ids:
            self._bm25_nodes = []
            return
        nodes = docstore.get_nodes(node_ids)
        if self.settings.chunk_mode == "layout_aware":
            from llama_index.core.node_parser import get_leaf_nodes

            nodes = get_leaf_nodes(nodes)
        self._bm25_nodes = nodes

    def _bm25_drop_ref_doc(self, doc_id: str) -> None:
        """Remove all nodes belonging to `doc_id` from the in-memory BM25
        corpus (mirrors index.delete_ref_doc on the vector store/docstore
        side) when a file is changed or removed during update_index().
        """
        if self._bm25_nodes is None:
            return
        self._bm25_nodes = [node for node in self._bm25_nodes if node.ref_doc_id != doc_id]

    def _bm25_add_nodes(self, nodes: list) -> None:
        if not self.settings.hybrid_search_enabled:
            return
        if self._bm25_nodes is None:
            self._hydrate_bm25_nodes()
        self._bm25_nodes = (self._bm25_nodes or []) + list(nodes)

    # ------------------------------------------------------------------
    # Per-file tracking table (content hash + doc_id) for incremental updates
    # ------------------------------------------------------------------

    def _engine(self):
        return create_engine(self.settings.postgres_dsn)

    def _ensure_file_index_table(self) -> None:
        engine = self._engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{self.settings.file_index_table}" ('
                    "file_relpath TEXT PRIMARY KEY, "
                    "content_hash TEXT NOT NULL, "
                    "doc_id TEXT NOT NULL, "
                    "node_count INTEGER NOT NULL DEFAULT 0, "
                    "indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                    ")"
                )
            )

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _collect_file_records(
        self, paths: list[Path], leaf_nodes: list, hashes: dict[str, str] | None = None
    ) -> list[dict]:
        """Build tracking-table rows for `paths`, pairing each file's content
        hash + deterministic doc_id with how many leaf nodes it contributed
        (looked up from leaf_nodes via ref_doc_id).
        """
        upload_dir = self.settings.upload_dir
        node_counts: dict[str, int] = {}
        for node in leaf_nodes:
            doc_id = node.ref_doc_id
            if doc_id:
                node_counts[doc_id] = node_counts.get(doc_id, 0) + 1

        records = []
        for path in paths:
            relpath = path.resolve().relative_to(upload_dir.resolve()).as_posix()
            doc_id = _doc_id_for_relpath(relpath)
            content_hash = hashes[relpath] if hashes is not None else self._hash_file(path)
            records.append(
                {
                    "file_relpath": relpath,
                    "content_hash": content_hash,
                    "doc_id": doc_id,
                    "node_count": node_counts.get(doc_id, 0),
                }
            )
        return records

    def _reset_tracked_files(self, records: list[dict]) -> None:
        """Replace the entire tracking table (used by a full rebuild)."""
        self._ensure_file_index_table()
        engine = self._engine()
        with engine.begin() as conn:
            conn.execute(text(f'TRUNCATE TABLE "{self.settings.file_index_table}"'))
            if records:
                conn.execute(
                    text(
                        f'INSERT INTO "{self.settings.file_index_table}" '
                        "(file_relpath, content_hash, doc_id, node_count) "
                        "VALUES (:file_relpath, :content_hash, :doc_id, :node_count)"
                    ),
                    records,
                )

    def _load_tracked_files(self) -> dict[str, dict]:
        self._ensure_file_index_table()
        engine = self._engine()
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT file_relpath, content_hash, doc_id, node_count "
                    f'FROM "{self.settings.file_index_table}"'
                )
            ).mappings().all()
        return {row["file_relpath"]: dict(row) for row in rows}

    def _apply_tracked_file_changes(self, upserts: list[dict], removed_relpaths: list[str]) -> None:
        """Incrementally patch the tracking table (used by update_index)."""
        self._ensure_file_index_table()
        engine = self._engine()
        table = self.settings.file_index_table
        with engine.begin() as conn:
            if removed_relpaths:
                conn.execute(
                    text(f'DELETE FROM "{table}" WHERE file_relpath = ANY(:relpaths)'),
                    {"relpaths": removed_relpaths},
                )
            if upserts:
                conn.execute(
                    text(
                        f'INSERT INTO "{table}" (file_relpath, content_hash, doc_id, node_count) '
                        "VALUES (:file_relpath, :content_hash, :doc_id, :node_count) "
                        "ON CONFLICT (file_relpath) DO UPDATE SET "
                        "content_hash = EXCLUDED.content_hash, "
                        "doc_id = EXCLUDED.doc_id, "
                        "node_count = EXCLUDED.node_count, "
                        "indexed_at = now()"
                    ),
                    upserts,
                )

    def _stamp_document_ids(self, documents: list) -> None:
        """Assign each Document a deterministic doc_id derived from its path
        relative to upload_dir (see _doc_id_for_relpath). Documents without
        file_path metadata, or living outside upload_dir, are left with
        their default (random) id and simply won't be precisely replaceable
        by a future incremental update.
        """
        upload_dir = self.settings.upload_dir
        for document in documents:
            file_path = document.metadata.get("file_path")
            if not file_path:
                continue
            try:
                relpath = Path(file_path).resolve().relative_to(upload_dir.resolve()).as_posix()
            except ValueError:
                continue
            document.doc_id = _doc_id_for_relpath(relpath)

    def _build_node_parser(self):
        """Return the node parser configured by CHUNK_MODE."""
        if self.settings.chunk_mode == "semantic":
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            return SemanticSplitterNodeParser(
                embed_model=LlamaSettings.embed_model,
                buffer_size=self.settings.semantic_buffer_size,
                breakpoint_percentile_threshold=self.settings.semantic_breakpoint_threshold,
            )
        if self.settings.chunk_mode == "sentence_window":
            from llama_index.core.node_parser import SentenceWindowNodeParser

            return SentenceWindowNodeParser.from_defaults(
                window_size=self.settings.sentence_window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
        return SentenceSplitter()

    def _parse_documents_to_nodes(self, documents: list) -> tuple[list, list]:
        """Turn loaded Documents into (leaf_nodes, all_nodes) per CHUNK_MODE.

        layout_aware builds a true 3-layer parent/child tree via
        HierarchicalNodeParser (root -> mid -> leaf, sizes from
        HIERARCHICAL_CHUNK_SIZES). Only leaf_nodes are meant to be
        embedded/BM25-indexed; all_nodes (every layer, with real
        NodeRelationship.PARENT/CHILD links) must still reach the docstore so
        AutoMergingRetriever can merge leaves back into their parent at query
        time. The other three CHUNK_MODE values are flat/single-layer, so
        leaf_nodes and all_nodes are the same list.
        """
        if self.settings.chunk_mode != "layout_aware":
            nodes = self._build_node_parser().get_nodes_from_documents(documents)
            return nodes, nodes

        from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.settings.hierarchical_chunk_sizes
        )
        all_nodes = parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)
        return leaf_nodes, all_nodes

    def _build_docling_nodes(self, paths: list[Path] | None = None) -> tuple[list, list]:
        """Docling branch: PDF/Office files go through docling's structured
        parsing + HierarchicalChunker (CHUNK_MODE does not apply to them;
        docling's own chunks are already fine-grained, so they're treated as
        leaves with no extra parent layer). Everything else falls back to
        SimpleDirectoryReader + _parse_documents_to_nodes() (CHUNK_MODE,
        including layout_aware). Returns (leaf_nodes, all_nodes) merged
        across both batches.

        When `paths` is given, only those files are processed — used by
        update_index() for incremental updates; a full rebuild passes
        paths=None to scan the whole upload_dir.
        """
        upload_dir = self.settings.upload_dir
        if paths is None:
            all_paths = [p for p in upload_dir.rglob("*") if p.is_file()]
        else:
            all_paths = list(paths)
        docling_paths = [p for p in all_paths if p.suffix.lower() in DOCLING_SUFFIXES]
        other_paths = [p for p in all_paths if p.suffix.lower() not in DOCLING_SUFFIXES]

        leaf_nodes: list = []
        all_nodes: list = []
        doc_count = 0

        if docling_paths:
            from llama_index.core.schema import NodeRelationship
            from llama_index.node_parser.docling import DoclingNodeParser
            from llama_index.readers.docling import DoclingReader

            # DoclingReader.lazy_load_data() applies the same extra_info dict
            # to every yielded Document, so file_name must be set per file by
            # calling it once per path rather than passing the whole list.
            reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
            docling_documents = []
            for path in docling_paths:
                docling_documents.extend(
                    reader.lazy_load_data(
                        file_path=str(path),
                        extra_info={"file_name": path.name, "file_path": str(path)},
                    )
                )
            doc_count += len(docling_documents)
            # Stamp a deterministic doc_id BEFORE chunking so the produced
            # nodes' SOURCE relationship (and thus ref_doc_id) carries it.
            self._stamp_document_ids(docling_documents)

            docling_nodes = DoclingNodeParser().get_nodes_from_documents(docling_documents)
            # DoclingNodeParser overwrites node.metadata with docling's own
            # chunk metadata (headings etc.) and does not carry over
            # file_name/file_path from the source Document — copy them back
            # from the SOURCE relationship so citations keep working.
            for node in docling_nodes:
                source_info = node.relationships.get(NodeRelationship.SOURCE)
                if source_info is not None and source_info.metadata:
                    node.metadata.setdefault("file_name", source_info.metadata.get("file_name"))
                    node.metadata.setdefault("file_path", source_info.metadata.get("file_path"))
            leaf_nodes.extend(docling_nodes)
            all_nodes.extend(docling_nodes)

        if other_paths:
            fallback_documents = SimpleDirectoryReader(input_files=[str(p) for p in other_paths]).load_data()
            doc_count += len(fallback_documents)
            self._stamp_document_ids(fallback_documents)
            fallback_leaf, fallback_all = self._parse_documents_to_nodes(fallback_documents)
            leaf_nodes.extend(fallback_leaf)
            all_nodes.extend(fallback_all)

        self._docling_doc_count = doc_count
        return leaf_nodes, all_nodes

    def _load_documents_default(self, paths: list[Path] | None = None) -> list:
        """Load Documents via SimpleDirectoryReader for `paths` (or the whole
        upload_dir when None), honouring PDF_PARSER's optional pymupdf4llm
        pre-conversion step. Does not apply to PDF_PARSER=docling — see
        _build_docling_nodes for that branch.

        Every returned Document's file_path/file_name metadata is normalised
        back to its true location under upload_dir (pymupdf4llm stages
        converted files in a temp dir under different names) and stamped
        with a deterministic doc_id for incremental re-indexing.
        """
        temp_dir: Path | None = None
        name_to_original: dict[str, Path] = {}
        if self.settings.pdf_parser == "pymupdf4llm":
            from app.pdf_utils import convert_pdfs_to_markdown_temp

            temp_dir, name_to_original = convert_pdfs_to_markdown_temp(self.settings.upload_dir, paths)
            load_dir = temp_dir
        else:
            load_dir = self.settings.upload_dir

        try:
            if paths is not None and self.settings.pdf_parser != "pymupdf4llm":
                documents = SimpleDirectoryReader(input_files=[str(p) for p in paths]).load_data()
            else:
                documents = SimpleDirectoryReader(input_dir=str(load_dir), recursive=True).load_data()
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if name_to_original:
            for doc in documents:
                temp_name = Path(doc.metadata.get("file_path", "")).name
                original = name_to_original.get(temp_name)
                if original is not None:
                    doc.metadata["file_path"] = str(original)
                    doc.metadata["file_name"] = original.name

        self._stamp_document_ids(documents)
        return documents

    def rebuild_index(self) -> int:
        if self.settings.pdf_parser == "docling":
            leaf_nodes, all_nodes = self._build_docling_nodes()
            document_count = self._docling_doc_count
        else:
            documents = self._load_documents_default()
            document_count = len(documents)
            leaf_nodes, all_nodes = self._parse_documents_to_nodes(documents) if documents else ([], [])

        self.reset_index_storage()
        if not leaf_nodes:
            self._index = None
            self._bm25_nodes = None
            return 0

        storage_context = self._storage_context()
        # All layers (root/mid/leaf) go to the docstore so AutoMergingRetriever
        # can look parents up by id; only leaf_nodes get embedded below.
        storage_context.docstore.add_documents(all_nodes, allow_update=True)
        self._index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
        )
        self._bm25_nodes = leaf_nodes  # kept in memory for hybrid search

        upload_dir = self.settings.upload_dir
        all_paths = [p for p in upload_dir.rglob("*") if p.is_file()] if upload_dir.exists() else []
        self._reset_tracked_files(self._collect_file_records(all_paths, leaf_nodes))

        return document_count

    def update_index(self) -> IndexUpdateStats:
        """Incrementally sync the index with the current contents of
        upload_dir: only new or content-changed files are loaded/chunked;
        unchanged files are left untouched. Files that were removed from
        upload_dir since the last rebuild/update have their vectors and
        docstore entries pruned. Safe to call even if no rebuild has run
        yet (everything is simply classified as new).
        """
        upload_dir = self.settings.upload_dir
        on_disk = [p for p in upload_dir.rglob("*") if p.is_file()] if upload_dir.exists() else []
        current = {p.resolve().relative_to(upload_dir.resolve()).as_posix(): p for p in on_disk}
        hashes = {relpath: self._hash_file(path) for relpath, path in current.items()}
        tracked = self._load_tracked_files()

        new_relpaths = [r for r in current if r not in tracked]
        changed_relpaths = [
            r for r in current if r in tracked and hashes[r] != tracked[r]["content_hash"]
        ]
        removed_relpaths = [r for r in tracked if r not in current]
        unchanged_count = len(current) - len(new_relpaths) - len(changed_relpaths)

        to_process_relpaths = new_relpaths + changed_relpaths
        to_process = [current[r] for r in to_process_relpaths]
        stale_relpaths = changed_relpaths + removed_relpaths

        index = self._ensure_index()

        # Purge changed/removed files' old nodes from the vector store and
        # docstore (and the in-memory BM25 corpus) before (re)inserting.
        for relpath in stale_relpaths:
            doc_id = tracked[relpath]["doc_id"]
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
            self._bm25_drop_ref_doc(doc_id)

        leaf_nodes: list = []
        if to_process:
            if self.settings.pdf_parser == "docling":
                leaf_nodes, all_nodes = self._build_docling_nodes(to_process)
            else:
                documents = self._load_documents_default(to_process)
                leaf_nodes, all_nodes = self._parse_documents_to_nodes(documents) if documents else ([], [])

            if leaf_nodes:
                storage_context = self._storage_context()
                storage_context.docstore.add_documents(all_nodes, allow_update=True)
                index.insert_nodes(leaf_nodes)
                self._bm25_add_nodes(leaf_nodes)

        upserts = self._collect_file_records(to_process, leaf_nodes, hashes) if to_process else []
        self._apply_tracked_file_changes(upserts, removed_relpaths)

        return IndexUpdateStats(
            added=len(new_relpaths),
            updated=len(changed_relpaths),
            removed=len(removed_relpaths),
            unchanged=unchanged_count,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _fetch_k(self) -> int:
        """Number of candidates to fetch before reranking (or final top_k)."""
        return self.settings.retrieval_top_k if self.settings.reranker_enabled else self.settings.top_k

    def _build_retriever(self, index: VectorStoreIndex):
        """Assemble retriever with optional hybrid (BM25 + vector) fusion, and
        (for CHUNK_MODE=layout_aware) an AutoMergingRetriever wrapper that
        merges retrieved leaf nodes back into their parent — via the docstore
        — once enough of that parent's children are present in the results.
        """
        from llama_index.core.retrievers import QueryFusionRetriever

        fetch_k = self._fetch_k()
        vector_retriever = index.as_retriever(similarity_top_k=fetch_k)

        if self.settings.hybrid_search_enabled and self._bm25_nodes:
            from llama_index.retrievers.bm25 import BM25Retriever

            bm25_retriever = BM25Retriever.from_defaults(
                nodes=self._bm25_nodes,
                similarity_top_k=fetch_k,
            )
            # num_queries=1: use the original query only; fusion is RRF across
            # the two retriever result lists, not multi-query generation.
            retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=fetch_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
        else:
            retriever = vector_retriever

        if self.settings.chunk_mode == "layout_aware":
            from llama_index.core.retrievers import AutoMergingRetriever

            return AutoMergingRetriever(retriever, self._storage_context())

        return retriever

    def _retrieve_nodes(self, message: str, index: VectorStoreIndex) -> list[NodeWithScore]:
        """Run retrieval with the configured query transformation strategy."""
        retriever = self._build_retriever(index)
        self._set_retrieval_metadata({
            "transformed_queries": [message],
            "query_transform_ms": 0.0,
        })

        if self.settings.query_transform_mode == "multi_query":
            from llama_index.core.retrievers import QueryFusionRetriever

            # Wrap the (potentially hybrid) retriever so the LLM generates
            # num_queries-1 additional query variants alongside the original.
            multi_retriever = QueryFusionRetriever(
                retrievers=[retriever],
                similarity_top_k=self._fetch_k(),
                num_queries=self.settings.num_queries,
                mode="reciprocal_rerank",
                use_async=False,
            )
            transform_started = perf_counter()
            generated_queries = multi_retriever._get_queries(message)
            self._set_retrieval_metadata({
                "transformed_queries": [message, *(query.query_str for query in generated_queries)],
                "query_transform_ms": (perf_counter() - transform_started) * 1000,
            })
            # QueryFusionRetriever normally generates queries inside retrieve().
            # Reuse this exact generated list so trace capture does not create a
            # second LLM call or change the retrieval inputs.
            multi_retriever._get_queries = lambda _query: generated_queries
            return multi_retriever.retrieve(message)

        if self.settings.query_transform_mode == "hyde":
            from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
            from llama_index.core.schema import QueryBundle

            # Generate a hypothetical answer, embed it, and retrieve.
            # include_original=True fuses results from both the hypothetical
            # document and the original query embeddings.
            hyde = HyDEQueryTransform(include_original=True)
            transform_started = perf_counter()
            query_bundle = hyde(QueryBundle(query_str=message))
            self._set_retrieval_metadata({
                "transformed_queries": list(query_bundle.embedding_strs),
                "query_transform_ms": (perf_counter() - transform_started) * 1000,
            })
            return retriever.retrieve(query_bundle)

        return retriever.retrieve(message)

    # ------------------------------------------------------------------
    # Context & generation
    # ------------------------------------------------------------------

    def _set_retrieval_metadata(self, metadata: dict[str, Any]) -> None:
        trace_local = getattr(self, "_retrieval_trace_local", None)
        if trace_local is None:
            trace_local = threading.local()
            self._retrieval_trace_local = trace_local
        trace_local.metadata = metadata

    def _get_retrieval_metadata(self, message: str) -> dict[str, Any]:
        trace_local = getattr(self, "_retrieval_trace_local", None)
        if trace_local is None:
            return {"transformed_queries": [message], "query_transform_ms": 0.0}
        return getattr(
            trace_local,
            "metadata",
            {"transformed_queries": [message], "query_transform_ms": 0.0},
        )

    def _build_snippet(self, text: str) -> str:
        """Short preview string used in UI citations (not sent to the LLM)."""
        snippet = " ".join(text.split())
        if len(snippet) <= SNIPPET_LENGTH:
            return snippet
        return f"{snippet[:SNIPPET_LENGTH].rstrip()}..."

    def _build_context(self, response: Response) -> str:
        """Build full-text context for the LLM prompt.

        Uses the complete chunk content (whitespace normalised, not truncated)
        so the LLM has access to the entire retrieved passage.  For the
        shorter UI snippet shown in citations use _build_snippet() instead.
        """
        parts: list[str] = []
        for idx, node in enumerate(response.source_nodes, start=1):
            metadata = node.node.metadata
            file_name = metadata.get("file_name") or "Unknown"
            page_label = metadata.get("page_label")
            location = f" (page {page_label})" if page_label is not None else ""
            content = " ".join(node.node.get_content().split())
            parts.append(f"[{idx}] {file_name}{location}\n{content}")
        return "\n\n".join(parts)

    def _build_generation_request(self, message: str, response: Response) -> GenerationRequest:
        context = self._build_context(response)
        user_prompt = (
            "Answer the user's question using the retrieved context when relevant. "
            "If the answer is not grounded in the retrieved context, say so clearly.\n\n"
            f"Question:\n{message}\n\n"
            f"Retrieved context:\n{context or 'No relevant context retrieved.'}"
        )
        return GenerationRequest(
            system_prompt=self.settings.system_prompt,
            user_prompt=user_prompt,
            serialized_context=context,
        )

    def _generate_request(self, request: GenerationRequest) -> str:
        if self.settings.llm_provider == "deepseek":
            return self._generate_with_deepseek(request.user_prompt)
        return self._generate_with_gemini(request.user_prompt)

    def _generate_answer(self, message: str, response: Response) -> str:
        """Backward-compatible helper for callers that only need an answer."""
        return self._generate_request(self._build_generation_request(message, response))

    def _generate_with_gemini(self, user_prompt: str) -> str:
        assert self._gemini_client is not None
        started = perf_counter()
        logger.info("llm_request_start provider=gemini model=%s prompt_chars=%d", self.settings.gemini_model, len(user_prompt))
        completion = self._gemini_client.models.generate_content(
            model=self.settings.gemini_model,
            contents=f"{self.settings.system_prompt}\n\n{user_prompt}",
        )
        response = completion.text or ""
        logger.info("llm_request_success provider=gemini duration_ms=%.1f response_chars=%d", (perf_counter() - started) * 1000, len(response))
        return response

    def _generate_with_deepseek(self, user_prompt: str) -> str:
        assert self._deepseek_client is not None
        started = perf_counter()
        logger.info("llm_request_start provider=deepseek model=%s timeout_seconds=%d prompt_chars=%d", self.settings.deepseek_model, self.settings.llm_timeout_seconds, len(user_prompt))
        try:
            completion = self._deepseek_client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[
                    {"role": "system", "content": self.settings.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            logger.warning("llm_request_failure provider=deepseek duration_ms=%.1f error_type=%s error=%s", (perf_counter() - started) * 1000, type(exc).__name__, str(exc)[:300])
            raise
        content = completion.choices[0].message.content
        if isinstance(content, str):
            response = content
        elif isinstance(content, list):
            response = "".join(
                item.text for item in content if hasattr(item, "text") and isinstance(item.text, str)
            )
        else:
            response = ""
        logger.info("llm_request_success provider=deepseek duration_ms=%.1f response_chars=%d", (perf_counter() - started) * 1000, len(response))
        return response

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _node_snapshot(self, node: NodeWithScore, rank: int) -> dict[str, object | None]:
        metadata = node.node.metadata
        node_id = getattr(node.node, "node_id", None) or getattr(node, "node_id", None)
        page_label = metadata.get("page_label")
        return {
            "rank": rank,
            "node_id": str(node_id) if node_id is not None else None,
            "score": node.score,
            "file_name": metadata.get("file_name") or "Unknown",
            "file_path": metadata.get("file_path"),
            "page_label": str(page_label) if page_label is not None else None,
            "text": node.node.get_content(),
        }

    def _retrieval_mode(self) -> str:
        if self.settings.query_transform_mode == "hyde":
            return "hyde"
        if self.settings.query_transform_mode == "multi_query":
            return "fusion"
        if self.settings.hybrid_search_enabled and self._bm25_nodes:
            return "hybrid"
        return "vector"

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return f"sha256:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _token_count(value: str) -> int | None:
        try:
            from llama_index.core.utils import get_tokenizer

            return len(get_tokenizer()(value))
        except Exception:
            return None

    def evaluate_query_with_trace(self, message: str) -> RAGPipelineResult:
        """Execute the RAG pipeline once and retain its retrieval/generation trace."""
        index = self._ensure_index()
        self._set_retrieval_metadata({"transformed_queries": [message], "query_transform_ms": 0.0})
        retrieve_started = perf_counter()
        source_nodes = self._retrieve_nodes(message, index)
        retrieve_ms = (perf_counter() - retrieve_started) * 1000
        retrieved_nodes = list(source_nodes)

        # Rerank: cross-encoder narrows RETRIEVAL_TOP_K candidates to TOP_K.
        rerank_ms = 0.0
        if self.settings.reranker_enabled and self._reranker is not None:
            rerank_started = perf_counter()
            source_nodes = self._reranker.postprocess_nodes(source_nodes, query_str=message)
            rerank_ms = (perf_counter() - rerank_started) * 1000

        response = Response(response=None, source_nodes=source_nodes)
        citations: list[Citation] = []
        retrieved_contexts: list[RetrievedContext] = []
        for node in response.source_nodes:
            metadata = node.node.metadata
            file_name = metadata.get("file_name") or "Unknown"
            file_path = metadata.get("file_path")
            page_label = metadata.get("page_label")
            content = node.node.get_content()
            citations.append(
                Citation(
                    file_name=file_name,
                    file_path=file_path,
                    page_label=str(page_label) if page_label is not None else None,
                    score=node.score,
                    snippet=self._build_snippet(content),
                )
            )
            retrieved_contexts.append(
                RetrievedContext(
                    file_name=file_name,
                    file_path=file_path,
                    page_label=str(page_label) if page_label is not None else None,
                    score=node.score,
                    text=content,
                )
            )
        generation_request = self._build_generation_request(message, response)
        generation_started = perf_counter()
        answer = self._generate_request(generation_request)
        generation_ms = (perf_counter() - generation_started) * 1000
        retrieval_metadata = self._get_retrieval_metadata(message)

        retrieval_trace: dict[str, Any] = {
            "query": message,
            "transformed_queries": retrieval_metadata["transformed_queries"],
            "retrieval_mode": self._retrieval_mode(),
            "top_k": self.settings.top_k,
            "fetch_k": self._fetch_k(),
            "candidate_count": len(retrieved_nodes),
            "reranker_enabled": self.settings.reranker_enabled and self._reranker is not None,
            "rerank_input_count": len(retrieved_nodes),
            "rerank_output_count": len(source_nodes),
            "retrieved_nodes": [
                self._node_snapshot(node, rank)
                for rank, node in enumerate(retrieved_nodes, start=1)
            ],
            "final_contexts": [
                self._node_snapshot(node, rank)
                for rank, node in enumerate(source_nodes, start=1)
            ],
            "generation_input": {
                "system_prompt_template_id": "app.system_prompt",
                "system_prompt_hash": self._prompt_hash(generation_request.system_prompt),
                "user_prompt_template_id": "app.answer_prompt.v1",
                "user_prompt_hash": self._prompt_hash(generation_request.user_prompt),
                "serialized_context": generation_request.serialized_context,
                "context_token_count": self._token_count(generation_request.serialized_context),
                "request_token_count": self._token_count(
                    f"{generation_request.system_prompt}\n\n{generation_request.user_prompt}"
                ),
                "output_token_count": None,
                "context_operations": ["normalize_whitespace", "preserve_final_order", "serialize"],
            },
            "timings_ms": {
                "query_transform": retrieval_metadata["query_transform_ms"],
                "retrieve": retrieve_ms,
                "rerank": rerank_ms,
                "generation": generation_ms,
            },
        }
        return RAGPipelineResult(
            answer=answer,
            citations=citations,
            retrieved_contexts=retrieved_contexts,
            retrieval_trace=retrieval_trace,
            generation_request=generation_request,
        )

    def evaluate_query(self, message: str) -> tuple[str, list[Citation], list[RetrievedContext]]:
        result = self.evaluate_query_with_trace(message)
        return result.answer, result.citations, result.retrieved_contexts

    def chat(self, message: str) -> tuple[str, list[Citation]]:
        result = self.evaluate_query_with_trace(message)
        return result.answer, result.citations


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(get_settings())
