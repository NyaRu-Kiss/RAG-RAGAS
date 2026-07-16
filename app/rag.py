import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

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


EMBED_DIM = 1024
SNIPPET_LENGTH = 220
DOCLING_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}

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

    def _storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self._vector_store())

    def _ensure_index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store(),
            )
        return self._index

    def reset_index_storage(self) -> None:
        engine = create_engine(self.settings.postgres_dsn)
        table_name = f"data_{self.settings.pg_table}"
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))

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

    def _parse_documents_to_nodes(self, documents: list) -> list:
        """Turn loaded Documents into Nodes per CHUNK_MODE.

        layout_aware dispatches per file extension; LlamaIndex's own
        NodeParser handles node relationships/metadata automatically, so
        this only needs to pick the right parser:
          .md   -> MarkdownNodeParser (heading hierarchy)
          .json -> JSONNodeParser (JSON structure)
          other -> SentenceSplitter (safe default, same as CHUNK_MODE=sentence)
        The other three CHUNK_MODE values keep the existing single-parser
        behaviour via _build_node_parser().
        """
        if self.settings.chunk_mode != "layout_aware":
            return self._build_node_parser().get_nodes_from_documents(documents)

        from llama_index.core.node_parser import JSONNodeParser, MarkdownNodeParser

        groups: dict[str, list] = {"md": [], "json": [], "other": []}
        for doc in documents:
            suffix = Path(doc.metadata.get("file_name", "")).suffix.lower()
            key = "md" if suffix == ".md" else "json" if suffix == ".json" else "other"
            groups[key].append(doc)

        nodes: list = []
        if groups["md"]:
            nodes.extend(MarkdownNodeParser().get_nodes_from_documents(groups["md"]))
        if groups["json"]:
            nodes.extend(JSONNodeParser().get_nodes_from_documents(groups["json"]))
        if groups["other"]:
            nodes.extend(SentenceSplitter().get_nodes_from_documents(groups["other"]))
        return nodes

    def _build_docling_nodes(self) -> list:
        """Docling branch: PDF/Office files go through docling's structured
        parsing + HierarchicalChunker (CHUNK_MODE does not apply to them).
        Everything else falls back to SimpleDirectoryReader +
        _parse_documents_to_nodes() (CHUNK_MODE, including layout_aware).
        Both batches of nodes are merged into one index.
        """
        upload_dir = self.settings.upload_dir
        all_paths = [p for p in upload_dir.rglob("*") if p.is_file()]
        docling_paths = [p for p in all_paths if p.suffix.lower() in DOCLING_SUFFIXES]
        other_paths = [p for p in all_paths if p.suffix.lower() not in DOCLING_SUFFIXES]

        nodes: list = []
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
            nodes.extend(docling_nodes)

        if other_paths:
            fallback_documents = SimpleDirectoryReader(input_files=[str(p) for p in other_paths]).load_data()
            doc_count += len(fallback_documents)
            nodes.extend(self._parse_documents_to_nodes(fallback_documents))

        self._docling_doc_count = doc_count
        return nodes

    def rebuild_index(self) -> int:
        if self.settings.pdf_parser == "docling":
            nodes = self._build_docling_nodes()
            document_count = self._docling_doc_count
        else:
            # Optionally pre-process PDFs to Markdown before loading.
            temp_dir: Path | None = None
            if self.settings.pdf_parser == "pymupdf4llm":
                from app.pdf_utils import convert_pdfs_to_markdown_temp
                temp_dir = convert_pdfs_to_markdown_temp(self.settings.upload_dir)
                load_dir = temp_dir
            else:
                load_dir = self.settings.upload_dir

            try:
                documents = SimpleDirectoryReader(
                    input_dir=str(load_dir),
                    recursive=True,
                ).load_data()
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            document_count = len(documents)
            nodes = self._parse_documents_to_nodes(documents) if documents else []

        self.reset_index_storage()
        if not nodes:
            self._index = None
            self._bm25_nodes = None
            return 0

        self._index = VectorStoreIndex(
            nodes,
            storage_context=self._storage_context(),
        )
        self._bm25_nodes = nodes  # kept in memory for hybrid search
        return document_count

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _fetch_k(self) -> int:
        """Number of candidates to fetch before reranking (or final top_k)."""
        return self.settings.retrieval_top_k if self.settings.reranker_enabled else self.settings.top_k

    def _build_retriever(self, index: VectorStoreIndex):
        """Assemble retriever with optional hybrid (BM25 + vector) fusion."""
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
            return QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=fetch_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )

        return vector_retriever

    def _retrieve_nodes(self, message: str, index: VectorStoreIndex) -> list[NodeWithScore]:
        """Run retrieval with the configured query transformation strategy."""
        retriever = self._build_retriever(index)

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
            return multi_retriever.retrieve(message)

        if self.settings.query_transform_mode == "hyde":
            from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
            from llama_index.core.schema import QueryBundle

            # Generate a hypothetical answer, embed it, and retrieve.
            # include_original=True fuses results from both the hypothetical
            # document and the original query embeddings.
            hyde = HyDEQueryTransform(include_original=True)
            query_bundle = hyde(QueryBundle(query_str=message))
            return retriever.retrieve(query_bundle)

        return retriever.retrieve(message)

    # ------------------------------------------------------------------
    # Context & generation
    # ------------------------------------------------------------------

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

    def _generate_answer(self, message: str, response: Response) -> str:
        context = self._build_context(response)
        user_prompt = (
            "Answer the user's question using the retrieved context when relevant. "
            "If the answer is not grounded in the retrieved context, say so clearly.\n\n"
            f"Question:\n{message}\n\n"
            f"Retrieved context:\n{context or 'No relevant context retrieved.'}"
        )
        if self.settings.llm_provider == "deepseek":
            return self._generate_with_deepseek(user_prompt)
        return self._generate_with_gemini(user_prompt)

    def _generate_with_gemini(self, user_prompt: str) -> str:
        assert self._gemini_client is not None
        completion = self._gemini_client.models.generate_content(
            model=self.settings.gemini_model,
            contents=f"{self.settings.system_prompt}\n\n{user_prompt}",
        )
        return completion.text or ""

    def _generate_with_deepseek(self, user_prompt: str) -> str:
        assert self._deepseek_client is not None
        completion = self._deepseek_client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.text for item in content if hasattr(item, "text") and isinstance(item.text, str)
            )
        return ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_query(self, message: str) -> tuple[str, list[Citation], list[RetrievedContext]]:
        index = self._ensure_index()
        source_nodes = self._retrieve_nodes(message, index)

        # Rerank: cross-encoder narrows RETRIEVAL_TOP_K candidates to TOP_K.
        if self.settings.reranker_enabled and self._reranker is not None:
            source_nodes = self._reranker.postprocess_nodes(source_nodes, query_str=message)

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
        answer = self._generate_answer(message, response)
        return answer, citations, retrieved_contexts

    def chat(self, message: str) -> tuple[str, list[Citation]]:
        answer, citations, _ = self.evaluate_query(message)
        return answer, citations


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(get_settings())
