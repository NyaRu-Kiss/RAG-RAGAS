from functools import lru_cache
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from openai import OpenAI
from sqlalchemy import create_engine, text
from google import genai

from app.config import Settings, get_settings
from app.schemas import Citation, RetrievedContext


EMBED_DIM = 1024
SNIPPET_LENGTH = 220


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._index: VectorStoreIndex | None = None
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

    def _configure_llama_index(self) -> None:
        model_path = self.settings.embed_model_path
        if not model_path.exists():
            model_path = Path(self.settings.embed_model_name)

        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=str(model_path),
            trust_remote_code=True,
        )
        LlamaSettings.llm = MockLLM()

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

    def rebuild_index(self) -> int:
        documents = SimpleDirectoryReader(
            input_dir=str(self.settings.upload_dir),
            recursive=True,
        ).load_data()
        self.reset_index_storage()
        if not documents:
            self._index = None
            return 0

        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context(),
        )
        return len(documents)

    def _build_snippet(self, text: str) -> str:
        snippet = " ".join(text.split())
        if len(snippet) <= SNIPPET_LENGTH:
            return snippet
        return f"{snippet[:SNIPPET_LENGTH].rstrip()}..."

    def _build_context(self, response: Response) -> str:
        parts: list[str] = []
        for idx, node in enumerate(response.source_nodes, start=1):
            metadata = node.node.metadata
            file_name = metadata.get("file_name") or "Unknown"
            page_label = metadata.get("page_label")
            location = f" (page {page_label})" if page_label is not None else ""
            content = self._build_snippet(node.node.get_content())
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

    def evaluate_query(self, message: str) -> tuple[str, list[Citation], list[RetrievedContext]]:
        index = self._ensure_index()
        retriever = index.as_retriever(similarity_top_k=self.settings.top_k)
        source_nodes = retriever.retrieve(message)
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
