from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: Literal["gemini", "deepseek"] = Field(default="gemini", alias="LLM_PROVIDER")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://api.aicodemirror.com/api/gemini",
        alias="GOOGLE_GEMINI_BASE_URL",
    )
    gemini_model: str = Field(default="gemini-3-flash-preview", alias="GEMINI_MODEL")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")

    embed_model_name: str = Field(default="BAAI/bge-m3", alias="EMBED_MODEL_NAME")
    embed_model_path: Path = Field(alias="EMBED_MODEL_PATH")

    pg_host: str = Field(default="127.0.0.1", alias="PG_HOST")
    pg_port: int = Field(default=5432, alias="PG_PORT")
    pg_database: str = Field(default="llama_rag", alias="PG_DATABASE")
    pg_user: str = Field(default="postgres", alias="PG_USER")
    pg_password: str = Field(default="postgres", alias="PG_PASSWORD")
    pg_table: str = Field(default="rag_documents", alias="PG_TABLE")

    upload_dir: Path = Field(alias="UPLOAD_DIR")
    top_k: int = Field(default=5, alias="TOP_K")
    system_prompt: str = Field(alias="SYSTEM_PROMPT")

    # --- Reranker (cross-encoder) ---
    # Set RERANKER_ENABLED=true to activate. Requires rebuilding the index so
    # RETRIEVAL_TOP_K candidates are fetched first, then narrowed to TOP_K.
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_NAME")
    reranker_model_path: Path | None = Field(default=None, alias="RERANKER_MODEL_PATH")
    retrieval_top_k: int = Field(default=20, alias="RETRIEVAL_TOP_K")

    # --- Hybrid search (BM25 + vector with RRF fusion) ---
    # BM25 index lives in memory; call /api/index/rebuild after enabling.
    hybrid_search_enabled: bool = Field(default=False, alias="HYBRID_SEARCH_ENABLED")

    # --- Query transformation ---
    # "none"        – disabled (default)
    # "multi_query" – generate NUM_QUERIES variants via LLM, fuse with RRF
    # "hyde"        – generate a hypothetical answer, embed it, then retrieve
    query_transform_mode: Literal["none", "multi_query", "hyde"] = Field(
        default="none", alias="QUERY_TRANSFORM_MODE"
    )
    num_queries: int = Field(default=4, alias="NUM_QUERIES")

    # --- Chunking strategy ---
    # "sentence"        – SentenceSplitter, fixed token size (default)
    # "semantic"        – SemanticSplitterNodeParser, embedding-based boundaries
    # "sentence_window" – SentenceWindowNodeParser, small chunks + window expansion
    # "layout_aware"    – dispatch by file extension: .md -> MarkdownNodeParser,
    #                      .json -> JSONNodeParser, other -> SentenceSplitter.
    #                      Does not apply to PDF/Office files when
    #                      PDF_PARSER=docling (those use docling's own
    #                      HierarchicalChunker regardless of CHUNK_MODE).
    chunk_mode: Literal["sentence", "semantic", "sentence_window", "layout_aware"] = Field(
        default="sentence", alias="CHUNK_MODE"
    )
    semantic_buffer_size: int = Field(default=1, alias="SEMANTIC_BUFFER_SIZE")
    semantic_breakpoint_threshold: float = Field(default=95.0, alias="SEMANTIC_BREAKPOINT_THRESHOLD")
    sentence_window_size: int = Field(default=3, alias="SENTENCE_WINDOW_SIZE")

    # --- PDF parsing ---
    # "default"     – SimpleDirectoryReader built-in PDF parsing
    # "pymupdf4llm" – convert PDF to Markdown via pymupdf4llm before indexing
    # "docling"     – convert PDF/DOCX/PPTX/XLSX via docling (layout-aware,
    #                 hierarchical chunking); slower, better on tables/complex
    #                 layout. Other file types still fall back to
    #                 SimpleDirectoryReader + CHUNK_MODE.
    pdf_parser: Literal["default", "pymupdf4llm", "docling"] = Field(
        default="default", alias="PDF_PARSER"
    )

    @model_validator(mode="after")
    def validate_llm_provider_config(self) -> "Settings":
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        if self.llm_provider == "deepseek" and not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        return self

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def active_llm_model(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return self.gemini_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
