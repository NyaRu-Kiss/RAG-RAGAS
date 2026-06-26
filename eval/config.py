from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.eval", env_file_encoding="utf-8", extra="ignore")

    eval_dataset_path: Path = Field(default=Path("eval/datasets/rag_eval_v1.jsonl"), alias="EVAL_DATASET_PATH")
    eval_output_dir: Path = Field(default=Path("eval/reports"), alias="EVAL_OUTPUT_DIR")
    eval_baseline_dir: Path = Field(default=Path("eval/baselines"), alias="EVAL_BASELINE_DIR")
    eval_judge_provider: Literal["gemini", "deepseek"] = Field(default="gemini", alias="EVAL_JUDGE_PROVIDER")
    eval_judge_model: str | None = Field(default=None, alias="EVAL_JUDGE_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://api.aicodemirror.com/api/gemini",
        alias="GOOGLE_GEMINI_BASE_URL",
    )
    gemini_model: str = Field(default="gemini-3-flash-preview", alias="GEMINI_MODEL")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")
    eval_batch_size: int = Field(default=4, alias="EVAL_BATCH_SIZE")
    eval_timeout_seconds: int = Field(default=120, alias="EVAL_TIMEOUT_SECONDS")
    eval_max_retries: int = Field(default=2, alias="EVAL_MAX_RETRIES")
    eval_max_workers: int = Field(default=1, alias="EVAL_MAX_WORKERS")
    eval_raise_exceptions: bool = Field(default=False, alias="EVAL_RAISE_EXCEPTIONS")

    @model_validator(mode="after")
    def validate_judge_provider_config(self) -> "EvalSettings":
        if self.eval_judge_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EVAL_JUDGE_PROVIDER=gemini")
        if self.eval_judge_provider == "deepseek" and not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when EVAL_JUDGE_PROVIDER=deepseek")
        return self

    @property
    def active_judge_model(self) -> str:
        if self.eval_judge_model:
            return self.eval_judge_model
        if self.eval_judge_provider == "deepseek":
            return self.deepseek_model
        return self.gemini_model


@lru_cache(maxsize=1)
def get_eval_settings() -> EvalSettings:
    settings = EvalSettings()
    settings.eval_output_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_baseline_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
