from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.config import EvalSettings
from eval.ragas_compat import ensure_ragas_import_compat
from eval.runner import _build_judge_llm, run_evaluation


def test_judge_client_receives_timeout_and_retry_settings() -> None:
    ensure_ragas_import_compat()
    settings = EvalSettings(
        EVAL_JUDGE_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="test-key",
        EVAL_TIMEOUT_SECONDS=37,
        EVAL_MAX_RETRIES=4,
    )

    with (
        patch("langchain_openai.ChatOpenAI") as chat_openai,
        patch("ragas.llms.LangchainLLMWrapper", return_value=MagicMock()),
    ):
        _build_judge_llm(settings)

    assert chat_openai.call_args.kwargs["timeout"] == 37
    assert chat_openai.call_args.kwargs["max_retries"] == 4
    assert chat_openai.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_run_evaluation_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    settings = EvalSettings(
        EVAL_DATASET_PATH=str(dataset),
        EVAL_OUTPUT_DIR=str(tmp_path / "reports"),
        EVAL_BASELINE_DIR=str(tmp_path / "baselines"),
        EVAL_JUDGE_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="test-key",
        EVAL_TIMEOUT_SECONDS=120,
        EVAL_MAX_RETRIES=2,
        EVAL_MAX_WORKERS=1,
    )

    with pytest.raises(ValueError, match="contains no samples"):
        run_evaluation(settings)

    assert not (tmp_path / "reports").exists()
