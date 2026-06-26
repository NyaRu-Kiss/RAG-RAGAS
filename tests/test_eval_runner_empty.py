from pathlib import Path

from eval.config import EvalSettings
from eval.runner import run_evaluation


def test_run_evaluation_with_empty_dataset(tmp_path: Path) -> None:
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

    artifacts = run_evaluation(settings)

    assert artifacts.summary["sample_count"] == 0
    assert (artifacts.run_dir / "summary.json").exists()
