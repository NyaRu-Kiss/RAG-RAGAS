import json
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.schemas import Citation, RetrievedContext
from eval.config import EvalSettings
from eval.runner import run_evaluation


def _settings(tmp_path: Path) -> EvalSettings:
    return EvalSettings(
        EVAL_DATASET_PATH=str(tmp_path / "dataset.jsonl"),
        EVAL_OUTPUT_DIR=str(tmp_path / "reports"),
        EVAL_BASELINE_DIR=str(tmp_path / "baselines"),
        EVAL_JUDGE_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="test-judge-key",
    )


def test_pipeline_failure_writes_structured_artifacts_without_judge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    dataset = Path(settings.eval_dataset_path)
    dataset.write_text(
        json.dumps(
            {
                "id": "local-1",
                "source_sample_id": "source-1",
                "user_input": "question",
                "reference": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failing_service = MagicMock()
    failing_service.evaluate_query_with_trace.side_effect = RuntimeError("pipeline unavailable")

    with patch("eval.runner.build_metric_specs", return_value=[]):
        artifacts = run_evaluation(settings, rag_service=failing_service)

    assert artifacts.manifest["status"] == "completed"
    assert (artifacts.run_dir / "retrieval_traces.jsonl").read_text(encoding="utf-8") == ""
    samples = [json.loads(line) for line in (artifacts.run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert samples == [{"id": "local-1", "source_sample_id": "source-1", "status": "pipeline_failed"}]
    failures = json.loads((artifacts.run_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures["items"] == [
        {
            "source_sample_id": "source-1",
            "stage": "pipeline",
            "error_type": "RuntimeError",
            "message": "pipeline unavailable",
        }
    ]
    assert failures["summary"][0]["count"] == 1


def test_manifest_snapshot_does_not_include_judge_key(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    dataset = Path(settings.eval_dataset_path)
    dataset.write_text(
        json.dumps({"id": "local-1", "user_input": "question", "reference": "answer"}) + "\n",
        encoding="utf-8",
    )
    failing_service = MagicMock()
    failing_service.evaluate_query_with_trace.side_effect = RuntimeError("pipeline unavailable")

    with patch("eval.runner.build_metric_specs", return_value=[]):
        artifacts = run_evaluation(settings, rag_service=failing_service)

    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in artifacts.run_dir.iterdir())
    assert "test-judge-key" not in artifact_text


def test_pipeline_uses_configured_parallel_workers_and_preserves_dataset_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"eval_pipeline_max_workers": 2})
    dataset = Path(settings.eval_dataset_path)
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"local-{index}",
                    "source_sample_id": f"source-{index}",
                    "user_input": f"question-{index}",
                    "reference": "answer",
                }
            )
            for index in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    active = 0
    peak_active = 0
    lock = Lock()

    def evaluate(question: str) -> SimpleNamespace:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        sleep(0.05)
        with lock:
            active -= 1
        return SimpleNamespace(
            answer="answer",
            citations=[Citation(file_name="source.txt", snippet="snippet")],
            retrieved_contexts=[RetrievedContext(file_name="source.txt", text="context")],
            retrieval_trace={"query": question},
            generation_request=None,
        )

    service = MagicMock()
    service.evaluate_query_with_trace.side_effect = evaluate
    with patch("eval.runner.build_metric_specs", return_value=[]):
        artifacts = run_evaluation(settings, rag_service=service)

    assert peak_active == 2
    assert [row["source_sample_id"] for row in artifacts.sample_rows] == ["source-1", "source-2"]
