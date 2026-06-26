from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from eval.adapter import RagEvaluationAdapter
from eval.config import EvalSettings
from eval.dataset import EvalSample, filter_dataset, load_dataset
from eval.metrics import build_metric_specs
from eval.ragas_compat import ensure_ragas_import_compat
from eval.reporting import create_run_dir, write_json, write_jsonl, write_summary_md

if TYPE_CHECKING:
    from app.rag import RagService


@dataclass
class EvalRunArtifacts:
    run_dir: Path
    manifest: dict[str, object]
    summary: dict[str, object]
    sample_rows: list[dict[str, object]]


def _build_judge_llm(settings: EvalSettings):
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI

    run_config = RunConfig(
        timeout=settings.eval_timeout_seconds,
        max_retries=settings.eval_max_retries,
        max_workers=settings.eval_max_workers,
    )

    if settings.eval_judge_provider == "deepseek":
        llm = ChatOpenAI(
            model=settings.active_judge_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
        )
        return LangchainLLMWrapper(llm, run_config=run_config, bypass_n=True)

    llm = ChatOpenAI(
        model=settings.active_judge_model,
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        temperature=0,
    )
    return LangchainLLMWrapper(llm, run_config=run_config, bypass_n=True)


def _build_embeddings(settings: EvalSettings):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from app.config import get_settings

    app_settings = get_settings()
    model_path = app_settings.embed_model_path
    model_name = str(model_path if model_path.exists() else app_settings.embed_model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _sample_to_row(sample: EvalSample, adapter: RagEvaluationAdapter) -> dict[str, object]:
    rag_result = adapter.run(sample.user_input)
    return {
        "id": sample.id,
        "user_input": sample.user_input,
        "reference": sample.reference,
        "reference_contexts": sample.reference_contexts,
        "response": rag_result.response,
        "retrieved_contexts": rag_result.retrieved_contexts,
        "citations": rag_result.citations,
        "tags": sample.tags,
        "difficulty": sample.difficulty,
        "question_type": sample.question_type,
        "images": sample.images,
        "reference_images": sample.reference_images,
    }


def _extract_numeric_metric_scores(frame: pd.DataFrame) -> list[dict[str, float]]:
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    metric_rows: list[dict[str, float]] = []
    for record in frame[numeric_columns].to_dict(orient="records"):
        metric_rows.append({key: float(value) for key, value in record.items()})
    return metric_rows


def run_evaluation(
    eval_settings: EvalSettings,
    *,
    dataset_path: Path | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    rag_service: "RagService | None" = None,
) -> EvalRunArtifacts:
    from app.config import get_settings
    from app.rag import RagService

    dataset_file = dataset_path or eval_settings.eval_dataset_path
    loaded = load_dataset(dataset_file)
    selected = filter_dataset(loaded, tags=tags, limit=limit)
    run_dir = create_run_dir(eval_settings.eval_output_dir)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    if selected:
        effective_rag_service = rag_service or RagService(get_settings())
        adapter = RagEvaluationAdapter(effective_rag_service)
        for sample in selected:
            try:
                rows.append(_sample_to_row(sample, adapter))
            except Exception as exc:
                failures.append({"id": sample.id, "error": str(exc)})
                if eval_settings.eval_raise_exceptions:
                    raise

    if rows:
        ensure_ragas_import_compat()
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.run_config import RunConfig

        judge_llm = _build_judge_llm(eval_settings)
        judge_embeddings = _build_embeddings(eval_settings)
        metric_specs = build_metric_specs(judge_llm=judge_llm, embeddings=judge_embeddings)
    else:
        metric_specs = build_metric_specs()

    active_metrics = [spec.metric for spec in metric_specs if spec.metric is not None]
    skipped_metrics = {
        spec.key: spec.skip_reason
        for spec in metric_specs
        if spec.metric is None and spec.skip_reason is not None
    }
    metrics_summary: dict[str, object] = {}

    if rows:
        ensure_ragas_import_compat()
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.run_config import RunConfig

        ragas_samples = [
            SingleTurnSample(
                user_input=row["user_input"],
                response=row["response"],
                reference=row["reference"],
                retrieved_contexts=row["retrieved_contexts"],
                reference_contexts=row["reference_contexts"],
            )
            for row in rows
        ]
        result = evaluate(
            dataset=EvaluationDataset(samples=ragas_samples),
            metrics=active_metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(
                timeout=eval_settings.eval_timeout_seconds,
                max_retries=eval_settings.eval_max_retries,
                max_workers=eval_settings.eval_max_workers,
            ),
            raise_exceptions=eval_settings.eval_raise_exceptions,
            batch_size=eval_settings.eval_batch_size,
        )
        frame = result.to_pandas()
        metric_records = _extract_numeric_metric_scores(frame)
        for row, metric_record in zip(rows, metric_records, strict=True):
            row["metric_scores"] = metric_record
        for column in frame.columns:
            if column in {"user_input"}:
                continue
            series = frame[column]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            metrics_summary[column] = float(series.mean())

    app_settings = get_settings()
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_file),
        "sample_count": len(selected),
        "scored_sample_count": len(rows),
        "failed_sample_count": len(failures),
        "metrics": [spec.key for spec in metric_specs],
        "app_config": {
            "top_k": app_settings.top_k,
            "pg_table": app_settings.pg_table,
            "llm_provider": app_settings.llm_provider,
            "llm_model": app_settings.active_llm_model,
        },
        "eval_config": {
            "judge_provider": eval_settings.eval_judge_provider,
            "judge_model": eval_settings.active_judge_model,
            "batch_size": eval_settings.eval_batch_size,
            "timeout_seconds": eval_settings.eval_timeout_seconds,
            "max_retries": eval_settings.eval_max_retries,
            "max_workers": eval_settings.eval_max_workers,
            "raise_exceptions": eval_settings.eval_raise_exceptions,
        },
    }
    summary = {
        "run_id": run_dir.name,
        "dataset_path": str(dataset_file),
        "sample_count": len(selected),
        "failed_sample_count": len(failures),
        "metrics": metrics_summary,
        "skipped_metrics": skipped_metrics,
    }

    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "failures.json", failures)
    write_jsonl(run_dir / "samples.jsonl", rows)
    write_summary_md(run_dir / "summary.md", summary)
    return EvalRunArtifacts(run_dir=run_dir, manifest=manifest, summary=summary, sample_rows=rows)
