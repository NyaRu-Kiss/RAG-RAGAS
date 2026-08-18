from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from eval.adapter import RagEvaluationAdapter, find_generation_input_leaks
from eval.artifacts import canonical_json_sha256, file_sha256, get_git_sha, safe_config_snapshot, source_sample_ids_sha256
from eval.config import EvalSettings
from eval.comparison import compare_runs
from eval.dataset import EvalSample, filter_dataset, load_dataset
from eval.metrics import build_metric_specs
from eval.ragas_compat import ensure_ragas_import_compat
from eval.reporting import AtomicRunWriter

logger = logging.getLogger(__name__)

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

    llm = ChatOpenAI(
        model=settings.active_judge_model,
        api_key=settings.active_judge_api_key,
        base_url=settings.active_judge_base_url,
        temperature=settings.eval_judge_temperature,
        timeout=settings.eval_timeout_seconds,
        max_retries=settings.eval_max_retries,
        extra_body={"thinking": {"type": "disabled"}},
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
        "source_sample_id": sample.source_sample_id,
        "status": "pipeline_succeeded",
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
        "retrieval_trace": rag_result.retrieval_trace,
        "generation_request": rag_result.generation_request,
    }


def _pipeline_failure_row(sample: EvalSample) -> dict[str, object]:
    return {
        "id": sample.id,
        "source_sample_id": sample.source_sample_id,
        "status": "pipeline_failed",
    }


def _pipeline_failure(sample: EvalSample, exc: Exception) -> dict[str, str]:
    return {
        "source_sample_id": sample.source_sample_id,
        "stage": "pipeline",
        "error_type": type(exc).__name__,
        "message": str(exc),
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


def _verify_local_embedding() -> dict[str, str]:
    from app.config import get_settings

    settings = get_settings()
    if not settings.embed_model_path.exists():
        raise FileNotFoundError(f"Configured local embedding model not found: {settings.embed_model_path}")
    return {
        "model": settings.embed_model_name,
        "config_sha256": canonical_json_sha256(
            {"model": settings.embed_model_name, "path": str(settings.embed_model_path)}
        ),
    }


def preflight_generation(eval_settings: EvalSettings, *, dataset_path: Path) -> dict[str, object]:
    """Validate fixed generation data and dependencies before RAG or Judge calls."""
    selected = load_dataset(dataset_path)
    if len(selected) != 20:
        raise ValueError("Generation evaluation requires exactly 20 samples")
    root = dataset_path.parent
    manifest_path = root / "dataset_manifest.json"
    corpus_path = root / "corpus.jsonl"
    state_path = root / "index_state.json"
    if not manifest_path.is_file() or not corpus_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Generation dataset requires manifest, corpus, and index state")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    corpus_hash = file_sha256(corpus_path)
    if manifest.get("corpus_sha256") != corpus_hash or state.get("corpus_sha256") != corpus_hash:
        raise ValueError("Generation index state does not match the prepared corpus hash")
    table_names = manifest.get("eval_tables")
    if not isinstance(table_names, dict) or state.get("storage_tables") != table_names:
        raise ValueError("Generation index state does not match isolated eval storage tables")
    if any(not isinstance(table, str) or not table.startswith("eval_") for table in table_names.values()):
        raise ValueError("Generation dataset contains an unsafe eval storage table")
    expected_ids = manifest.get("generation_source_sample_ids")
    actual_ids = [sample.source_sample_id for sample in selected]
    if expected_ids != actual_ids:
        raise ValueError("Generation source sample IDs do not match dataset manifest")
    embedding = _verify_local_embedding()
    ensure_ragas_import_compat()
    build_metric_specs()
    return {"embedding": embedding, "corpus_sha256": corpus_hash, "index_state": state}


def metric_outcomes(
    metric_specs: list,
    scores: dict[str, float],
    *,
    failed_metrics: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    outcomes: dict[str, dict[str, object]] = {}
    for spec in metric_specs:
        if spec.metric is None:
            outcomes[spec.key] = {"status": "skipped", "reason": spec.skip_reason}
            continue
        value = scores.get(spec.key)
        if value is None:
            outcomes[spec.key] = {"status": "failed", "error": "Judge returned no score"}
        elif not math.isfinite(value):
            outcomes[spec.key] = {"status": "nan"}
        else:
            outcomes[spec.key] = {"status": "scored", "score": value}
    for key, error in (failed_metrics or {}).items():
        outcomes[key] = {"status": "failed", "error": error}
    return outcomes


def summarize_metric_outcomes(rows: list[dict[str, dict[str, object]]]) -> dict[str, dict[str, object]]:
    keys = {key for row in rows for key in row}
    summary: dict[str, dict[str, object]] = {}
    for key in sorted(keys):
        outcomes = [row[key] for row in rows if key in row]
        scores = [float(item["score"]) for item in outcomes if item.get("status") == "scored"]
        counts = {status: sum(item.get("status") == status for item in outcomes) for status in ("scored", "failed", "skipped", "nan")}
        summary[key] = {
            "attempted": len(outcomes) - counts["skipped"],
            **counts,
            "mean": sum(scores) / len(scores) if scores else None,
            "p50": _percentile(scores, 0.50),
            "p95": _percentile(scores, 0.95),
        }
    return summary


def _percentile(scores: list[float], percentile: float) -> float | None:
    if not scores:
        return None
    ordered = sorted(scores)
    index = (len(ordered) - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bucket_metric_outcomes(samples: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    buckets: dict[str, dict[str, list[dict[str, dict[str, object]]]]] = {
        "tag": {},
        "difficulty": {},
        "question_type": {},
    }
    for sample in samples:
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict):
            continue
        labels = {
            "tag": [str(tag) for tag in sample.get("tags", [])],
            "difficulty": [str(sample["difficulty"])] if sample.get("difficulty") else [],
            "question_type": [str(sample["question_type"])] if sample.get("question_type") else [],
        }
        for category, values in labels.items():
            for value in values:
                buckets[category].setdefault(value, []).append(metrics)
    return {
        category: {value: summarize_metric_outcomes(rows) for value, rows in values.items()}
        for category, values in buckets.items()
    }


def summarize_failures(failures: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for failure in failures:
        key = (failure["stage"], failure["error_type"], failure["message"])
        grouped.setdefault(key, []).append(failure["source_sample_id"])
    return [
        {
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "count": len(source_ids),
            "source_sample_ids": sorted(source_ids),
        }
        for (stage, error_type, message), source_ids in sorted(grouped.items())
    ]


def run_evaluation(
    eval_settings: EvalSettings,
    *,
    dataset_path: Path | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    rag_service: "RagService | None" = None,
    require_generation_preflight: bool = False,
    baseline: Path | None = None,
) -> EvalRunArtifacts:
    logger.info("evaluation_start dataset=%s", dataset_path or eval_settings.eval_dataset_path)
    from app.config import get_settings
    from app.rag import ANSWER_PROMPT_TEMPLATE, RagService

    dataset_file = dataset_path or eval_settings.eval_dataset_path
    loaded = load_dataset(dataset_file)
    selected = filter_dataset(loaded, tags=tags, limit=limit)
    preflight: dict[str, object] | None = None
    if require_generation_preflight:
        preflight = preflight_generation(eval_settings, dataset_path=dataset_file)
    writer = AtomicRunWriter(eval_settings.eval_output_dir)
    app_settings = get_settings()
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(Path.cwd()),
        "dataset_path": str(dataset_file.resolve()),
        "dataset_sha256": file_sha256(dataset_file),
        "source_sample_ids_sha256": source_sample_ids_sha256([sample.source_sample_id for sample in selected]),
        "sample_count": len(selected),
        "scored_sample_count": 0,
        "failed_sample_count": 0,
        "metrics": [],
        "pipeline_config": {
            "retrieval": {
                "top_k": app_settings.top_k,
                "fetch_k": app_settings.retrieval_top_k if app_settings.reranker_enabled else app_settings.top_k,
                "hybrid_search_enabled": app_settings.hybrid_search_enabled,
                "query_transform_mode": app_settings.query_transform_mode,
                "num_queries": app_settings.num_queries,
                "reranker_enabled": app_settings.reranker_enabled,
                "reranker_model": app_settings.reranker_model_name,
            },
            "chunking": {
                "chunk_mode": app_settings.chunk_mode,
                "pdf_parser": app_settings.pdf_parser,
                "semantic_buffer_size": app_settings.semantic_buffer_size,
                "semantic_breakpoint_threshold": app_settings.semantic_breakpoint_threshold,
                "sentence_window_size": app_settings.sentence_window_size,
                "hierarchical_chunk_sizes": app_settings.hierarchical_chunk_sizes,
            },
            "generation": {"provider": app_settings.llm_provider, "model": app_settings.active_llm_model},
        },
        "app_config": safe_config_snapshot(app_settings.model_dump(mode="json")),
        "prompt_config": {
            "system_prompt_hash": canonical_json_sha256(app_settings.system_prompt),
            "user_prompt_template_id": "app.answer_prompt.v1",
            "user_prompt_template_hash": canonical_json_sha256(ANSWER_PROMPT_TEMPLATE),
        },
        "filters": {"tags": tags or [], "limit": limit},
        "eval_config": safe_config_snapshot({
            "judge_provider": eval_settings.eval_judge_provider,
            "judge_model": eval_settings.active_judge_model,
            "judge_base_url": eval_settings.active_judge_base_url,
            "batch_size": eval_settings.eval_batch_size,
            "timeout_seconds": eval_settings.eval_timeout_seconds,
            "max_retries": eval_settings.eval_max_retries,
            "max_workers": eval_settings.eval_max_workers,
            "pipeline_max_workers": eval_settings.eval_pipeline_max_workers,
            "raise_exceptions": eval_settings.eval_raise_exceptions,
            "temperature": eval_settings.eval_judge_temperature,
        }),
    }
    if preflight is not None:
        manifest["embedding"] = preflight["embedding"]
        manifest["corpus_sha256"] = preflight["corpus_sha256"]

    sample_rows: list[dict[str, object]] = []
    successful_rows: list[dict[str, object]] = []
    retrieval_traces: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        if selected:
            effective_rag_service = rag_service or RagService(app_settings)
            adapter = RagEvaluationAdapter(effective_rag_service)
            # Load the shared index once before worker threads use the service.
            effective_rag_service._ensure_index()

            def evaluate_sample(sample: EvalSample) -> tuple[EvalSample, dict[str, object] | None, dict[str, str] | None]:
                logger.info("evaluation_sample_start source_sample_id=%s", sample.source_sample_id)
                try:
                    return sample, _sample_to_row(sample, adapter), None
                except Exception as exc:
                    logger.warning("evaluation_sample_failure source_sample_id=%s stage=pipeline error_type=%s error=%s", sample.source_sample_id, type(exc).__name__, str(exc)[:300])
                    return sample, None, _pipeline_failure(sample, exc)

            results: dict[str, tuple[EvalSample, dict[str, object] | None, dict[str, str] | None]] = {}
            with ThreadPoolExecutor(max_workers=eval_settings.eval_pipeline_max_workers) as executor:
                futures = [executor.submit(evaluate_sample, sample) for sample in selected]
                for future in as_completed(futures):
                    sample, row, failure = future.result()
                    results[sample.source_sample_id] = (sample, row, failure)

            # Restore dataset order so artifacts and metric rows remain deterministic.
            for sample in selected:
                _, row, failure = results[sample.source_sample_id]
                if failure is not None:
                    sample_rows.append(_pipeline_failure_row(sample))
                    failures.append(failure)
                    if eval_settings.eval_raise_exceptions:
                        raise RuntimeError(failure["message"])
                    continue
                assert row is not None
                try:
                    retrieval_trace = row.pop("retrieval_trace")
                    generation_request = row.pop("generation_request")
                    if generation_request is not None:
                        leaks = find_generation_input_leaks(
                            input_sources=generation_request.input_sources,
                        )
                        if leaks:
                            row["status"] = "invalid"
                            row["invalid_reason"] = f"Generation input contains {', '.join(leaks)}"
                            sample_rows.append(row)
                            failures.append(
                                {
                                    "source_sample_id": sample.source_sample_id,
                                    "stage": "validation",
                                    "error_type": "ReferenceLeakError",
                                    "message": row["invalid_reason"],
                                }
                            )
                            continue
                    sample_rows.append(row)
                    successful_rows.append(row)
                    logger.info("evaluation_sample_success source_sample_id=%s retrieved_contexts=%d", sample.source_sample_id, len(row["retrieved_contexts"]))
                    retrieval_traces.append(
                        {
                            "id": sample.id,
                            "source_sample_id": sample.source_sample_id,
                            "retrieval_trace": retrieval_trace,
                        }
                    )
                except Exception as exc:
                    logger.warning("evaluation_sample_failure source_sample_id=%s stage=pipeline error_type=%s error=%s", sample.source_sample_id, type(exc).__name__, str(exc)[:300])
                    sample_rows.append(_pipeline_failure_row(sample))
                    failures.append(_pipeline_failure(sample, exc))
                    if eval_settings.eval_raise_exceptions:
                        raise

        if successful_rows:
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

        if successful_rows:
            ragas_samples = [
                SingleTurnSample(
                    user_input=row["user_input"],
                    response=row["response"],
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"],
                    reference_contexts=row["reference_contexts"],
                )
                for row in successful_rows
            ]
            metric_records: list[dict[str, float]] = [{} for _ in successful_rows]
            failed_metrics: list[dict[str, str]] = [{} for _ in successful_rows]
            for spec in metric_specs:
                if spec.metric is None:
                    continue
                logger.info("judge_metric_start metric=%s sample_count=%d", spec.key, len(successful_rows))
                try:
                    result = evaluate(
                        dataset=EvaluationDataset(samples=ragas_samples),
                        metrics=[spec.metric],
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
                    score_rows = _extract_numeric_metric_scores(frame)
                    for record, score_row in zip(metric_records, score_rows, strict=True):
                        if spec.score_key in score_row:
                            record[spec.key] = score_row[spec.score_key]
                    if not any(spec.score_key in score_row for score_row in score_rows):
                        raise ValueError(
                            f"Ragas result column {spec.score_key!r} not found; columns={list(frame.columns)!r}"
                        )
                    logger.info("judge_metric_success metric=%s scored_rows=%d", spec.key, len(score_rows))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("judge_metric_failure metric=%s error_type=%s error=%s", spec.key, type(exc).__name__, str(exc)[:300])
                    for failed in failed_metrics:
                        failed[spec.key] = error
            for row, metric_record, failed in zip(successful_rows, metric_records, failed_metrics, strict=True):
                row["metrics"] = metric_outcomes(metric_specs, metric_record, failed_metrics=failed)
                if any(item["status"] == "scored" for item in row["metrics"].values()):
                    row["status"] = "scored"
                else:
                    row["status"] = "judge_failed"
                for metric_key, error in failed.items():
                    failures.append(
                        {
                            "source_sample_id": str(row["source_sample_id"]),
                            "stage": "judge",
                            "error_type": "JudgeMetricError",
                            "message": f"{metric_key}: {error}",
                        }
                    )
            metrics_summary = summarize_metric_outcomes([row["metrics"] for row in successful_rows])
        elif metric_specs:
            metrics_summary = summarize_metric_outcomes([
                metric_outcomes(metric_specs, {})
            ])

        manifest["scored_sample_count"] = len(successful_rows)
        manifest["failed_sample_count"] = len({failure["source_sample_id"] for failure in failures})
        manifest["metrics"] = [spec.key for spec in metric_specs]
        summary = {
            "run_id": writer.run_id,
            "dataset_path": str(dataset_file.resolve()),
            "sample_count": len(selected),
            "failed_sample_count": manifest["failed_sample_count"],
            "metrics": metrics_summary,
            "skipped_metrics": skipped_metrics,
            "buckets": bucket_metric_outcomes(sample_rows),
            "failures": [
                {"source_sample_id": item["source_sample_id"], "stage": item["stage"], "trace": "retrieval_traces.jsonl"}
                for item in failures
            ],
            "failure_summary": summarize_failures(failures),
        }
        comparison: dict[str, object] | None = None
        if baseline is not None:
            comparison = compare_runs(
                current_manifest=manifest,
                current_samples=sample_rows,
                baseline_dir=baseline,
            )
            summary["comparison"] = comparison
        writer.write_json("summary.json", summary)
        writer.write_jsonl("samples.jsonl", sample_rows)
        writer.write_jsonl("retrieval_traces.jsonl", retrieval_traces)
        writer.write_json("failures.json", {"items": failures, "summary": summarize_failures(failures)})
        if comparison is not None:
            writer.write_json("comparison.json", comparison)
        writer.write_summary_md(summary)
        run_dir = writer.complete(manifest)
        manifest["status"] = "completed"
        manifest["run_id"] = writer.run_id
        logger.info("evaluation_complete run_id=%s successful_samples=%d failed_samples=%d", writer.run_id, len(successful_rows), manifest["failed_sample_count"])
        return EvalRunArtifacts(run_dir=run_dir, manifest=manifest, summary=summary, sample_rows=sample_rows)
    except Exception:
        logger.exception("evaluation_failed processed_samples=%d", len(sample_rows))
        writer.fail(manifest, status="partial" if sample_rows else "failed")
        raise
