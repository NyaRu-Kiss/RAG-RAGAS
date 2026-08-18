import math

from eval.metrics import MetricSpec
from eval.runner import metric_outcomes, summarize_metric_outcomes


def test_metric_outcomes_keep_nan_and_failures_out_of_scores() -> None:
    specs = [MetricSpec("faithfulness", object()), MetricSpec("factual_correctness", object())]
    outcomes = metric_outcomes(
        specs,
        {"faithfulness": 0.8, "factual_correctness": math.nan},
        failed_metrics={"response_relevancy": "timeout"},
    )

    assert outcomes["faithfulness"] == {"status": "scored", "score": 0.8}
    assert outcomes["factual_correctness"]["status"] == "nan"
    assert outcomes["response_relevancy"] == {"status": "failed", "error": "timeout"}
    summary = summarize_metric_outcomes([outcomes])
    assert summary["faithfulness"]["mean"] == 0.8
    assert summary["faithfulness"]["p50"] == 0.8
    assert summary["faithfulness"]["p95"] == 0.8
    assert summary["factual_correctness"]["scored"] == 0
    assert summary["factual_correctness"]["nan"] == 1


def test_generation_metric_specs_exclude_retrieval_metrics() -> None:
    from eval.metrics import build_metric_specs

    specs = build_metric_specs()
    keys = {spec.key for spec in specs}

    assert {"faithfulness", "response_relevancy", "factual_correctness"}.issubset(keys)
    assert "context_precision" not in keys
    assert "context_recall" not in keys
    assert next(spec for spec in specs if spec.key == "response_relevancy").score_key == "answer_relevancy"
    assert next(spec for spec in specs if spec.key == "factual_correctness").score_key == "factual_correctness(mode=f1)"
