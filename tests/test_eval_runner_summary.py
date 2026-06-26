import pandas as pd


def test_only_numeric_series_should_be_aggregated() -> None:
    frame = pd.DataFrame(
        {
            "faithfulness": [0.5, 1.0],
            "response_relevancy": [0.2, 0.4],
            "status": pd.Series(["ok", "timeout"], dtype="string"),
        }
    )

    metrics_summary: dict[str, float] = {}
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        metrics_summary[column] = float(series.mean())

    assert metrics_summary == {
        "faithfulness": 0.75,
        "response_relevancy": 0.30000000000000004,
    }
