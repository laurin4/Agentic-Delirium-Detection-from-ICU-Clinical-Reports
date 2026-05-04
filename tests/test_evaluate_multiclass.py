import pandas as pd

from src.pipeline.evaluate_predictions import _binary_confusion, _metrics_from_counts


def test_binary_confusion_counts():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 1, 0])
    counts = _binary_confusion(y_true=y_true, y_pred=y_pred)
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_metrics_from_counts():
    metrics = _metrics_from_counts({"tp": 3, "tn": 5, "fp": 1, "fn": 1})
    assert metrics["n_patients"] == 10
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1"] == 0.75
