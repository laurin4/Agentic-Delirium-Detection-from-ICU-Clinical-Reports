import pandas as pd

from src.pipeline.evaluate_predictions import _multiclass_confusion, _per_class_metrics


def test_multiclass_confusion_diagonal_perfect():
    df = pd.DataFrame(
        {
            "PatientenID": ["a", "b", "c"],
            "klasse": [0, 1, 2],
            "baseline_reference_class": [0, 1, 2],
        }
    )
    cm = _multiclass_confusion(df)
    assert cm.shape == (3, 3)
    assert cm.loc[0, 0] == 1
    assert cm.loc[1, 1] == 1
    assert cm.loc[2, 2] == 1


def test_per_class_metrics_from_confusion():
    cm = pd.DataFrame(
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        index=[0, 1, 2],
        columns=[0, 1, 2],
    )
    m = _per_class_metrics(cm)
    assert len(m) == 3
    assert m["recall"].tolist() == [1.0, 1.0, 1.0]
