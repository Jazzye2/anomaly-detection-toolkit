import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from anomaly_toolkit.classifier import best_threshold, fit_classifier, sweep_thresholds


@pytest.fixture
def toy_classifier_data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + X["b"] > 1).astype(bool)  # a learnable, imbalanced-ish pattern
    return X, y


def test_fit_classifier_trains_and_predicts(toy_classifier_data):
    X, y = toy_classifier_data
    model = fit_classifier(X, y, {"n_estimators": 20, "max_depth": 3})
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({True, False})


def test_sweep_thresholds_returns_one_row_per_threshold(toy_classifier_data):
    X, y = toy_classifier_data
    model = fit_classifier(X, y, {"n_estimators": 20, "max_depth": 3})
    sweep = sweep_thresholds(model, X, y, thresholds=(0.5, 0.3, 0.1))
    assert list(sweep["threshold"]) == [0.5, 0.3, 0.1]
    assert set(sweep.columns) == {"threshold", "precision", "recall", "f1"}


def test_sweep_thresholds_recall_increases_as_threshold_drops(toy_classifier_data):
    """Lowering the threshold should never catch fewer positives, recall
    must be monotonically non-decreasing as the cutoff relaxes."""
    X, y = toy_classifier_data
    model = fit_classifier(X, y, {"n_estimators": 50, "max_depth": 4})
    sweep = sweep_thresholds(model, X, y, thresholds=(0.8, 0.5, 0.2))
    recalls = sweep.sort_values("threshold", ascending=False)["recall"].tolist()
    assert recalls == sorted(recalls)


def test_best_threshold_picks_the_row_with_max_metric():
    sweep = pd.DataFrame({
        "threshold": [0.5, 0.3, 0.1],
        "precision": [0.9, 0.7, 0.4],
        "recall": [0.2, 0.6, 0.9],
        "f1": [0.33, 0.65, 0.55],
    })
    assert best_threshold(sweep, metric="f1") == 0.3
    assert best_threshold(sweep, metric="precision") == 0.5
