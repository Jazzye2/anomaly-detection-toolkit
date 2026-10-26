import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from anomaly_toolkit.explain import _positive_class_values, build_explainer, explain_row


@pytest.fixture
def toy_model_and_data():
    rng = np.random.default_rng(7)
    n = 150
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.choice([True, False], size=n),  # bool dtype, like a one-hot column
    })
    y = (X["a"] + X["b"].astype(float) > 0.5).astype(bool)
    model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0).fit(X, y)
    return model, X


def test_positive_class_values_handles_3d_array():
    """Regression test: newer shap versions return a single 3D array
    (n_samples, n_features, n_classes) instead of a list of per-class
    arrays. Both shapes must resolve to the positive class correctly."""
    arr = np.zeros((2, 4, 2))
    arr[:, :, 1] = 1.0  # positive-class slice is all 1s
    result = _positive_class_values(arr)
    assert result.shape == (2, 4)
    assert (result == 1.0).all()


def test_positive_class_values_handles_list_format():
    neg = np.zeros((2, 4))
    pos = np.ones((2, 4))
    result = _positive_class_values([neg, pos])
    assert (result == 1.0).all()


def test_build_explainer_accepts_bool_and_mixed_dtypes(toy_model_and_data):
    """Regression test: SHAP's compiled tree-walking code expects plain
    float64 arrays. A bool-dtype column (what one-hot encoding and newer
    pandas dtype backends produce) must not crash the explainer."""
    model, X = toy_model_and_data
    explainer = build_explainer(model, X)
    assert explainer is not None


def test_explain_row_returns_sorted_feature_impacts(toy_model_and_data):
    model, X = toy_model_and_data
    explainer = build_explainer(model, X)
    row = X.iloc[[0]]
    pairs = explain_row(explainer, row, top_n=3)

    assert len(pairs) == 3
    magnitudes = [abs(impact) for _, impact in pairs]
    assert magnitudes == sorted(magnitudes, reverse=True)
    feature_names = [name for name, _ in pairs]
    assert set(feature_names).issubset(set(X.columns))
