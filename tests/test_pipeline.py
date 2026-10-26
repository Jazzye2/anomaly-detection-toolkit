import numpy as np
import pandas as pd

from anomaly_toolkit.pipeline import run_pipeline


def test_run_pipeline_end_to_end_on_synthetic_data():
    """Smoke test: the whole scoring -> classifier flow should run without
    error on a small synthetic dataset and return a usable result."""
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], size=n),
        "category": rng.choice(["a", "b", "c"], size=n),
        "year": rng.choice([2020, 2021, 2022], size=n),
        "value": rng.normal(loc=10, scale=2, size=n),
    })
    # inject a handful of genuine outliers so there is something to flag
    df.loc[: 10, "value"] = 100

    result = run_pipeline(
        df,
        value_col="value",
        group_cols=["region", "category"],
        additional_feature_cols=["region", "category", "year"],
    )

    assert "model" in result
    assert "threshold_sweep" in result
    assert len(result["threshold_sweep"]) > 0
    assert result["chosen_threshold"] in result["threshold_sweep"]["threshold"].tolist()
    assert len(result["X_test"]) == len(result["y_test"])
