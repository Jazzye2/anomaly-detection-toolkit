"""Independent check that athena_scoring.sql's math matches scoring.py.

This re-implements the SQL's exact steps in pandas (a separate code path
from scoring.py, not just calling it) and compares the result row for row.
No live Athena connection involved, this verifies the formula is right
before anyone spends time wiring up real AWS infrastructure to run it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from anomaly_toolkit.scoring import score_peer_anomalies


def sql_equivalent_scoring(df, value_col, group_cols, min_cohort_size=5, extreme_z=99.0):
    """Line-for-line pandas re-implementation of athena_scoring.sql's CTEs,
    deliberately NOT reusing any code from scoring.py, so this is an
    independent check, not a tautology."""
    out = df.copy()
    x = out[value_col].astype(float)

    group_sum = out.groupby(group_cols)[value_col].transform("sum")
    group_sumsq = out.groupby(group_cols)[value_col].transform(lambda s: (s ** 2).sum())
    cohort_size = out.groupby(group_cols)[value_col].transform("count")

    with np.errstate(divide="ignore", invalid="ignore"):
        loo_mean = (group_sum - x) / (cohort_size - 1)
        loo_var = np.maximum(
            ((group_sumsq - x ** 2) - (cohort_size - 1) * loo_mean ** 2) / (cohort_size - 2),
            0.0,
        )
        z = (x - loo_mean) / np.sqrt(loo_var)

    tied = (loo_var == 0) & (np.abs(x - loo_mean) < 1e-9)
    deviant = (loo_var == 0) & ~(np.abs(x - loo_mean) < 1e-9)
    z = z.where(~tied, 0.0)
    z = z.where(~deviant, np.sign(x - loo_mean) * extreme_z)
    z = z.where(cohort_size >= min_cohort_size, np.nan)

    out["sql_equivalent_zscore"] = z
    return out


def main():
    rng = np.random.default_rng(99)
    n = 2000
    df = pd.DataFrame({
        "specialty": rng.choice(["cardiology", "family_medicine", "oncology"], size=n),
        "region": rng.choice(["northeast", "south", "midwest", "west"], size=n),
        "value": rng.normal(loc=50, scale=10, size=n),
    })
    df.loc[:5, "value"] = 500  # a handful of genuine outliers

    python_result = score_peer_anomalies(df, "value", ["specialty", "region"], min_cohort_size=5)
    sql_result = sql_equivalent_scoring(df, "value", ["specialty", "region"], min_cohort_size=5)

    python_z = python_result["value_loo_zscore"]
    sql_z = sql_result["sql_equivalent_zscore"]

    both_nan = python_z.isna() & sql_z.isna()
    close_enough = np.isclose(python_z, sql_z, equal_nan=False)
    matches = (both_nan | close_enough).all()

    max_diff = (python_z - sql_z).abs().max()
    print(f"Rows compared: {len(df):,}")
    print(f"All rows match (within floating point tolerance): {matches}")
    print(f"Max absolute difference: {max_diff}")

    if not matches:
        mismatches = df[~(both_nan | close_enough)]
        print(f"\n{len(mismatches)} mismatched rows, first 5:")
        print(mismatches.head())
        sys.exit(1)


if __name__ == "__main__":
    main()
