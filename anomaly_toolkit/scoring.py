"""Peer-group leave-one-out anomaly scoring.

Flags rows whose value is unusual relative to a peer cohort, using
leave-one-out statistics so a row can never inflate the baseline it's
measured against. A plain z-score (mean/std computed over the whole
cohort, including the row itself) lets a genuine outlier in a small
cohort inflate that cohort's own standard deviation and shrink its own
z-score, hiding itself. That effect, "masking," gets stronger the
smaller the cohort. Leave-one-out removes it at the source.
"""
import numpy as np
import pandas as pd


def score_peer_anomalies(df, value_col, group_cols, min_cohort_size=5, extreme_z=99.0):
    """Compute a leave-one-out z-score for value_col within each group_cols cohort.

    Parameters
    ----------
    df : DataFrame
    value_col : str
        The numeric column to screen for anomalies.
    group_cols : list[str]
        Columns that define a row's peer cohort (e.g. region, category, type).
    min_cohort_size : int
        Cohorts smaller than this produce a null score rather than an
        unreliable one computed from too little data.
    extreme_z : float
        Sentinel magnitude assigned when every peer in a cohort reports
        an identical value and this row's value differs from it. That
        case is a 0/0 in the raw formula, not a real "undefined" result,
        a lone disagreement against a unanimous peer group is about as
        anomalous as a single data point can be.

    Returns
    -------
    DataFrame
        A copy of df with three new columns:
          "{value_col}_cohort_size"       : int, how many rows share this cohort
          "{value_col}_loo_zscore"        : float or NaN if unscorable
          "{value_col}_missing_peer_group": bool, True if this row has a
            missing value in one or more group_cols, so it has no defined
            peer group at all. pandas' groupby silently drops such rows
            from every group by default, which otherwise looks identical
            to "cohort too small" in the output, a different and more
            diagnosable problem worth reporting separately.
    """
    out = df.copy()
    x = out[value_col].astype(float)

    missing_group = out[group_cols].isna().any(axis=1)
    out[f"{value_col}_missing_peer_group"] = missing_group

    group_sum = out.groupby(group_cols)[value_col].transform("sum")
    group_sumsq = out.groupby(group_cols)[value_col].transform(lambda s: (s ** 2).sum())
    n = out.groupby(group_cols)[value_col].transform("count")

    with np.errstate(divide="ignore", invalid="ignore"):
        loo_mean = (group_sum - x) / (n - 1)
        loo_var = ((group_sumsq - x ** 2) - (n - 1) * loo_mean ** 2) / (n - 2)
        loo_var = np.maximum(loo_var, 0)

        zero_var = loo_var == 0
        tied = zero_var & np.isclose(x - loo_mean, 0)
        deviant = zero_var & ~np.isclose(x - loo_mean, 0)

        z = (x - loo_mean) / np.sqrt(loo_var)

    z = z.where(~tied, 0.0)
    z = z.where(~deviant, np.sign(x - loo_mean) * extreme_z)
    z = z.where(n >= min_cohort_size, np.nan)

    out[f"{value_col}_cohort_size"] = n
    out[f"{value_col}_loo_zscore"] = z
    return out


def flag_candidates(df, zscore_col, threshold=2.0):
    """Flag rows whose |zscore_col| exceeds threshold.

    Uses pandas' nullable boolean dtype so a missing score stays missing
    (pd.NA) instead of silently becoming False the way a plain
    `NaN > threshold` comparison does with numpy bools. That silent
    False is a real, easy-to-miss bug: it quietly counts "we couldn't
    score this row" as "we scored it and it's normal."
    """
    out = df.copy()
    flag = (out[zscore_col].abs() > threshold).astype("boolean")
    flag[out[zscore_col].isna()] = pd.NA
    out[f"{zscore_col}_flag"] = flag
    return out
