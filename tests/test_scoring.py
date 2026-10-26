import numpy as np
import pandas as pd
import pytest

from anomaly_toolkit.scoring import flag_candidates, score_peer_anomalies


def make_df(values, group="g"):
    return pd.DataFrame({"group": [group] * len(values), "value": values})


def test_leave_one_out_zscore_matches_independent_computation():
    """The function's z-score for one row should match a z-score computed
    by hand from that row's peers alone, the whole point of leave-one-out."""
    df = make_df([10, 12, 14, 16, 18])
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=5)

    peers = pd.Series([12, 14, 16, 18])  # every value except the row being checked (10)
    expected_mean = peers.mean()
    expected_std = peers.std(ddof=1)
    expected_z = (10 - expected_mean) / expected_std

    actual_z = scored.loc[0, "value_loo_zscore"]
    assert actual_z == pytest.approx(expected_z)


def test_outlier_is_not_masked_by_its_own_inflated_spread():
    """The bug this whole module exists to fix: a plain z-score lets an
    outlier inflate its own cohort's standard deviation and shrink its own
    score. [10, 10, 10, 10, 50] under a plain (non-leave-one-out) z-score
    gives the outlier z ~= 1.79, below a typical |z| > 2 flagging threshold,
    a real miss. Leave-one-out must not reproduce that miss."""
    df = make_df([10, 10, 10, 10, 50])
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=5)

    plain_mean = df["value"].mean()
    plain_std = df["value"].std(ddof=1)
    plain_z_for_outlier = (50 - plain_mean) / plain_std
    assert abs(plain_z_for_outlier) < 2  # confirms the plain method would miss it

    loo_z_for_outlier = scored.loc[4, "value_loo_zscore"]
    assert abs(loo_z_for_outlier) > abs(plain_z_for_outlier)
    assert abs(loo_z_for_outlier) > 2  # leave-one-out correctly flags it


def test_zero_variance_tied_row_scores_zero():
    """A row matching an otherwise-unanimous peer group is not an outlier."""
    df = make_df([5, 5, 5, 5, 5])
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=5)
    assert scored.loc[0, "value_loo_zscore"] == 0.0


def test_zero_variance_deviant_row_gets_extreme_sentinel():
    """A row that disagrees with an otherwise-unanimous peer group is
    maximally anomalous, not an undefined 0/0."""
    df = make_df([5, 5, 5, 5, 7])
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=5, extreme_z=99.0)
    deviant_row_z = scored.loc[4, "value_loo_zscore"]
    assert deviant_row_z == 99.0


def test_near_zero_variance_is_capped_not_left_unbounded():
    """Regression test for a real bug found running this against actual
    CMS billing-rate data: a tiny cohort whose peers cluster at almost,
    but not exactly, identical values (common with a bounded ratio like a
    rate between 0 and 1) can produce a variance that's nonzero but tiny
    enough to blow the z-score up toward unbounded values, tens of
    millions, in the real case that surfaced this. extreme_z must cap
    every z-score's magnitude, not just the exact-zero-variance case."""
    df = make_df([0.60, 0.60, 0.60, 0.6000001, 0.9])
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=5, extreme_z=99.0)
    assert scored["value_loo_zscore"].abs().max() <= 99.0


def test_cohorts_below_minimum_size_are_not_scored():
    df = make_df([1, 2, 3])  # only 3 rows, below the default min of 5
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=5)
    assert scored["value_loo_zscore"].isna().all()
    assert (scored["value_cohort_size"] == 3).all()


def test_rows_with_missing_group_value_are_flagged_and_unscored():
    df = pd.DataFrame({"group": ["a", "a", "a", "a", None], "value": [1, 2, 3, 4, 5]})
    scored = score_peer_anomalies(df, "value", ["group"], min_cohort_size=2)
    assert scored["value_missing_peer_group"].tolist() == [False, False, False, False, True]
    assert pd.isna(scored.loc[4, "value_loo_zscore"])


def test_flag_candidates_keeps_missing_scores_missing_not_false():
    """Regression test for a real bug: NaN > threshold evaluates to False
    in pandas/numpy, which silently turns 'we could not score this row'
    into 'we scored it and it is normal.' flag_candidates must not do that."""
    df = pd.DataFrame({"z": [3.0, 0.5, np.nan]})
    flagged = flag_candidates(df, "z", threshold=2.0)

    assert flagged.loc[0, "z_flag"] == True  # noqa: E712
    assert flagged.loc[1, "z_flag"] == False  # noqa: E712
    assert pd.isna(flagged.loc[2, "z_flag"])
    assert flagged["z_flag"].dtype == "boolean"
