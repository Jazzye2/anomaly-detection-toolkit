"""Run the actual peer-group scoring against real CMS billing data.

Aggregates the raw (provider, HCPCS code, place-of-service) rows up to
one row per provider: total E&M visit volume and what fraction were
billed at the two highest complexity levels (99214+99215), matching the
real OIG "high-coding physician" definition cited in the README. Then
runs the exact same score_peer_anomalies used throughout this project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from anomaly_toolkit.scoring import flag_candidates, score_peer_anomalies

IN_PATH = Path(__file__).resolve().parent / "cms_em_visits_real.csv"
OUT_PATH = Path(__file__).resolve().parent / "real_provider_scores.csv"

HIGH_LEVEL_CODES = {99214, 99215}  # HCPCS_Cd loads as int64, not string, from this CSV


def main():
    raw = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Raw rows (provider x code x place of service): {len(raw):,}")

    raw["is_high_level"] = raw["HCPCS_Cd"].isin(HIGH_LEVEL_CODES)
    raw["high_level_srvcs"] = np.where(raw["is_high_level"], raw["Tot_Srvcs"], 0)

    provider = raw.groupby("Rndrng_NPI").agg(
        total_em_srvcs=("Tot_Srvcs", "sum"),
        high_level_srvcs=("high_level_srvcs", "sum"),
        specialty=("Rndrng_Prvdr_Type", "first"),
        state=("Rndrng_Prvdr_State_Abrvtn", "first"),
        distinct_codes_billed=("HCPCS_Cd", "nunique"),
    ).reset_index()

    provider["high_level_rate"] = provider["high_level_srvcs"] / provider["total_em_srvcs"]

    print(f"Unique providers: {len(provider):,}")
    print(provider["high_level_rate"].describe())

    scored = score_peer_anomalies(
        provider, "high_level_rate", ["specialty", "state"], min_cohort_size=5
    )
    scored = flag_candidates(scored, "high_level_rate_loo_zscore", threshold=2.0)

    total = len(scored)
    missing_group = int(scored["high_level_rate_missing_peer_group"].sum())
    unscored_small = int(
        scored["high_level_rate_loo_zscore"].isna().sum() - missing_group
    )
    scoreable = total - missing_group - unscored_small
    flagged = int(scored["high_level_rate_loo_zscore_flag"].sum(skipna=True))

    print(f"\nTotal providers: {total:,}")
    print(f"Excluded, missing peer-group data: {missing_group:,}")
    print(f"Excluded, too few peers: {unscored_small:,}")
    print(f"Successfully scored: {scoreable:,}")
    print(f"Flagged (|z| > 2): {flagged:,}")

    top_flagged = scored[scored["high_level_rate_loo_zscore_flag"] == True].sort_values(  # noqa: E712
        "high_level_rate_loo_zscore", key=abs, ascending=False
    )
    print("\nTop 10 flagged providers (most extreme):")
    cols = ["Rndrng_NPI", "specialty", "state", "total_em_srvcs",
            "high_level_rate", "high_level_rate_cohort_size", "high_level_rate_loo_zscore"]
    print(top_flagged[cols].head(10).to_string(index=False))

    scored.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
