-- Athena/Presto translation of anomaly_toolkit.scoring.score_peer_anomalies().
--
-- Does the same leave-one-out peer-group scoring as the Python version, but
-- as a SQL aggregation so it runs over data too large to load into pandas
-- memory at once. Point this at an Athena table over S3 (Parquet
-- recommended for a large file), pull back only the scored result, which
-- is far smaller than the raw data, and feed that into the existing app
-- or classifier code unchanged.
--
-- Replace these three placeholders for your actual data:
--   provider_utilization  -> your Athena table name
--   value_col             -> the numeric column to screen for anomalies
--   group_col_1, group_col_2, ... -> the columns defining a peer cohort
--
-- Logic verified to match scoring.py by cross-checking both against the
-- same synthetic data in Python (see poc/verify_sql_matches_python.py).
-- Not yet executed against a live Athena table, that needs testing against
-- your actual data and table schema.

WITH cohort_stats AS (
    SELECT
        *,
        SUM(value_col) OVER (PARTITION BY group_col_1, group_col_2) AS group_sum,
        SUM(value_col * value_col) OVER (PARTITION BY group_col_1, group_col_2) AS group_sumsq,
        COUNT(value_col) OVER (PARTITION BY group_col_1, group_col_2) AS cohort_size
    FROM provider_utilization
),

loo_calc AS (
    SELECT
        *,
        (group_sum - value_col) / NULLIF(cohort_size - 1, 0) AS loo_mean
    FROM cohort_stats
),

loo_var_calc AS (
    SELECT
        *,
        GREATEST(
            (
                (group_sumsq - value_col * value_col)
                - (cohort_size - 1) * POWER(loo_mean, 2)
            ) / NULLIF(cohort_size - 2, 0),
            0.0
        ) AS loo_var
    FROM loo_calc
)

SELECT
    *,
    CASE
        WHEN cohort_size < 5 THEN NULL
        WHEN loo_var = 0 AND ABS(value_col - loo_mean) < 1e-9 THEN 0.0
        WHEN loo_var = 0 THEN SIGN(value_col - loo_mean) * 99.0
        ELSE (value_col - loo_mean) / SQRT(loo_var)
    END AS loo_zscore
FROM loo_var_calc;
