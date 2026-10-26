# Proof of concept: real-scale data

This folder stress-tests the methodology against a real, large, public
dataset, not Medicare claims (those require official CMS data access), but
a genuine CMS public dataset. **This has actually been run against real
data, not just planned**, see "Results on real data" below.

## The dataset

CMS publishes **"Medicare Physician & Other Practitioners by Provider and
Service"** via a public data-api on data.cms.gov: provider-level billing
data by HCPCS/CPT procedure code, roughly 9.8M rows total across all
procedures. `download_real_data.py` pulls a real slice of it, specifically
the five established-patient office visit E&M codes (99211-99215), paginated
directly from the live API.

This is a direct fit for the upcoding example already in the main README:
the data includes provider specialty, state, procedure code, and billing
volume, exactly the columns needed to compare a provider's high-complexity
billing rate against same-specialty, same-state peers, matching the real
OIG "high-coding physician" methodology cited in the main README.

## Results on real data

Ran the full pipeline against 496,054 real rows (five E&M codes, pulled
directly from the live CMS API, `download_real_data.py`), aggregated up to
307,771 individual providers (`run_real_analysis.py`), each scored on what
fraction of their E&M visits were billed at the two highest complexity
levels (99214+99215), the same definition OIG uses for a "high-coding
physician."

```
Total providers: 307,771
Excluded, missing peer-group data: 0
Excluded, too few peers: 1,863
Successfully scored: 305,908
Flagged (|z| > 2): 9,146  (3.0%)
```

Top flagged providers include real, named specialties where the signal
reads as genuinely sensible: a plastic surgeon billing 100% of visits at
the two highest complexity levels, an oncologist at 98%, a psychiatrist
at 0% (flagged in the *other* direction, under-coding relative to peers,
which the same bidirectional threshold correctly catches, not just
upcoding). A real, small sample of the actual flagged output is in
`sample_real_flagged_providers.csv`.

**A real bug surfaced by this run, and got fixed**: the first attempt
produced z-scores in the tens of millions on some of the smallest peer
cohorts, a bounded ratio (0 to 1) can produce a variance that's nonzero
but tiny enough to blow a z-score up toward unbounded values once a cohort
gets small. Fixed in `scoring.py` by making the `extreme_z` sentinel a hard
cap on every z-score's magnitude, not just the exact-zero-variance special
case, with a regression test (`test_near_zero_variance_is_capped_not_left_unbounded`)
added to the main suite. This is exactly why running against real data
matters, synthetic test data didn't happen to produce this pattern, real
CMS billing rates did.

The raw downloaded data (190 MB) and the full scored output (27 MB) aren't
committed to the repo, too large for a normal git push, re-run
`download_real_data.py` then `run_real_analysis.py` to regenerate them.

## Why bigger than this would need a different architecture

496,054 rows and 307,771 providers ran fine locally in plain pandas, no
special infrastructure needed, that's genuinely useful to know: the
methodology holds up well past the original 150K-row demo scale without
any changes. But this was five procedure codes out of roughly 15,000 in
the full CMS file, and one data-api dataset out of many. The full
provider-and-service file is around 9.8M rows across every procedure code,
and pulling every code, not just E&M visits, would mean a file easily in
the multi-gigabyte range, past where loading everything into one pandas
DataFrame at once is still comfortable.

For that scale, the fix is pushing the aggregation into a database instead
of pulling all the raw data into the app. The leave-one-out scoring math
(`scoring.py`) is just `SUM`, `SUM` of squares, and `COUNT` per peer group,
exactly what a SQL engine does natively and efficiently at any scale.
`athena_scoring.sql` already implements this, verified to produce identical
results to the Python version (see below), not yet run against a live
Athena table.

## The Athena plan, for scale beyond what pandas handled here

1. **Land the data in S3.** Convert the CSV to Parquet (smaller, faster to
   query) and upload to an S3 bucket.
2. **Create an Athena table over it**, either through the Glue Data Catalog
   crawler or a manual `CREATE EXTERNAL TABLE` statement pointing at the S3
   location.
3. **Run `athena_scoring.sql`** against that table (after filling in your
   real column names in place of the `value_col` / `group_col_1` /
   `group_col_2` placeholders). This computes the exact same leave-one-out
   z-score as `scoring.py`, verified to match it exactly, see
   `verify_sql_matches_python.py` and its output below.
4. **Export the scored result**, which is far smaller than the raw table
   (one row per original record, but no longer needing the full row to be
   held in memory at once), back to S3 or directly into a pandas DataFrame.
5. **Feed that into the existing classifier and app code unchanged.** None
   of `classifier.py`, `explain.py`, or `app.py` need to change, they
   already expect a DataFrame with a `_loo_zscore` column, which is what
   the SQL produces too.
6. **Train on a stratified sample, not every row**, if the scored dataset
   is still too large to fit comfortably in memory for model training.
   Standard practice, and `sklearn`'s `train_test_split` with `stratify=`
   already used throughout this project makes that straightforward.

## Verification already done

```
$ python poc/verify_sql_matches_python.py
Rows compared: 2,000
All rows match (within floating point tolerance): True
Max absolute difference: 0.0
```

The SQL logic is confirmed mathematically identical to the already-tested
Python implementation, on synthetic data, independently re-implemented
rather than just calling the same code twice. What's not yet done: running
`athena_scoring.sql` against a real Athena table. That needs your AWS
account, the real CMS file loaded into S3, and the placeholder column names
filled in to match its actual schema.
