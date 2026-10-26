# Proof of concept: real-scale data

This folder is for actually stress-testing the methodology against a real,
large, public dataset, not Medicare claims (those require official CMS data
access), but a genuine CMS public dataset large enough to prove the
architecture holds up past pandas' comfortable range.

## The dataset

CMS publishes **"Medicare Physician & Other Practitioners by Provider and
Service"** on data.cms.gov: provider-level billing data by HCPCS/CPT
procedure code, typically several million rows per year, free, no special
access required. Search data.cms.gov for that title to find the current
year's file, the exact URL changes release to release so it isn't hardcoded
here.

This is a direct fit for the upcoding example already in the main README:
the file includes provider specialty, geography, procedure code (including
the E&M codes like 99212-99215), and billing volume, exactly the columns
needed to compare a provider's high-complexity billing rate against
same-specialty, same-region peers.

## Why this needs a different architecture than the demo app

The demo app loads the whole CSV into pandas memory via `pd.read_csv`.
That's fine at the 13 MB / 150K row scale tested so far. A real CMS
utilization file is large enough that loading it whole, then computing the
peer-group aggregation in pandas, risks the same out-of-memory failure that
already happened once on Streamlit Cloud's free tier with a much smaller
file.

The fix: push the aggregation into the database instead of pulling all the
raw data into the app. The leave-one-out scoring math (`scoring.py`) is
just `SUM`, `SUM` of squares, and `COUNT` per peer group, exactly what a SQL
engine does natively and efficiently at any scale.

## The plan

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
