"""Pull real CMS Medicare Physician & Other Practitioners data for the
office-visit E&M codes (99211-99215), paginated, with a bounded per-code
cap. Not the full 9.8M-row national dataset, a genuine, large, real slice
of it, enough to actually stress-test the pipeline past the 150K-row demo
scale while staying a reasonable download.
"""
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://data.cms.gov/data-api/v1/dataset/92396110-2aed-4d63-a6a2-5d6207d46a29/data"
EM_CODES = ["99211", "99212", "99213", "99214", "99215"]
PAGE_SIZE = 6500  # confirmed max the API will return per request
MAX_PAGES_PER_CODE = 20  # caps each code at 130,000 rows, ~650,000 rows total across 5 codes
OUT_PATH = Path(__file__).resolve().parent / "cms_em_visits_real.csv"


def fetch_code(code):
    rows = []
    offset = 0
    for page in range(MAX_PAGES_PER_CODE):
        resp = requests.get(
            BASE_URL,
            params={"size": PAGE_SIZE, "offset": offset, "filter[HCPCS_Cd]": code},
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        print(f"  {code}: page {page + 1}, {len(batch)} rows, running total {len(rows):,}")
        if len(batch) < PAGE_SIZE:
            break  # last page was partial, nothing more to fetch
        offset += PAGE_SIZE
        time.sleep(0.3)  # be polite to a public, rate-limited government API
    return rows


def main():
    import pandas as pd

    all_rows = []
    for code in EM_CODES:
        print(f"Fetching HCPCS {code}...")
        all_rows.extend(fetch_code(code))

    df = pd.DataFrame(all_rows)
    print(f"\nTotal real rows pulled: {len(df):,}")
    print(df["HCPCS_Cd"].value_counts())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
