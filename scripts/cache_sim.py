"""Descriptor-cache rate illustrations for the write-up (D6 / plan T4).

(1) MoneyData per-person new-descriptor rate by year: DEB + DD rows in date order
    (ties broken by original row order in the statement); a row is "new" the first
    time its exact description appears. This is a real measurement of one person's
    repeat structure, locked once it reproduces the numbers recorded in the plan.
(2) Synthetic bank-wide cache hit rate by posted month, on whatever data/bank_feed.parquet
    currently holds (ties within a day broken by txn_id). Synthetic dates are drawn
    uniformly at random, so this is a parameter illustration, not a measurement, and
    the JSON says so.

Usage: python scripts/cache_sim.py
"""
import json

import polars as pl

from tranx import config

OUT = config.REPORTS_DIR / "cache_sim.json"


def running_new_flags(descriptions: list[str]) -> list[bool]:
    """True where a description has not occurred earlier in the (already ordered)
    sequence. The complement is a cache hit."""
    seen: set[str] = set()
    flags = []
    for d in descriptions:
        flags.append(d not in seen)
        seen.add(d)
    return flags


def moneydata_new_descriptor_rate(raw_csv_path) -> dict[str, float]:
    """Per-year new-descriptor rate over DEB/DD rows, sorted by date then original
    row order."""
    df = pl.read_csv(raw_csv_path).with_row_index("_orig")
    sub = (df.filter(pl.col("Transaction Type").is_in(["DEB", "DD"]))
             .with_columns(pl.col("Transaction Date").str.to_date("%d/%m/%Y").alias("_date"))
             .sort(["_date", "_orig"]))
    is_new = running_new_flags(sub["Transaction Description"].to_list())
    years = [d.year for d in sub["_date"]]
    out = {}
    for y in sorted(set(years)):
        flags = [n for n, yy in zip(is_new, years) if yy == y]
        out[str(y)] = round(sum(flags) / len(flags), 3)
    return out


def synthetic_cache_hit_rate_by_month(feed_path) -> dict[str, float]:
    """Bank-wide cache hit rate by posted month: a row hits if its exact description
    appeared in any earlier-posted row (ties within a day ordered by txn_id)."""
    df = (pl.read_parquet(feed_path)
            .with_columns(pl.col("posted_date").str.to_date("%Y-%m-%d").alias("_date"))
            .sort(["_date", "txn_id"]))
    is_hit = [not n for n in running_new_flags(df["description"].to_list())]
    months = [d.strftime("%Y-%m") for d in df["_date"]]
    out = {}
    for m in sorted(set(months)):
        flags = [h for h, mm in zip(is_hit, months) if mm == m]
        out[m] = round(sum(flags) / len(flags), 3)
    return out


def main():
    moneydata = moneydata_new_descriptor_rate(config.DATA_DIR / "real" / "moneydata_raw.csv")
    synthetic = synthetic_cache_hit_rate_by_month(config.DATA_DIR / "bank_feed.parquet")
    from tranx.eval.manifest import file_sha256
    res = {
        "feed_sha256": file_sha256(config.DATA_DIR / "bank_feed.parquet"),
        "moneydata_new_descriptor_rate_by_year": moneydata,
        "synthetic_cache_hit_rate_by_month": {
            "note": "parameter illustration; dates are uniform",
            "by_month": synthetic,
        },
    }
    print(json.dumps(res, indent=2))
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
