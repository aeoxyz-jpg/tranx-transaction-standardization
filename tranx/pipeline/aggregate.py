import polars as pl


def rollup_by_merchant(df: pl.DataFrame) -> pl.DataFrame:
    """Total absolute spend and transaction count per (customer, canonical_merchant)."""
    return (
        df.with_columns(pl.col("amount").abs().alias("_abs"))
        .group_by(["customer_id", "canonical_merchant"])
        .agg(
            pl.col("_abs").sum().round(2).alias("total_spend"),
            pl.len().alias("txn_count"),
        )
        .sort(["customer_id", "total_spend"], descending=[False, True])
    )
