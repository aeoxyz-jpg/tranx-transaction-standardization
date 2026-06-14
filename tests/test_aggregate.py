import polars as pl
from tranx.pipeline.aggregate import rollup_by_merchant


def test_rollup_sums_abs_spend_per_customer_merchant():
    df = pl.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2"],
        "canonical_merchant": ["McDonald's", "McDonald's", "BP", "McDonald's"],
        "amount": [-10.0, -12.0, -40.0, -5.0],
    })
    out = rollup_by_merchant(df)
    c1_mcd = out.filter((pl.col("customer_id") == "C1") &
                        (pl.col("canonical_merchant") == "McDonald's"))
    assert c1_mcd["total_spend"][0] == 22.0
    c1_bp = out.filter((pl.col("customer_id") == "C1") &
                       (pl.col("canonical_merchant") == "BP"))
    assert c1_bp["total_spend"][0] == 40.0


def test_rollup_counts_transactions():
    df = pl.DataFrame({
        "customer_id": ["C1", "C1"],
        "canonical_merchant": ["McDonald's", "McDonald's"],
        "amount": [-10.0, -12.0],
    })
    out = rollup_by_merchant(df)
    assert out["txn_count"][0] == 2
