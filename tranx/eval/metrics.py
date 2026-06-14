import re
import polars as pl
from sklearn.metrics import f1_score
from tranx.pipeline.aggregate import rollup_by_merchant

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _norm_merchant(s: str) -> str:
    """Case-insensitive, punctuation/space-stripped form for fair matching."""
    return _NON_ALNUM.sub("", s.lower())


def _accuracy(pred: pl.DataFrame, gold: pl.DataFrame, col: str) -> float:
    j = pred.select(["txn_id", col]).join(
        gold.select(["txn_id", col]), on="txn_id", suffix="_gold")
    return float((j[col] == j[f"{col}_gold"]).mean())


def merchant_normalized_match(pred: pl.DataFrame, gold: pl.DataFrame) -> float:
    """Merchant match ignoring case and punctuation — fairer to generative output
    (credits 'PARAMEDIC'=='Paramedic', 'Canes'=="Cane's") than exact string match."""
    j = pred.select(["txn_id", "canonical_merchant"]).join(
        gold.select(["txn_id", "canonical_merchant"]), on="txn_id", suffix="_gold")
    p = [_norm_merchant(x) for x in j["canonical_merchant"].to_list()]
    g = [_norm_merchant(x) for x in j["canonical_merchant_gold"].to_list()]
    if not p:
        return 0.0
    return sum(a == b for a, b in zip(p, g)) / len(p)


def category_accuracy(pred: pl.DataFrame, gold: pl.DataFrame) -> float:
    return _accuracy(pred, gold, "category")


def merchant_exact_match(pred: pl.DataFrame, gold: pl.DataFrame) -> float:
    return _accuracy(pred, gold, "canonical_merchant")


def direction_accuracy(pred: pl.DataFrame, gold: pl.DataFrame) -> float:
    return _accuracy(pred, gold, "direction")


def category_macro_f1(pred: pl.DataFrame, gold: pl.DataFrame) -> float:
    j = pred.select(["txn_id", "category"]).join(
        gold.select(["txn_id", "category"]), on="txn_id", suffix="_gold")
    return float(f1_score(j["category_gold"].to_list(), j["category"].to_list(),
                          average="macro", zero_division=0))


def dedup_ratio(pred: pl.DataFrame, raw_distinct: int) -> float:
    """Distinct predicted merchants / distinct raw descriptions. Lower = more collapse."""
    if raw_distinct == 0:
        return 0.0
    return pred["canonical_merchant"].n_unique() / raw_distinct


def merchant_spend_kpi(feed: pl.DataFrame, pred: pl.DataFrame, gold: pl.DataFrame,
                       tolerance: float = 0.01) -> dict:
    """Compare per-(customer, merchant) spend totals between predicted and gold.

    Returns mean absolute error of bucket totals and the fraction of gold buckets
    reproduced within `tolerance` (relative).
    """
    base = feed.select(["txn_id", "customer_id", "amount"])
    pred_roll = rollup_by_merchant(
        base.join(pred.select(["txn_id", "canonical_merchant"]), on="txn_id"))
    gold_roll = rollup_by_merchant(
        base.join(gold.select(["txn_id", "canonical_merchant"]), on="txn_id"))

    merged = gold_roll.join(
        pred_roll, on=["customer_id", "canonical_merchant"], how="left",
        suffix="_pred").with_columns(pl.col("total_spend_pred").fill_null(0.0))

    diff = (merged["total_spend"] - merged["total_spend_pred"]).abs()
    mae = float(diff.mean()) if len(diff) else 0.0
    rel = diff / merged["total_spend"].abs().clip(lower_bound=1e-9)
    within = float((rel <= tolerance).mean()) if len(rel) else 0.0
    return {"mae": round(mae, 4), "within_tolerance": round(within, 4)}
