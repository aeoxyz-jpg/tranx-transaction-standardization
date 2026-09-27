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
    # Rows with a null gold value (merchant-less transaction types) are not scored.
    j = j.filter(pl.col(f"{col}_gold").is_not_null())
    if not len(j):
        return 0.0
    return float((j[col] == j[f"{col}_gold"]).fill_null(False).mean())


def merchant_normalized_match(pred: pl.DataFrame, gold: pl.DataFrame) -> float:
    """Merchant match ignoring case and punctuation — fairer to generative output
    (credits 'PARAMEDIC'=='Paramedic', 'Canes'=="Cane's") than exact string match."""
    j = pred.select(["txn_id", "canonical_merchant"]).join(
        gold.select(["txn_id", "canonical_merchant"]), on="txn_id", suffix="_gold")
    j = j.filter(pl.col("canonical_merchant_gold").is_not_null())  # merchant-less rows unscored
    p = [_norm_merchant(x or "") for x in j["canonical_merchant"].to_list()]
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
    # Buckets are keyed on the normalized name (same leniency as merchant_norm), and
    # only rows whose gold has a merchant are rolled up.
    gold = gold.filter(pl.col("canonical_merchant").is_not_null())
    norm = pl.col("canonical_merchant").map_elements(lambda s: _norm_merchant(s or ""),
                                                     return_dtype=pl.Utf8)
    base = feed.select(["txn_id", "customer_id", "amount"]).filter(
        pl.col("txn_id").is_in(gold["txn_id"]))
    pred_roll = rollup_by_merchant(
        base.join(pred.select("txn_id", norm.alias("canonical_merchant")), on="txn_id"))
    gold_roll = rollup_by_merchant(
        base.join(gold.select("txn_id", norm.alias("canonical_merchant")), on="txn_id"))

    merged = gold_roll.join(
        pred_roll, on=["customer_id", "canonical_merchant"], how="left",
        suffix="_pred").with_columns(pl.col("total_spend_pred").fill_null(0.0))

    diff = (merged["total_spend"] - merged["total_spend_pred"]).abs()
    mae = float(diff.mean()) if len(diff) else 0.0
    rel = diff / merged["total_spend"].abs().clip(lower_bound=1e-9)
    within = float((rel <= tolerance).mean()) if len(rel) else 0.0
    return {"mae": round(mae, 4), "within_tolerance": round(within, 4)}


def cluster_bootstrap_ci(pred: pl.DataFrame, gold: pl.DataFrame, n_boot: int = 1000,
                         seed: int = 0) -> dict:
    """95% CI half-widths for category accuracy and merchant_norm, resampling whole
    gold-merchant clusters (transaction-type rows cluster by type). Rows of one
    merchant are correlated, so an iid CI would be too narrow on the unseen split,
    where a few merchants carry many rows."""
    import numpy as np
    j = pred.select("txn_id", pl.col("category").alias("p_cat"),
                    pl.col("canonical_merchant").alias("p_m")).join(gold, on="txn_id")
    key = (j["canonical_merchant"] if "txn_type" not in j.columns
           else j.select(pl.coalesce("canonical_merchant", "txn_type"))[:, 0]).fill_null("?")
    cat_ok = (j["p_cat"] == j["category"]).to_numpy().astype(float)
    has_m = j["canonical_merchant"].is_not_null().to_numpy()
    m_ok = np.array([_norm_merchant(p or "") == _norm_merchant(g or "")
                     for p, g in zip(j["p_m"], j["canonical_merchant"])], dtype=float)
    codes, cluster = np.unique(key.to_numpy(), return_inverse=True)
    k = len(codes)
    rng = np.random.default_rng(seed)
    cat_sum = np.bincount(cluster, cat_ok, k); n_all = np.bincount(cluster, minlength=k)
    m_sum = np.bincount(cluster, m_ok * has_m, k); n_m = np.bincount(cluster, has_m, k)
    cats, ms = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        cats.append((w * cat_sum).sum() / max(1, (w * n_all).sum()))
        ms.append((w * m_sum).sum() / max(1, (w * n_m).sum()))
    half = lambda xs: round(float((np.percentile(xs, 97.5) - np.percentile(xs, 2.5)) / 2), 3)
    return {"category": half(cats), "merchant_norm": half(ms)}
