from pathlib import Path
import polars as pl

_COLUMNS = [
    ("route", "route"),
    ("split", "split"),
    ("category_acc", "Category Acc"),
    ("category_ci", "Cat ±95%"),
    ("macro_f1", "Macro F1"),
    ("merchant_acc", "Merchant Acc"),
    ("merchant_norm", "Merchant Norm"),
    ("merchant_norm_ci", "Merch ±95%"),
    ("dedup_ratio", "Dedup Ratio"),
    ("gold_dedup_ratio", "Gold Dedup"),
    ("kpi_within_tol", "Spend KPI"),
    ("avg_ms", "ms/txn"),
]


def build_leaderboard_md(results: list[dict]) -> str:
    """Render a markdown leaderboard table from per-route metric dicts."""
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for r in results:
        cells = []
        for key, _ in _COLUMNS:
            v = r.get(key, "")
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_leaderboard(results: list[dict], out_dir: Path, name: str = "leaderboard.md",
                     title: str = "Route Leaderboard", preamble: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    body = f"{preamble}\n\n" if preamble else ""
    path.write_text(f"# {title}\n\n{body}" + build_leaderboard_md(results))
    return path


def plot_customer_spend(feed: pl.DataFrame, pred: pl.DataFrame, gold: pl.DataFrame,
                        customer_id: str, out_dir: Path) -> Path:
    """The money shot: gold vs predicted spend-by-merchant for one customer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tranx.pipeline.aggregate import rollup_by_merchant

    base = feed.select(["txn_id", "customer_id", "amount"])
    gold = gold.filter(pl.col("canonical_merchant").is_not_null())
    pred = pred.filter(pl.col("txn_id").is_in(gold["txn_id"]))
    g = rollup_by_merchant(base.join(gold.select(["txn_id", "canonical_merchant"]),
                                     on="txn_id")).filter(pl.col("customer_id") == customer_id)
    p = rollup_by_merchant(base.join(pred.select(["txn_id", "canonical_merchant"]),
                                     on="txn_id")).filter(pl.col("customer_id") == customer_id)

    merchants = g["canonical_merchant"].to_list()
    g_map = dict(zip(g["canonical_merchant"], g["total_spend"]))
    p_map = dict(zip(p["canonical_merchant"], p["total_spend"]))
    gold_vals = [g_map.get(m, 0) for m in merchants]
    pred_vals = [p_map.get(m, 0) for m in merchants]

    x = range(len(merchants))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], gold_vals, width=0.4, label="gold")
    ax.bar([i + 0.2 for i in x], pred_vals, width=0.4, label="predicted")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merchants, rotation=45, ha="right")
    ax.set_ylabel("Total spend")
    ax.set_title(f"Spend by merchant — customer {customer_id}")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"spend_{customer_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
