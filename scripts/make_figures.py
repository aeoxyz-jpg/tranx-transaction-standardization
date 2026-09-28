"""Generate the write-up figures from this run's recorded results (spec D7 step 4).

No numbers are hardcoded: everything is read from
  - reports/leaderboard_hard_jev.json  (written by `python -m tranx.cli eval
    --name leaderboard_hard_jev.md`; a list of per-route/split/view result dicts)
  - reports/significance.json          (written by scripts/significance.py)
  - reports/real/moneydata_summary_high-medium_aliased.json (committed)

Any missing input file is skipped with a printed warning; the figures that
depend on it are not produced, but the script does not crash.

Dropped in this revision (spec D7 step 4, D8): the LoRA route
(slm_lora / lora_vs_fewshot.png) and the two static tables (table_engines.png,
table_results.png) are no longer generated. Their old PNG files are left on
disk as orphans -- see the note printed at the end of main().
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "reports" / "figures"
DEFAULT_LEADERBOARD = ROOT / "reports" / "leaderboard_hard_jev.json"
DEFAULT_SIGNIFICANCE = ROOT / "reports" / "significance.json"
DEFAULT_MONEYDATA = ROOT / "reports" / "real" / "moneydata_summary_high-medium_aliased.json"

# Preferred route order for the per-route figures; any route present in the
# data but not listed here is appended at the end (nothing is dropped).
MERCHANT_NORM_ROUTES = ["rules", "embedding", "slm_fewshot", "jev_merchant", "jev_slm", "cleaner"]
CATEGORY_ROUTES = ["rules", "embedding", "slm_fewshot", "jev_merchant", "jev_slm"]
LATENCY_ROUTE_ORDER = ["cleaner", "metadata", "rules", "embedding", "slm_fewshot",
                       "jev_merchant", "jev_slm"]
SPLIT_COLORS = {"random": "#4C72B0", "unseen": "#DD8452"}

ORPHANED_FIGURES = ["lora_vs_fewshot.png", "table_engines.png", "table_results.png"]


def _load_json(path: Path):
    """Returns the parsed JSON, or None (with a warning) if the file is missing."""
    if path is None or not Path(path).exists():
        print(f"make_figures: WARNING missing input {path}, skipping dependent figure(s)")
        return None
    return json.loads(Path(path).read_text())


def _num(v):
    """Leaderboard cells use the string '-' for not-applicable; treat as missing."""
    return v if isinstance(v, (int, float)) else None


def _order(present, preferred):
    ordered = [r for r in preferred if r in present]
    ordered += [r for r in present if r not in ordered]
    return ordered


def _rows_by_route_split(leaderboard, view):
    """route -> split -> row, restricted to the given view."""
    out = {}
    for r in leaderboard:
        if r.get("view") != view:
            continue
        out.setdefault(r["route"], {})[r["split"]] = r
    return out


def fig_merchant_norm(leaderboard, out_dir: Path):
    """Model view, random vs unseen, merchant_norm per route with CI error bars.
    `cleaner` is included as a reference bar (hatched, no CI is meaningful lift claim)."""
    idx = _rows_by_route_split(leaderboard, "model")
    routes = _order([r for r in MERCHANT_NORM_ROUTES if r in idx], MERCHANT_NORM_ROUTES)
    routes = [r for r in routes if any(_num(idx[r].get(s, {}).get("merchant_norm")) is not None
                                       for s in ("random", "unseen"))]
    if not routes:
        print("make_figures: no merchant_norm data, skipping merchant_norm.png")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(routes))
    w = 0.38
    cleaner_idx = routes.index("cleaner") if "cleaner" in routes else None
    for i, split in enumerate(["random", "unseen"]):
        vals, errs, present = [], [], []
        for r in routes:
            row = idx[r].get(split, {})
            v = _num(row.get("merchant_norm"))
            e = _num(row.get("merchant_norm_ci"))
            vals.append(v if v is not None else 0.0)
            errs.append(e if e is not None else 0.0)
            present.append(v is not None)
        bars = ax.bar([p + (i - 0.5) * w for p in x], vals, width=w, yerr=errs,
                      label=split, color=SPLIT_COLORS[split], capsize=3)
        for b, v, ok in zip(bars, vals, present):
            if ok:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
        if cleaner_idx is not None:
            bars[cleaner_idx].set_hatch("//")
            bars[cleaner_idx].set_edgecolor("#444444")
    labels = [f"{r}\n(reference)" if r == "cleaner" else r for r in routes]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("merchant norm accuracy")
    ax.set_title("Merchant normalization — model view, random vs unseen")
    ax.legend(title="split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "merchant_norm.png", dpi=130)
    plt.close(fig)
    print("wrote", out_dir / "merchant_norm.png")


def fig_category(leaderboard, out_dir: Path):
    """Model view, category_acc per route with CI error bars; the metadata-only
    baseline is drawn as a horizontal reference line (it has no merchant score)."""
    idx = _rows_by_route_split(leaderboard, "model")
    routes = _order([r for r in CATEGORY_ROUTES if r in idx], CATEGORY_ROUTES)
    routes = [r for r in routes if any(_num(idx[r].get(s, {}).get("category_acc")) is not None
                                       for s in ("random", "unseen"))]
    if not routes:
        print("make_figures: no category data, skipping category.png")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(routes))
    w = 0.38
    for i, split in enumerate(["random", "unseen"]):
        vals, errs, present = [], [], []
        for r in routes:
            row = idx[r].get(split, {})
            v = _num(row.get("category_acc"))
            e = _num(row.get("category_ci"))
            vals.append(v if v is not None else 0.0)
            errs.append(e if e is not None else 0.0)
            present.append(v is not None)
        bars = ax.bar([p + (i - 0.5) * w for p in x], vals, width=w, yerr=errs,
                      label=split, color=SPLIT_COLORS[split], capsize=3)
        for b, v, ok in zip(bars, vals, present):
            if ok:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
    meta = idx.get("metadata", {})
    meta_row = meta.get("random") or meta.get("unseen")
    if meta_row is not None and _num(meta_row.get("category_acc")) is not None:
        ax.axhline(_num(meta_row["category_acc"]), color="#888888", linestyle="--",
                   linewidth=1.3, label=f"metadata baseline ({meta_row['split']})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(routes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("category accuracy")
    ax.set_title("Category accuracy — model view, random vs unseen")
    ax.legend(title="split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "category.png", dpi=130)
    plt.close(fig)
    print("wrote", out_dir / "category.png")


def fig_latency(leaderboard, out_dir: Path):
    """avg_ms per route (model view, random split preferred, unseen as fallback),
    log scale."""
    idx = _rows_by_route_split(leaderboard, "model")
    routes, vals = [], []
    for r in _order(list(idx.keys()), LATENCY_ROUTE_ORDER):
        row = idx[r].get("random") or idx[r].get("unseen")
        v = _num(row.get("avg_ms")) if row else None
        if v is not None:
            routes.append(r)
            vals.append(v)
    if not routes:
        print("make_figures: no latency data, skipping latency.png")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(routes, vals, color="#55A868")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("ms / transaction (log scale)")
    ax.set_title("Cost per transaction — model view")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "latency.png", dpi=130)
    plt.close(fig)
    print("wrote", out_dir / "latency.png")


def _forest_rows(significance: dict):
    """Flattens {"primary": {...}, "exploratory": {...}} into an ordered list of
    (key, section, entry) for entries that have a diff/lo/hi triple, plus the list
    of (key, section) omitted because they were skipped."""
    rows, skipped = [], []
    for section in ("primary", "exploratory"):
        for key, entry in (significance.get(section) or {}).items():
            if not isinstance(entry, dict):
                continue
            if "skipped" in entry:
                skipped.append(f"{key} ({section})")
            elif {"diff", "lo", "hi"} <= set(entry):
                rows.append((key, section, entry))
    return rows, skipped


def fig_significance(significance: dict, out_dir: Path):
    """Forest plot of primary + exploratory paired diffs with CIs. Skipped
    entries (missing preds) are omitted from the plot and noted in the caption."""
    rows, skipped = _forest_rows(significance)
    if not rows:
        print("make_figures: no significance entries with a CI, skipping significance.png")
        return
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.55 * len(rows)))
    y = list(range(len(rows)))[::-1]
    for yi, (key, section, entry) in zip(y, rows):
        diff, lo, hi = entry["diff"], entry["lo"], entry["hi"]
        color = "#4C72B0" if section == "primary" else "#999999"
        ax.errorbar([diff], [yi], xerr=[[diff - lo], [hi - diff]], fmt="o",
                   color=color, ecolor=color, capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e.get('comparison', key)}" + (" [primary]" if section == "primary" else "")
                        for key, section, e in rows])
    ax.set_xlabel("paired accuracy difference (route A - route B)")
    title = "Paired significance: primary comparisons in blue, exploratory in gray"
    if skipped:
        title += f"\nomitted (missing predictions): {', '.join(skipped)}"
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "significance.png", dpi=130)
    plt.close(fig)
    print("wrote", out_dir / "significance.png")


# MoneyData methods shown, with display names (the summary also holds cascade variants).
MONEYDATA_METHODS = {
    "derive": "cleaner", "fuzzy_known": "fuzzy (full list)", "embed_known": "embedding (full list)",
    "jev_known": "Jev (full list)", "slm": "SLM", "cascade_jevnone_to_slm": "Jev -> SLM (realistic list)",
}


def fig_moneydata(summary: dict, out_dir: Path):
    """MoneyData merchant accuracy per method: per-descriptor (distinct) vs
    spend-weighted. Only methods carrying both numbers are plotted."""
    methods, distinct, spend, spend_x = [], [], [], []
    for key in MONEYDATA_METHODS:
        entry = summary.get(key)
        if not isinstance(entry, dict):
            continue
        d, s = entry.get("distinct"), entry.get("spend_weighted")
        if isinstance(d, (int, float)) and isinstance(s, (int, float)):
            methods.append(MONEYDATA_METHODS[key])
            distinct.append(d)
            spend.append(s)
            spend_x.append(entry.get("spend_weighted_excl_top") or 0.0)
    if not methods:
        print("make_figures: no MoneyData distinct/spend_weighted pairs, skipping moneydata.png")
        return
    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(methods)), 5))
    x = range(len(methods))
    w = 0.27
    b1 = ax.bar([p - w for p in x], distinct, width=w, label="per descriptor",
               color="#4C72B0")
    b2 = ax.bar(list(x), spend, width=w, label="spend-weighted", color="#DD8452")
    b3 = ax.bar([p + w for p in x], spend_x, width=w,
               label="spend-weighted, largest descriptor removed", color="#55A868")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("merchant accuracy")
    ax.set_title("MoneyData merchant accuracy — per descriptor vs spend-weighted")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "moneydata.png", dpi=130)
    plt.close(fig)
    print("wrote", out_dir / "moneydata.png")


def run(leaderboard_path: Path, significance_path: Path, moneydata_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = _load_json(leaderboard_path)
    if leaderboard is not None:
        fig_merchant_norm(leaderboard, out_dir)
        fig_category(leaderboard, out_dir)
        fig_latency(leaderboard, out_dir)

    significance = _load_json(significance_path)
    if significance is not None:
        fig_significance(significance, out_dir)

    moneydata = _load_json(moneydata_path)
    if moneydata is not None:
        fig_moneydata(moneydata, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    ap.add_argument("--significance", type=Path, default=DEFAULT_SIGNIFICANCE)
    ap.add_argument("--moneydata", type=Path, default=DEFAULT_MONEYDATA)
    args = ap.parse_args()
    run(args.leaderboard, args.significance, args.moneydata, args.out_dir)
    print("make_figures: no longer generated (orphaned on disk if present): "
         + ", ".join(ORPHANED_FIGURES))


if __name__ == "__main__":
    main()
