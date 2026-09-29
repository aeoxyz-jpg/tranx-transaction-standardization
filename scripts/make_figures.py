"""Generate the README figures from this run's recorded results.

No numbers are hardcoded: everything is read from
  - reports/leaderboard_hard_jev.json  (written by `python -m tranx.cli eval
    --name leaderboard_hard_jev.md`; a list of per-route/split/view result dicts)
  - reports/significance.json          (written by scripts/significance.py)
  - reports/real/moneydata_summary_high-medium_aliased.json (committed)

Every figure is written twice, `<name>.png` for GitHub's light theme and
`<name>-dark.png` for its dark theme, both on a transparent background, so the
README can switch between them with <picture>. Any missing input file is skipped
with a printed warning; the figures that depend on it are not produced.
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

# Preferred route order; any route present in the data but not listed is appended.
MERCHANT_NORM_ROUTES = ["cleaner", "rules", "embedding", "slm_fewshot", "jev_merchant", "jev_slm"]
CATEGORY_ROUTES = ["rules", "embedding", "slm_fewshot", "jev_merchant", "jev_slm"]
LATENCY_ROUTE_ORDER = ["cleaner", "metadata", "rules", "embedding", "jev_merchant",
                       "slm_fewshot", "jev_slm"]
ROUTE_LABELS = {"cleaner": "cleaner (baseline)", "metadata": "metadata (baseline)",
                "rules": "rules", "embedding": "embedding", "slm_fewshot": "SLM few-shot",
                "jev_merchant": "Jev", "jev_slm": "Jev → SLM"}
SPLIT_LABELS = {"random": "known merchants", "unseen": "new merchants"}

ORPHANED_FIGURES = ["lora_vs_fewshot.png", "table_engines.png", "table_results.png"]

# Categorical slots 1-3 of the dataviz reference palette, stepped per theme and
# validated against GitHub's surfaces (#ffffff light, #0d1117 dark). Slot 3 is
# below 3:1 on white, so every bar that uses it carries a value label.
THEMES = {
    "light": {"suffix": "", "ink": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "neutral": "#8c959f"},
    "dark": {"suffix": "-dark", "ink": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "series": ["#3987e5", "#d95926", "#199e70"], "neutral": "#6e7681"},
}


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


def _style(ax, t, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink"], labelsize=10)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])


def _title(ax, t, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["ink"],
                 pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=t["muted"],
                va="bottom")


def _legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=10, **kw)
    for text in leg.get_texts():
        text.set_color(t["ink"])


def _both(draw, out_dir: Path, name: str):
    for t in THEMES.values():
        fig = draw(t)
        path = out_dir / f"{name}{t['suffix']}.png"
        fig.savefig(path, dpi=160, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


def fig_overview(leaderboard, out_dir: Path):
    """Headline trade-off: merchant accuracy on known vs new merchants per method,
    as a dumbbell (one row per method, one dot per split)."""
    idx = _rows_by_route_split(leaderboard, "model")
    routes = [r for r in _order(list(idx), MERCHANT_NORM_ROUTES) if r in MERCHANT_NORM_ROUTES
              and all(_num(idx[r].get(s, {}).get("merchant_norm")) is not None
                      for s in ("random", "unseen"))]
    if not routes:
        print("make_figures: no merchant_norm data, skipping overview.png")
        return

    def draw(t):
        fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(routes) + 1.6))
        ys = list(range(len(routes)))[::-1]
        for y, r in zip(ys, routes):
            k = idx[r]["random"]["merchant_norm"]
            n = idx[r]["unseen"]["merchant_norm"]
            ax.plot([k, n], [y, y], color=t["grid"], linewidth=2, zorder=1)
            ax.scatter([k], [y], s=90, color=t["series"][0], zorder=3)
            ax.scatter([n], [y], s=90, color=t["series"][1], zorder=3)
            lo, hi = min(k, n), max(k, n)
            ax.text(lo - 0.025, y, f"{lo:.2f}", ha="right", va="center", fontsize=9,
                    color=t["ink"])
            ax.text(hi + 0.025, y, f"{hi:.2f}", ha="left", va="center", fontsize=9,
                    color=t["ink"])
        ax.scatter([], [], s=90, color=t["series"][0], label=SPLIT_LABELS["random"])
        ax.scatter([], [], s=90, color=t["series"][1], label=SPLIT_LABELS["unseen"])
        ax.set_yticks(ys)
        ax.set_yticklabels([ROUTE_LABELS.get(r, r) for r in routes])
        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel("merchant accuracy (cache misses, synthetic feed)")
        _style(ax, t, grid_axis="x")
        _title(ax, t, "Known merchants vs new merchants",
               "List-bound methods collapse on new merchants; only the SLM names them")
        # below the axis, so it never sits on a data row
        _legend(ax, t, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
        return fig

    _both(draw, out_dir, "overview")


def fig_merchant_norm(leaderboard, out_dir: Path):
    """Model view, known vs new merchants, merchant accuracy per route with 95% CI."""
    idx = _rows_by_route_split(leaderboard, "model")
    routes = _order([r for r in MERCHANT_NORM_ROUTES if r in idx], MERCHANT_NORM_ROUTES)
    routes = [r for r in routes if any(_num(idx[r].get(s, {}).get("merchant_norm")) is not None
                                       for s in ("random", "unseen"))]
    if not routes:
        print("make_figures: no merchant_norm data, skipping merchant_norm.png")
        return

    def draw(t):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        w = 0.38
        for i, split in enumerate(("random", "unseen")):
            for j, r in enumerate(routes):
                row = idx[r].get(split, {})
                v = _num(row.get("merchant_norm"))
                if v is None:
                    continue
                e = _num(row.get("merchant_norm_ci")) or 0.0
                x = j + (i - 0.5) * (w + 0.02)
                ax.bar(x, v, width=w, color=t["series"][i],
                       label=SPLIT_LABELS[split] if j == 0 else None)
                if e:
                    ax.errorbar(x, v, yerr=e, fmt="none", ecolor=t["muted"], elinewidth=1,
                                capsize=3)
                ax.text(x, v + e + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                        color=t["ink"])
        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels([ROUTE_LABELS.get(r, r) for r in routes])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("merchant accuracy")
        _style(ax, t)
        _title(ax, t, "Merchant accuracy by method",
               "Cache misses on the synthetic feed; whiskers are 95% intervals")
        _legend(ax, t, loc="upper left", ncol=2)
        return fig

    _both(draw, out_dir, "merchant_norm")


def fig_category(leaderboard, out_dir: Path):
    """Model view, category accuracy per route for known vs new merchants; the
    metadata-only baseline is a dashed reference line."""
    idx = _rows_by_route_split(leaderboard, "model")
    routes = _order([r for r in CATEGORY_ROUTES if r in idx], CATEGORY_ROUTES)
    routes = [r for r in routes if any(_num(idx[r].get(s, {}).get("category_acc")) is not None
                                       for s in ("random", "unseen"))]
    if not routes:
        print("make_figures: no category data, skipping category.png")
        return
    meta = idx.get("metadata", {})
    meta_vals = [v for v in (_num(meta.get(s, {}).get("category_acc")) for s in ("random", "unseen"))
                 if v is not None]

    def draw(t):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        w = 0.38
        for i, split in enumerate(("random", "unseen")):
            for j, r in enumerate(routes):
                v = _num(idx[r].get(split, {}).get("category_acc"))
                if v is None:
                    continue
                x = j + (i - 0.5) * (w + 0.02)
                ax.bar(x, v, width=w, color=t["series"][i],
                       label=SPLIT_LABELS[split].replace("merchants", "brands") if j == 0 else None)
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                        color=t["ink"])
        if meta_vals:
            m = sum(meta_vals) / len(meta_vals)
            # legend entry rather than an in-plot label, which would sit on the bars
            ax.axhline(m, color=t["neutral"], linestyle="--", linewidth=1.2,
                       label=f"metadata only ({m:.2f})")
        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels([ROUTE_LABELS.get(r, r) for r in routes])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("category accuracy")
        _style(ax, t)
        _title(ax, t, "Category accuracy by method",
               "Cache misses on the synthetic feed; the dashed line reads no description")
        _legend(ax, t, loc="upper left", ncol=3)
        return fig

    _both(draw, out_dir, "category")


def fig_latency(leaderboard, out_dir: Path):
    """Milliseconds per transaction (serial, laptop), one bar per route, log scale."""
    idx = _rows_by_route_split(leaderboard, "model")
    vals = {}
    for r, splits in idx.items():
        row = splits.get("random") or splits.get("unseen") or {}
        v = _num(row.get("avg_ms"))
        if v is not None:
            vals[r] = v
    routes = _order(list(vals), LATENCY_ROUTE_ORDER)
    if not routes:
        print("make_figures: no latency data, skipping latency.png")
        return

    def draw(t):
        fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(routes) + 1.4))
        ys = list(range(len(routes)))[::-1]
        for y, r in zip(ys, routes):
            v = vals[r]
            ax.barh(y, v, height=0.6, color=t["series"][0])
            ax.text(v * 1.15, y, f"{v:.2g} ms" if v < 10 else f"{v:.0f} ms", va="center",
                    fontsize=9, color=t["ink"])
        ax.set_xscale("log")
        ax.set_yticks(ys)
        ax.set_yticklabels([ROUTE_LABELS.get(r, r) for r in routes])
        ax.set_xlabel("milliseconds per transaction (log scale)")
        _style(ax, t, grid_axis="x")
        _title(ax, t, "Latency per transaction", "Serial, one request at a time, on a laptop")
        return fig

    _both(draw, out_dir, "latency")


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
    """Forest plot of the paired differences with 95% intervals; primary comparisons
    in the accent color and bold, exploratory ones in neutral gray."""
    rows, skipped = _forest_rows(significance)
    if not rows:
        print("make_figures: no significance entries with a CI, skipping significance.png")
        return

    def draw(t):
        fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.6))
        ys = list(range(len(rows)))[::-1]
        for y, (key, section, e) in zip(ys, rows):
            color = t["series"][0] if section == "primary" else t["neutral"]
            ax.plot([e["lo"], e["hi"]], [y, y], color=color, linewidth=2)
            ax.scatter([e["diff"]], [y], s=60, color=color, zorder=3)
            ax.text(e["hi"] + 0.01, y, f"{e['diff']:+.3f}", va="center", fontsize=9,
                    color=t["ink"])
        ax.axvline(0, color=t["muted"], linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([e.get("comparison", key) for key, _, e in rows], fontsize=9)
        for label, (_, section, _) in zip(ax.get_yticklabels(), rows):
            if section == "primary":
                label.set_fontweight("bold")
        ax.set_xlabel("paired accuracy difference, A − B (95% interval)")
        _style(ax, t, grid_axis="x")
        sub = "Bold: the primary comparison fixed in advance; the rest are exploratory"
        if skipped:
            sub += f"; omitted: {', '.join(skipped)}"
        _title(ax, t, "Paired differences between methods", sub)
        return fig

    _both(draw, out_dir, "significance")


# MoneyData methods shown, with display names (the summary also holds cascade variants).
MONEYDATA_METHODS = {
    "derive": "cleaner", "fuzzy_known": "fuzzy\n(full list)", "embed_known": "embedding\n(full list)",
    "jev_known": "Jev\n(full list)", "slm": "SLM", "cascade_jevnone_to_slm": "Jev → SLM\n(realistic list)",
}


def fig_moneydata(summary: dict, out_dir: Path):
    """MoneyData merchant accuracy per method: per descriptor, spend-weighted, and
    spend-weighted without the largest descriptor. Only methods carrying both of
    the first two numbers are plotted."""
    methods, series = [], ([], [], [])
    for key in MONEYDATA_METHODS:
        entry = summary.get(key)
        if not isinstance(entry, dict):
            continue
        d, s = entry.get("distinct"), entry.get("spend_weighted")
        if isinstance(d, (int, float)) and isinstance(s, (int, float)):
            methods.append(MONEYDATA_METHODS[key])
            series[0].append(d)
            series[1].append(s)
            series[2].append(entry.get("spend_weighted_excl_top") or 0.0)
    if not methods:
        print("make_figures: no MoneyData distinct/spend_weighted pairs, skipping moneydata.png")
        return
    labels = ["per descriptor", "spend-weighted", "spend-weighted, largest descriptor removed"]

    def draw(t):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        w = 0.27
        for i, vals in enumerate(series):
            for j, v in enumerate(vals):
                x = j + (i - 1) * (w + 0.015)
                ax.bar(x, v, width=w, color=t["series"][i], label=labels[i] if j == 0 else None)
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                        color=t["ink"])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("merchant accuracy")
        _style(ax, t)
        _title(ax, t, "Real statements (MoneyData)",
               "One investment transfer carries 19% of debit spend and is missed by every method")
        _legend(ax, t, loc="upper left", ncol=3)
        return fig

    _both(draw, out_dir, "moneydata")


def run(leaderboard_path: Path, significance_path: Path, moneydata_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = _load_json(leaderboard_path)
    if leaderboard is not None:
        fig_overview(leaderboard, out_dir)
        fig_merchant_norm(leaderboard, out_dir)
        fig_category(leaderboard, out_dir)
        fig_latency(leaderboard, out_dir)

    significance = _load_json(significance_path)
    # Both inputs must come from the same eval run (same manifest).
    if leaderboard and significance:
        lb_hashes = {r.get("manifest_hash") for r in leaderboard}
        sig_hash = significance.get("manifest_hash")
        if sig_hash and lb_hashes != {sig_hash}:
            raise SystemExit(f"make_figures: leaderboard {lb_hashes} and significance {sig_hash} "
                             "come from different runs")
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
