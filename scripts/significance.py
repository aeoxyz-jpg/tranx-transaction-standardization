"""Paired significance tests on saved route predictions (spec D6).

Reads no model, makes no API calls: everything here scores parquets already
written by `eval`, `eval_moneydata.py` and `eval_ddt.py`. Synthetic comparisons
go through reports/run/manifest.json + tranx.eval.manifest.load_preds, so a
stale or hash-mismatched predictions file is refused rather than silently
scored. MoneyData and DDT predictions are read directly from their fixed
reports/real/ paths (they carry no manifest).

Primary comparison (pre-registered, spec D6): jev_slm vs slm_fewshot merchant
accuracy on the synthetic unseen split, and jev_slm vs slm on MoneyData.
Everything else is exploratory and labelled as such.

Usage: python3.11 scripts/significance.py [--manifest PATH] [--preds-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from tranx import config
from tranx.eval.manifest import load_preds
from tranx.eval.metrics import _norm_merchant as norm, merchant_ok
from tranx.eval.significance import paired_bootstrap, leave_one_cluster_out

# eval_moneydata.py lives next to this script and is not a package; import it
# as a sibling module for its ALIASES/accepted() and norm helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_moneydata as em  # noqa: E402

MONEYDATA_PREDS = config.REPORTS_DIR / "real" / "moneydata_preds_high-medium.parquet"
MONEYDATA_ALIASES = Path("data/real/moneydata_aliases.csv")
MONEYDATA_LABELS = Path("data/real/moneydata_labels.csv")
DDT_PREDS = config.REPORTS_DIR / "real" / "ddt_preds.parquet"

CI_NOTE = ("These CIs describe a hypothetical population of merchants like these "
          "(the same generator, or the same one person's statements); they are not "
          "a claim about other customers.")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"significance.py: manifest not found: {path}")
    return json.loads(path.read_text())


def _load_pair(split: str, route_a: str, route_b: str, manifest: dict, preds_dir: Path):
    """Load two routes' predictions for the same split/model view. Lets
    ManifestError (hash mismatch, duplicate/partial coverage) propagate."""
    a = load_preds(split, "model", route_a, manifest, preds_dir=preds_dir)
    b = load_preds(split, "model", route_b, manifest, preds_dir=preds_dir)
    return a, b


def synthetic_merchant(manifest: dict, preds_dir: Path, split: str, route_a: str,
                       route_b: str, label: str) -> dict:
    """Merchant normalized-match accuracy, route_a vs route_b, on rows with a
    non-null gold merchant, excluding abbreviated local-merchant rows (D2:
    the full name is not in the input for those). Clusters = gold merchant."""
    try:
        a, b = _load_pair(split, route_a, route_b, manifest, preds_dir)
    except FileNotFoundError as e:
        return {"skipped": f"missing preds: {e}", "label": label}
    j = a.select(["txn_id", "pred_merchant", "pred_category", "gold_merchant", "gold_category",
                  "origin", "noise_abbrev"]).join(
        b.select(["txn_id", pl.col("pred_merchant").alias("pred_merchant_b"),
                  pl.col("pred_category").alias("pred_category_b")]), on="txn_id")
    j = j.filter(pl.col("gold_merchant").is_not_null())
    j = j.filter(~(pl.col("origin").eq("local") & pl.col("noise_abbrev").fill_null(False)))
    gold = j["gold_merchant"].to_list()
    gc = j["gold_category"].to_list()
    ok_a = np.array([merchant_ok(p, g, c is not None and c == x) for p, g, c, x
                     in zip(j["pred_merchant"].to_list(), gold, j["pred_category"].to_list(), gc)])
    ok_b = np.array([merchant_ok(p, g, c is not None and c == x) for p, g, c, x
                     in zip(j["pred_merchant_b"].to_list(), gold, j["pred_category_b"].to_list(), gc)])
    res = paired_bootstrap(ok_a, ok_b, np.array(gold), n_boot=config.BOOTSTRAP_N, seed=0)
    res["label"] = label
    return res


def synthetic_category(manifest: dict, preds_dir: Path, split: str, route_a: str,
                       route_b: str, label: str) -> dict:
    """Category exact-match accuracy, route_a vs route_b, on all rows. Clusters
    = gold merchant, or gold txn_type for merchant-less rows."""
    try:
        a, b = _load_pair(split, route_a, route_b, manifest, preds_dir)
    except FileNotFoundError as e:
        return {"skipped": f"missing preds: {e}", "label": label}
    j = a.select(["txn_id", "pred_category", "gold_category", "gold_merchant", "gold_txn_type"]).join(
        b.select(["txn_id", pl.col("pred_category").alias("pred_category_b")]), on="txn_id")
    merchants = j["gold_merchant"].to_list()
    txn_types = j["gold_txn_type"].to_list()
    cluster = [m if m is not None else t for m, t in zip(merchants, txn_types)]
    gold_cat = j["gold_category"].to_list()
    ok_a = np.array([p == g for p, g in zip(j["pred_category"].to_list(), gold_cat)])
    ok_b = np.array([p == g for p, g in zip(j["pred_category_b"].to_list(), gold_cat)])
    res = paired_bootstrap(ok_a, ok_b, np.array(cluster), n_boot=config.BOOTSTRAP_N, seed=0)
    res["label"] = label
    return res


def moneydata_section(preds_path: Path, aliases_path: Path, labels_path: Path):
    """Returns (primary_entry, exploratory_entries, filter_note)."""
    if not preds_path.exists():
        return ({"skipped": f"missing preds: {preds_path}", "label": "primary"}, {},
                "MoneyData preds file missing")
    if aliases_path.exists():
        for r in pl.read_csv(aliases_path).iter_rows(named=True):
            em.ALIASES[r["canonical_merchant"]] = {norm(a) for a in (r["aliases"] or "").split("|")
                                                    if a.strip()}
    p = pl.read_parquet(preds_path)
    gold = p["canonical_merchant"].to_list()
    n = p["n"].to_numpy()
    cl = np.array(gold)

    def ok(col):
        return np.array([norm(x or "") in em.accepted(g) for x, g in zip(p[col].to_list(), gold)])

    # Resolution 11: MoneyData "jev_slm" = jev_hold, falling back to slm on "".
    jev_slm = [j if j else s for j, s in zip(p["jev_hold"], p["slm"])]
    ok_jev_slm = np.array([norm(x or "") in em.accepted(g) for x, g in zip(jev_slm, gold)])
    ok_slm = ok("slm")

    primary = paired_bootstrap(ok_jev_slm, ok_slm, cl, n_boot=config.BOOTSTRAP_N, seed=0)
    primary["label"] = "primary"
    # Point-only row-weighted diff, Amazon included (n_boot does not affect the
    # observed diff, only the CI, so a single resample is enough here).
    primary["row_weighted_point_with_amazon"] = paired_bootstrap(
        ok_jev_slm, ok_slm, cl, weights=n, n_boot=1, seed=0)["diff"]
    not_amazon = cl != "Amazon"
    primary["row_weighted_ci_without_amazon"] = paired_bootstrap(
        ok_jev_slm[not_amazon], ok_slm[not_amazon], cl[not_amazon], weights=n[not_amazon],
        n_boot=config.BOOTSTRAP_N, seed=0)
    lo, hi = leave_one_cluster_out(ok_jev_slm, ok_slm, cl)
    primary["lomo_range"] = {"lo": float(lo), "hi": float(hi)}

    exploratory = {}
    if {"jev_known", "embed_known"} <= set(p.columns):
        res = paired_bootstrap(ok("jev_known"), ok("embed_known"), cl, n_boot=config.BOOTSTRAP_N, seed=0)
        res["label"] = "exploratory"
        exploratory["moneydata_known"] = res

    n_low = 0
    if labels_path.exists():
        all_lab = pl.read_csv(labels_path)
        if "confidence" in all_lab.columns:
            n_low = all_lab.filter(pl.col("confidence") == "low").height
    note = f"{n_low} low-confidence labels excluded (kept confidence: high, medium)"
    return primary, exploratory, note


def ddt_section(ddt_path: Path) -> dict:
    """Embedding vs Jev category accuracy on DDT, descriptor-paired (each row
    is a distinct descriptor), clustered by row index, not by merchant: DDT
    rows are not merchant-labelled here, and template correlation across rows
    is not controlled."""
    if not ddt_path.exists():
        return {"skipped": f"missing preds: {ddt_path}", "label": "exploratory"}
    d = pl.read_parquet(ddt_path)
    ok_embed = (d["embedding"] == d["category"]).to_numpy().astype(float)
    ok_jev = (d["jev"] == d["category"]).to_numpy().astype(float)
    res = paired_bootstrap(ok_embed, ok_jev, np.arange(len(d)), n_boot=config.BOOTSTRAP_N, seed=0)
    res["label"] = "exploratory"
    return res


# Human-readable "A - B, metric, data" for each entry (figures and README use it).
COMPARISONS = {
    "synthetic_unseen": "jev_slm - slm_fewshot, merchant, synthetic unseen",
    "moneydata": "jev_slm - slm, merchant, MoneyData realistic list",
    "synthetic_random_merchant": "jev_merchant - embedding, merchant, synthetic random",
    "synthetic_unseen_merchant": "jev_merchant - embedding, merchant, synthetic unseen",
    "synthetic_random_category": "embedding - jev_merchant, category, synthetic random",
    "synthetic_unseen_category": "embedding - jev_merchant, category, synthetic unseen",
    "moneydata_known": "jev - embedding, merchant, MoneyData full list",
    "ddt_category": "embedding - jev, category, DoDataThings",
}


def build_report(manifest_path: Path, preds_dir: Path,
                 moneydata_preds_path: Path = MONEYDATA_PREDS,
                 aliases_path: Path = MONEYDATA_ALIASES,
                 labels_path: Path = MONEYDATA_LABELS,
                 ddt_path: Path = DDT_PREDS) -> dict:
    manifest = _load_manifest(manifest_path)

    primary = {"synthetic_unseen": synthetic_merchant(
        manifest, preds_dir, "unseen", "jev_slm", "slm_fewshot", "primary")}
    exploratory = {
        "synthetic_random_merchant": synthetic_merchant(
            manifest, preds_dir, "random", "jev_merchant", "embedding", "exploratory"),
        "synthetic_unseen_merchant": synthetic_merchant(
            manifest, preds_dir, "unseen", "jev_merchant", "embedding", "exploratory"),
        "synthetic_random_category": synthetic_category(
            manifest, preds_dir, "random", "embedding", "jev_merchant", "exploratory"),
        "synthetic_unseen_category": synthetic_category(
            manifest, preds_dir, "unseen", "embedding", "jev_merchant", "exploratory"),
    }

    md_primary, md_exploratory, filter_note = moneydata_section(
        moneydata_preds_path, aliases_path, labels_path)
    primary["moneydata"] = md_primary
    exploratory.update(md_exploratory)
    exploratory["ddt_category"] = ddt_section(ddt_path)

    for section in (primary, exploratory):
        for key, entry in section.items():
            entry["comparison"] = COMPARISONS[key]
    return {"primary": primary, "exploratory": exploratory, "manifest_hash": manifest["manifest_hash"],
           "notes": {"moneydata_filter": filter_note, "ci_interpretation": CI_NOTE}}


def _fmt(x) -> str:
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def _table(entries: dict) -> str:
    header = "| comparison | diff | lo | hi | n_clusters | n_rows |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for name, r in entries.items():
        if "skipped" in r:
            rows.append(f"| {name} | _skipped: {r['skipped']}_ | | | | |")
            continue
        rows.append(f"| {r.get('comparison', name)} | {_fmt(r['diff'])} | {_fmt(r['lo'])} | {_fmt(r['hi'])} "
                    f"| {r['n_clusters']} | {r['n_rows']} |")
    return "\n".join(rows)


def render_md(report: dict) -> str:
    lines = ["# Significance", "", CI_NOTE, ""]
    lines.append(f"MoneyData: {report['notes']['moneydata_filter']}.")
    lines.append("")
    lines.append("## Primary")
    lines.append("")
    lines.append(_table(report["primary"]))
    md = report["primary"].get("moneydata", {})
    if "skipped" not in md:
        lines += ["", "MoneyData row-weighted point estimate (Amazon included): "
                      f"{_fmt(md['row_weighted_point_with_amazon'])}.",
                  "MoneyData row-weighted CI (Amazon excluded): "
                  f"{_fmt(md['row_weighted_ci_without_amazon']['diff'])} "
                  f"[{_fmt(md['row_weighted_ci_without_amazon']['lo'])}, "
                  f"{_fmt(md['row_weighted_ci_without_amazon']['hi'])}].",
                  "MoneyData leave-one-merchant-out range: "
                  f"[{_fmt(md['lomo_range']['lo'])}, {_fmt(md['lomo_range']['hi'])}]."]
    lines += ["", "## Exploratory", "",
             "Not pre-registered; DDT clusters by row (descriptor-paired), not by merchant, "
             "so template correlation across rows is not controlled.", ""]
    lines.append(_table(report["exploratory"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.RUN_DIR / "manifest.json"))
    ap.add_argument("--preds-dir", default=str(config.PREDS_DIR))
    ap.add_argument("--out-dir", default=str(config.REPORTS_DIR))
    args = ap.parse_args()

    report = build_report(Path(args.manifest), Path(args.preds_dir))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "significance.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / "significance.md").write_text(render_md(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
