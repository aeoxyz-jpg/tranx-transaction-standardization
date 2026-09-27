"""Exact accuracy of the Jev -> SLM merchant cascade as a function of a Jev
confidence threshold: accept Jev's pick when it is not none_of_these and its
confidence >= t, otherwise use the few-shot SLM's merchant.

Synthetic: this run's saved jev_merchant and slm_fewshot model-view predictions
(reports/preds, checked against reports/run/manifest.json), joined on txn_id; no
model calls. MoneyData: Jev answers from scripts/jev_confidence.py's per-row parquet
(or, with --synthetic-only, reports/real/jev_confidence_moneydata_rows.parquet), SLM
predictions from scripts/eval_moneydata.py. Writes reports/jev_threshold_cascade.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from tranx import config
from tranx.eval.manifest import ManifestError, load_preds
from tranx.eval.metrics import _norm_merchant as norm
from tranx.routes.jev import NONE_OPTION
import eval_moneydata as md
import jev_confidence as jc

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
REAL = config.REPORTS_DIR / "real"


def synthetic_cascade_rows(manifest: dict, preds_dir: Path | None, data_dir: Path) -> dict:
    """Per split: Jev rows (choice, confidence, gold, gold_in_list) with the SLM's
    merchant for the same txn_id."""
    jev = jc.synthetic_rows(manifest, preds_dir, data_dir)
    out = {}
    for split in ("random", "unseen"):
        j = jev.filter(pl.col("setting") == split)
        s = load_preds(split, "model", "slm_fewshot", manifest, preds_dir).select(
            "txn_id", pl.col("pred_merchant").alias("slm"))
        d = j.join(s, on="txn_id", how="inner")
        if len(d) != len(j):
            raise ManifestError(f"{split}: slm_fewshot preds cover {len(d)} of {len(j)} jev rows")
        out[split] = d.sort("txn_id")
    return out


def sweep(d: pl.DataFrame, is_ok) -> list[dict]:
    choice, conf, slm, gold = (d[c].to_list() for c in ("choice", "confidence", "slm", "gold"))
    n = d["n"].fill_null(1).to_numpy() if "n" in d.columns else np.ones(len(d))
    unseen = ~d["gold_in_list"].to_numpy()
    rows = []
    for t in [None] + THRESHOLDS:
        accept = np.array([t is not None and c != NONE_OPTION and (x or 0) >= t for c, x in zip(choice, conf)])
        pred = [c if a else s for c, s, a in zip(choice, slm, accept)]
        ok = np.array([is_ok(p, g) for p, g in zip(pred, gold)])
        rows.append({"threshold": "slm_only" if t is None else t,
                     "merchant_acc": round(float(ok.mean()), 3),
                     "row_weighted": round(float((ok * n).sum() / n.sum()), 3),
                     "acc_on_merchants_not_in_list": round(float(ok[unseen].mean()), 3) if unseen.any() else None,
                     "escalated_to_slm": round(float(1 - accept.mean()), 3)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=config.RUN_DIR / "manifest.json")
    ap.add_argument("--preds-dir", type=Path, default=config.PREDS_DIR)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--synthetic-only", action="store_true",
                    help="read MoneyData Jev rows from --moneydata-rows instead of --rows")
    ap.add_argument("--rows", type=Path, default=REAL / "jev_confidence_rows.parquet")
    ap.add_argument("--moneydata-rows", type=Path, default=jc.MONEYDATA_ROWS)
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "jev_threshold_cascade.json")
    args = ap.parse_args(argv)

    if args.synthetic_only:
        rows = jc.saved_moneydata_rows(args.moneydata_rows)
    else:
        rows = pl.read_parquet(args.rows)
    manifest = jc.load_manifest(args.manifest)
    res = {}
    for split, d in synthetic_cascade_rows(manifest, args.preds_dir, args.data_dir).items():
        res[f"synthetic/{split}"] = sweep(d, lambda p, g: norm(p or "") == norm(g or ""))

    for r in pl.read_csv(config.DATA_DIR / "real" / "moneydata_aliases.csv").iter_rows(named=True):
        md.ALIASES[r["canonical_merchant"]] = {norm(a) for a in (r["aliases"] or "").split("|") if a.strip()}
    mslm = pl.read_parquet(REAL / "moneydata_preds_high-medium.parquet").select("description", "slm")
    for setting in ("known", "hold"):
        d = rows.filter((pl.col("dataset") == "moneydata") & (pl.col("setting") == setting)).join(
            mslm, on="description", how="left")
        assert d["slm"].null_count() == 0
        res[f"moneydata/{setting}"] = sweep(d, lambda p, g: norm(p or "") in md.accepted(g))

    res["manifest_hash"] = manifest["manifest_hash"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if k == "manifest_hash":
            continue
        print("==", k)
        for r in v:
            print("  ", r)


if __name__ == "__main__":
    main()
