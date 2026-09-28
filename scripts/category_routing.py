"""Does routing the category by Jev's merchant answer help? (exploratory, 2026-09-28)

Rule tested: when Jev's merchant Choice is none_of_these (or off the candidate list),
take Jev's category; otherwise take the embedding route's category. Compared with
each route alone on the synthetic model view, overall, by Jev's answer and by
merchant origin, with a paired merchant-cluster bootstrap. Scores saved predictions
only (through the manifest); no model or API calls.

Usage: python3.11 scripts/category_routing.py [--manifest PATH] [--preds-dir DIR] [--out PATH]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from tranx import config
from tranx.eval.manifest import load_preds
from tranx.eval.significance import paired_bootstrap


def routing_rows(split: str, manifest: dict, preds_dir: Path) -> pl.DataFrame:
    e = load_preds(split, "model", "embedding", manifest, preds_dir).select(
        "txn_id", "gold_category", "gold_merchant", "gold_txn_type", "origin",
        pl.col("pred_category").alias("embedding"))
    j = load_preds(split, "model", "jev_merchant", manifest, preds_dir).select(
        "txn_id", pl.col("pred_category").alias("jev"), "jev_choice", "candidates")
    d = e.join(j, on="txn_id", how="inner")
    if len(d) != len(e):
        raise SystemExit(f"category_routing: {split} join covers {len(d)} of {len(e)} rows")
    none = [c is None or cand is None or c not in list(cand)
            for c, cand in zip(d["jev_choice"].to_list(), d["candidates"].to_list())]
    d = d.with_columns(pl.Series("jev_none", none))
    return d.with_columns(pl.when(pl.col("jev_none")).then(pl.col("jev"))
                          .otherwise(pl.col("embedding")).alias("rule"))


def summarize(d: pl.DataFrame) -> dict:
    gold = d["gold_category"]
    ok = {k: (d[k] == gold).to_numpy() for k in ("embedding", "jev", "rule")}
    # Merchant-less rows cluster by transaction type, as in significance.py.
    cl = np.array([m if m is not None else f"type:{t}"
                   for m, t in zip(d["gold_merchant"].to_list(), d["gold_txn_type"].to_list())])
    acc = lambda mask: {k: round(float(ok[k][mask].mean()), 3) for k in ok} | {"rows": int(mask.sum())}
    origin = d["origin"].fill_null("no merchant").to_numpy()
    none = d["jev_none"].to_numpy()
    return {
        "rows": len(d),
        "jev_none_share": round(float(none.mean()), 3),
        "accuracy": acc(np.ones(len(d), bool)),
        "rule_minus_embedding": paired_bootstrap(ok["rule"], ok["embedding"], cl),
        "rule_minus_jev": paired_bootstrap(ok["rule"], ok["jev"], cl),
        "by_jev_answer": {"none_of_these": acc(none), "picked": acc(~none)},
        "by_origin": {o: acc(origin == o) for o in sorted(set(origin))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.RUN_DIR / "manifest.json"))
    ap.add_argument("--preds-dir", default=str(config.PREDS_DIR))
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "category_routing.json"))
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    res = {"manifest_hash": manifest["manifest_hash"],
           "rule": "Jev category when Jev's merchant answer is none_of_these, else embedding",
           **{s: summarize(routing_rows(s, manifest, Path(args.preds_dir))) for s in ("random", "unseen")}}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
