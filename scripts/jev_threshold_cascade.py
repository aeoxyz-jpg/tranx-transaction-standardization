"""Exact accuracy of the Jev -> SLM merchant cascade as a function of a Jev
confidence threshold: accept Jev's pick when it is not none_of_these and its
confidence >= t, otherwise use the few-shot SLM's merchant.

Reuses the per-row Jev answers from scripts/jev_confidence.py (no new Jev calls).
Runs the SLM once per synthetic eval row (cached to
reports/real/slm_synthetic_rows.parquet); MoneyData SLM predictions come from
scripts/eval_moneydata.py. Writes reports/jev_threshold_cascade.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from tranx import config
from tranx.cli import _split_random, _split_unseen, _cap_eval
from tranx.eval.metrics import _norm_merchant as norm
from tranx.routes.jev import NONE_OPTION
from tranx.routes.slm_fewshot import SlmFewshotRoute
from tranx.schema import Txn
import eval_moneydata as md

TXN = ["txn_id", "customer_id", "description", "transaction_type_code", "mcc", "amount",
       "payment_method", "posted_date", "country", "currency"]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
REAL = config.REPORTS_DIR / "real"


def synthetic_slm() -> pl.DataFrame:
    cache = REAL / "slm_synthetic_rows.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    feed = pl.read_parquet(config.DATA_DIR / "bank_feed.parquet")
    gold = pl.read_parquet(config.DATA_DIR / "gold.parquet")
    out = []
    for split, fn in (("random", _split_random), ("unseen", _split_unseen)):
        tf, tg, ef, eg = fn(feed, gold, config.SEED)
        ef, eg = _cap_eval(ef, eg, 1000, config.SEED)
        eg = eg.filter(pl.col("canonical_merchant").is_not_null())  # same rows as jev_confidence
        ef = ef.filter(pl.col("txn_id").is_in(eg["txn_id"]))
        route = SlmFewshotRoute()
        route.fit(tf, tg)
        for i, r in enumerate(ef.select(TXN).iter_rows(named=True)):
            out.append({"setting": split, "i": i, "description": r["description"],
                        "slm": route.standardize(Txn(**r)).canonical_merchant})
        print(f"slm {split}: {len(ef)} rows", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(cache)
    return df


def sweep(d: pl.DataFrame, is_ok) -> list[dict]:
    choice, conf, slm, gold = (d[c].to_list() for c in ("choice", "confidence", "slm", "gold"))
    n = d["n"].to_numpy() if "n" in d.columns else np.ones(len(d))
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


def main():
    rows = pl.read_parquet(REAL / "jev_confidence_rows.parquet")
    res = {}
    slm = synthetic_slm()
    for split in ("random", "unseen"):
        j = rows.filter((pl.col("dataset") == "synthetic") & (pl.col("setting") == split))
        s = slm.filter(pl.col("setting") == split).sort("i")
        assert j["description"].to_list() == s["description"].to_list(), "row order mismatch"
        d = j.with_columns(s["slm"])
        res[f"synthetic/{split}"] = sweep(d, lambda p, g: norm(p or "") == norm(g or ""))

    for r in pl.read_csv(config.DATA_DIR / "real" / "moneydata_aliases.csv").iter_rows(named=True):
        md.ALIASES[r["canonical_merchant"]] = {norm(a) for a in (r["aliases"] or "").split("|") if a.strip()}
    mslm = pl.read_parquet(REAL / "moneydata_preds_high-medium.parquet").select("description", "slm")
    for setting in ("known", "hold"):
        d = rows.filter((pl.col("dataset") == "moneydata") & (pl.col("setting") == setting)).join(
            mslm, on="description", how="left")
        assert d["slm"].null_count() == 0
        res[f"moneydata/{setting}"] = sweep(d, lambda p, g: norm(p or "") in md.accepted(g))

    (config.REPORTS_DIR / "jev_threshold_cascade.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print("==", k)
        for r in v:
            print("  ", r)


if __name__ == "__main__":
    main()
