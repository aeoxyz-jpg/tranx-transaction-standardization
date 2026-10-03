"""In-house cascade: fuzzy match, else SLM (exploratory, 2026-10-03).

The README's alternative to Jev was compared with fuzzy matching alone. A bank that
cannot send descriptors to a third party would instead run fuzzy matching (or
embeddings) for known merchants and the SLM for the rest. This script scores that
cascade against jev_slm from saved predictions only; no model or API calls.

- Synthetic: the rules route is refit on each split's model-view train rows to recover
  which eval descriptors had a fuzzy match (score >= 85); its merchant answers must
  reproduce the saved rules predictions exactly. Matched rows take the rules answer
  and category, the rest the slm_fewshot answer and category. Scored like the
  leaderboard headline (merchant rows, abbreviated locals excluded).
- MoneyData realistic list: fuzzy_hold is a top-1 match without a cutoff, so its
  token_set_ratio is recomputed and gated at 85; below the gate, the SLM's answer.
  The embedding cascade (cosine >= 0.6, else SLM) is scored for comparison.

Usage: PYTHONPATH=. python3.11 scripts/inhouse_cascade.py [--manifest PATH] [--preds-dir DIR] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from tranx import config
from tranx.cli import eval_rows
from tranx.eval.manifest import load_preds
from tranx.eval.metrics import _norm_merchant as norm, merchant_ok
from tranx.eval.significance import paired_bootstrap
from tranx.pipeline.clean import clean_description
from tranx.routes.rules import RulesRoute

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_moneydata as em  # noqa: E402

FUZZY_CUTOFF = 85  # RulesRoute default
EMBED_GATE = 0.6
MONEYDATA_PREDS = config.REPORTS_DIR / "real" / "moneydata_preds_high-medium.parquet"
MONEYDATA_ALIASES = Path("data/real/moneydata_aliases.csv")


def synthetic_rows(split: str, manifest: dict, preds_dir: Path) -> pl.DataFrame:
    """One row per headline merchant row with rules, slm, jev_slm answers and the
    rules match flag."""
    er = eval_rows(split, "model")
    route = RulesRoute(score_cutoff=FUZZY_CUTOFF)
    route.fit(er.train_feed, er.train_gold)
    desc = er.eval_feed.select("txn_id", "description")
    flags = [route._match_merchant_full(d) for d in desc["description"].to_list()]
    desc = desc.with_columns(pl.Series("refit_merchant", [m for m, _ in flags]),
                             pl.Series("matched", [ok for _, ok in flags]))

    def pick(route_name, alias):
        return load_preds(split, "model", route_name, manifest, preds_dir).select(
            "txn_id", pl.col("pred_merchant").alias(alias), pl.col("pred_category").alias(f"{alias}_cat"))

    rules = load_preds(split, "model", "rules", manifest, preds_dir).select(
        "txn_id", "gold_merchant", "gold_category", "origin", "noise_abbrev",
        pl.col("pred_merchant").alias("rules"), pl.col("pred_category").alias("rules_cat"))
    d = (rules.join(pick("slm_fewshot", "slm"), on="txn_id")
              .join(pick("jev_slm", "jev_slm"), on="txn_id")
              .join(desc, on="txn_id"))
    if len(d) != len(rules):
        raise SystemExit(f"inhouse_cascade: {split} join covers {len(d)} of {len(rules)} rows")
    bad = d.filter(pl.col("refit_merchant") != pl.col("rules")).height
    if bad:
        raise SystemExit(f"inhouse_cascade: refit rules differs from saved preds on {bad} {split} rows")
    d = d.filter(pl.col("gold_merchant").is_not_null())
    d = d.filter(~(pl.col("origin").eq("local") & pl.col("noise_abbrev").fill_null(False)))
    return d.with_columns(
        pl.when(pl.col("matched")).then(pl.col("rules")).otherwise(pl.col("slm")).alias("cascade"),
        pl.when(pl.col("matched")).then(pl.col("rules_cat")).otherwise(pl.col("slm_cat")).alias("cascade_cat"))


def synthetic_summary(d: pl.DataFrame) -> dict:
    gold, gc = d["gold_merchant"].to_list(), d["gold_category"].to_list()

    def ok(col):
        return np.array([merchant_ok(p, g, c is not None and c == x) for p, g, c, x
                         in zip(d[col].to_list(), gold, d[f"{col}_cat"].to_list(), gc)])

    res = {k: ok(k) for k in ("rules", "slm", "cascade", "jev_slm")}
    cl = np.array(gold)
    return {"rows": len(d), "fuzzy_matched_share": round(float(d["matched"].mean()), 3),
            "accuracy": {k: round(float(v.mean()), 3) for k, v in res.items()},
            "jev_slm_minus_cascade": paired_bootstrap(res["jev_slm"], res["cascade"], cl,
                                                      n_boot=config.BOOTSTRAP_N, seed=0)}


def moneydata_summary(preds_path: Path, aliases_path: Path) -> dict:
    if aliases_path.exists():
        for r in pl.read_csv(aliases_path).iter_rows(named=True):
            em.ALIASES[r["canonical_merchant"]] = {norm(a) for a in (r["aliases"] or "").split("|")
                                                    if a.strip()}
    p = pl.read_parquet(preds_path)
    gold, n = p["canonical_merchant"].to_list(), p["n"].to_numpy()
    # Same strings eval_moneydata.fuzzy_top compares, so this is the score of its top-1 hit.
    score = np.array([fuzz.token_set_ratio(clean_description(em.query(d)), clean_description(f))
                      for d, f in zip(p["description"].to_list(), p["fuzzy_hold"].to_list())])
    fuzzy_ok = score >= FUZZY_CUTOFF
    embed_ok = p["embed_hold_cos"].to_numpy() >= EMBED_GATE
    preds = {
        "slm": p["slm"].to_list(),
        "fuzzy_to_slm": [f if a else s for f, s, a in zip(p["fuzzy_hold"], p["slm"], fuzzy_ok)],
        "embedding_to_slm": [e if a else s for e, s, a in zip(p["embed_hold"], p["slm"], embed_ok)],
        "jev_to_slm": [j if j else s for j, s in zip(p["jev_hold"], p["slm"])],
    }
    ok = {k: np.array([norm(x or "") in em.accepted(g) for x, g in zip(v, gold)]) for k, v in preds.items()}
    cl = np.array(gold)

    def diff(a, b, weights=None):
        return paired_bootstrap(ok[a], ok[b], cl, weights=weights, n_boot=config.BOOTSTRAP_N, seed=0)

    return {
        "descriptors": len(p), "rows": int(n.sum()),
        "sent_to_slm": {"fuzzy_to_slm": round(float(1 - fuzzy_ok.mean()), 3),
                        "embedding_to_slm": round(float(1 - embed_ok.mean()), 3),
                        "jev_to_slm": round(float(np.mean([j == "" for j in p["jev_hold"]])), 3)},
        "per_descriptor": {k: round(float(v.mean()), 3) for k, v in ok.items()},
        "per_row": {k: round(float((v * n).sum() / n.sum()), 3) for k, v in ok.items()},
        "jev_to_slm_minus_fuzzy_to_slm": diff("jev_to_slm", "fuzzy_to_slm"),
        "jev_to_slm_minus_fuzzy_to_slm_row_weighted": diff("jev_to_slm", "fuzzy_to_slm", n),
        "jev_to_slm_minus_embedding_to_slm": diff("jev_to_slm", "embedding_to_slm"),
        "jev_to_slm_minus_embedding_to_slm_row_weighted": diff("jev_to_slm", "embedding_to_slm", n),
        "fuzzy_to_slm_minus_slm": diff("fuzzy_to_slm", "slm"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.RUN_DIR / "manifest.json"))
    ap.add_argument("--preds-dir", default=str(config.PREDS_DIR))
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "inhouse_cascade.json"))
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    res = {"manifest_hash": manifest["manifest_hash"],
           "cascade": f"fuzzy match (token_set_ratio >= {FUZZY_CUTOFF}) else SLM; "
                      f"MoneyData also embedding (cosine >= {EMBED_GATE}) else SLM",
           **{s: synthetic_summary(synthetic_rows(s, manifest, Path(args.preds_dir)))
              for s in ("random", "unseen")},
           "moneydata_realistic_list": moneydata_summary(MONEYDATA_PREDS, MONEYDATA_ALIASES)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
