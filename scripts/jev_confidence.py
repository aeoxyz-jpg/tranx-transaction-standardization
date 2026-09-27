"""Record Jev's merchant-choice confidence per row and test whether a confidence
threshold separates right from wrong picks.

Datasets:
  synthetic hard feed, random + unseen splits (1000 eval rows each, same rows as
  the leaderboard); MoneyData real descriptors, known list + 20%-held-out list.

Writes reports/real/jev_confidence_rows.parquet (per row, gitignored) and
reports/jev_confidence_summary.json. Usage: python scripts/jev_confidence.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from tranx import config
from tranx.cli import _split_random, _split_unseen, _cap_eval
from tranx.eval.metrics import _norm_merchant as norm
from tranx.pipeline.clean import strip_processor_prefix
from tranx.routes.jev import JevRoute, build_request, _jev_call, JEV_MODEL, NONE_OPTION
from tranx.schema import Txn
import eval_moneydata as md

TXN = ["txn_id", "customer_id", "description", "transaction_type_code", "mcc", "amount",
       "payment_method", "posted_date", "country", "currency"]
THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def parse(ans: dict, key: str) -> dict:
    a = ans.get("answers", {}).get(key, {})
    probs = a.get("probabilities", {}) or {}
    choice = a.get("choice")
    return {"choice": choice, "confidence": a.get("confidence"),
            "p_choice": probs.get(choice), "p_none": probs.get(NONE_OPTION)}


def synthetic_rows():
    feed = pl.read_parquet(config.DATA_DIR / "bank_feed.parquet")
    gold = pl.read_parquet(config.DATA_DIR / "gold.parquet")
    out = []
    for split, fn in (("random", _split_random), ("unseen", _split_unseen)):
        tf, tg, ef, eg = fn(feed, gold, config.SEED)
        ef, eg = _cap_eval(ef, eg, 1000, config.SEED)
        # merchant question only makes sense where the gold has a merchant
        eg = eg.filter(pl.col("canonical_merchant").is_not_null())
        ef = ef.filter(pl.col("txn_id").is_in(eg["txn_id"]))
        route = JevRoute(merchant_choice=True)
        route.fit(tf, tg)
        g = dict(zip(eg["txn_id"], eg["canonical_merchant"]))
        gc = dict(zip(eg["txn_id"], eg["category"]))
        vocab = {norm(v) for v in tg["canonical_merchant"].drop_nulls().unique()}
        txns = [Txn(**r) for r in ef.select(TXN).iter_rows(named=True)]

        def one(t):
            cands = route._candidates(strip_processor_prefix(t.description))
            ans = _jev_call(build_request(t, route._categories, JEV_MODEL, cands))
            m, c = parse(ans, "merchant"), parse(ans, "category")
            gold_m = g[t.txn_id]
            return {"dataset": "synthetic", "setting": split, "description": t.description,
                    "gold": gold_m, **m,
                    "correct": m["choice"] != NONE_OPTION and norm(m["choice"] or "") == norm(gold_m),
                    "gold_in_list": norm(gold_m) in vocab,
                    "gold_in_cands": norm(gold_m) in {norm(x) for x in cands},
                    "cat_choice": c["choice"], "cat_confidence": c["confidence"],
                    "cat_correct": c["choice"] == gc[t.txn_id]}
        with ThreadPoolExecutor(max_workers=8) as ex:
            out += list(ex.map(one, txns))
        print(f"synthetic {split}: {len(txns)} rows", flush=True)
    return out


def moneydata_rows():
    all_lab = pl.read_csv(config.DATA_DIR / "real" / "moneydata_labels.csv").filter(
        pl.col("canonical_merchant") != md.NOT_MERCHANT)
    lab = all_lab.filter(pl.col("confidence").is_in(["high", "medium"]))
    for r in pl.read_csv(config.DATA_DIR / "real" / "moneydata_aliases.csv").iter_rows(named=True):
        md.ALIASES[r["canonical_merchant"]] = {norm(a) for a in (r["aliases"] or "").split("|") if a.strip()}
    descs, gold, n = lab["description"].to_list(), lab["canonical_merchant"].to_list(), lab["n"].to_list()
    vocab_all = sorted(set(gold))
    # same realistic list as eval_moneydata.py: merchants seen in >= 2 labelled descriptors
    seen = all_lab.group_by("canonical_merchant").len()
    listed = set(seen.filter(pl.col("len") >= 2)["canonical_merchant"])
    vocab_hold = [v for v in vocab_all if v in listed]
    out = []
    for setting, vocab in (("known", vocab_all), ("hold", vocab_hold)):
        vset = set(vocab)

        def one(i):
            d, g = descs[i], gold[i]
            cands = md.fuzzy_top(d, vocab, 20)
            crit = {c: None for c in cands}
            crit[NONE_OPTION] = "None of the listed merchants is the one in the description"
            ans = _jev_call({"state": {"description": d}, "model": JEV_MODEL, "questions": {"merchant": {
                "type": "choice",
                "instructions": "Which known merchant does this bank transaction description refer to?",
                "criteria": crit}}})
            m = parse(ans, "merchant")
            return {"dataset": "moneydata", "setting": setting, "description": d, "gold": g, "n": n[i], **m,
                    "correct": m["choice"] != NONE_OPTION and norm(m["choice"] or "") in md.accepted(g),
                    "gold_in_list": g in vset, "gold_in_cands": g in cands}
        with ThreadPoolExecutor(max_workers=8) as ex:
            out += list(ex.map(one, range(len(descs))))
        print(f"moneydata {setting}: {len(descs)} rows", flush=True)
    return out


def auroc(pos, neg):
    """P(confidence of a right pick > confidence of a wrong pick); ties count half."""
    if not pos or not neg:
        return None
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def summarize(rows: pl.DataFrame) -> dict:
    res = {}
    for (ds, st), d in rows.group_by(["dataset", "setting"], maintain_order=True):
        acc = d.filter(pl.col("choice") != NONE_OPTION)
        right = acc.filter(pl.col("correct"))["confidence"].to_list()
        wrong = acc.filter(~pl.col("correct"))["confidence"].to_list()
        q = lambda xs: [round(float(np.quantile(xs, p)), 3) for p in (0.1, 0.25, 0.5, 0.75, 0.9)] if xs else []
        sweep = []
        for t in THRESHOLDS:
            a = d.filter((pl.col("choice") != NONE_OPTION) & (pl.col("confidence") >= t))
            sweep.append({"threshold": t,
                          "auto_accepted": round(len(a) / len(d), 3),
                          "precision_of_accepted": round(float(a["correct"].mean()), 3) if len(a) else None,
                          "wrong_accepts_of_all_rows": round(float((~a["correct"]).sum() / len(d)), 3),
                          "escalated": round(1 - len(a) / len(d), 3)})
        res[f"{ds}/{st}"] = {
            "rows": len(d), "gold_in_list": round(float(d["gold_in_list"].mean()), 3),
            "picked_none": round(float((d["choice"] == NONE_OPTION).mean()), 3),
            "accepted_right": len(right), "accepted_wrong": len(wrong),
            "confidence_quantiles_right_p10_25_50_75_90": q(right),
            "confidence_quantiles_wrong_p10_25_50_75_90": q(wrong),
            "auroc_right_vs_wrong": None if auroc(right, wrong) is None else round(auroc(right, wrong), 3),
            "sweep": sweep}
    return res


def main():
    out_rows = config.REPORTS_DIR / "real" / "jev_confidence_rows.parquet"
    rows = pl.DataFrame(synthetic_rows() + moneydata_rows(), infer_schema_length=None)
    rows.write_parquet(out_rows)
    summary = summarize(rows)
    (config.REPORTS_DIR / "jev_confidence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
