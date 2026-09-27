"""Merchant-normalization eval on real UK bank descriptors (MoneyData, Firat et
al. 2023, github.com/thevisgroup/MoneyVis). Card (DEB) and direct-debit (DD)
lines only, scored per distinct descriptor and row-weighted.

Settings:
  known — oracle list: contains every gold merchant (upper bound for list-bound methods).
  hold  — realistic list: only merchants seen in >= 2 labelled descriptors; merchants
          seen once are "new" (not on the list). Also runs the cascades to the SLM.
Scores: per descriptor (distinct), row-weighted (Amazon is ~35% of rows), and macro
(mean over merchants). "non_literal" repeats the distinct score without descriptors
whose label equals the cleaned descriptor (string cleaning reproduces those).

Usage: python scripts/eval_moneydata.py --labels data/real/moneydata_labels.csv \
           --methods derive,fuzzy,embedding,slm,jev
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from rapidfuzz import process, fuzz

from tranx import config
from tranx.eval.metrics import _norm_merchant as norm
from tranx.pipeline.clean import strip_processor_prefix, clean_description
from tranx.synth.canonical import derive_canonical
from tranx.routes.slm_fewshot import build_prompt, parse_response, _ollama_chat
from tranx.routes.jev import _jev_call, JEV_MODEL, NONE_OPTION

OUT = config.REPORTS_DIR / "real"
NOT_MERCHANT = "__NOT_MERCHANT__"


def query(desc: str) -> str:
    return derive_canonical(strip_processor_prefix(desc))


class Embed:
    def __init__(self, vocab):
        from tranx.routes.embedding import _default_encoder
        self.enc, self.vocab = _default_encoder, vocab
        self.vecs = self.enc(vocab)

    def top(self, descs):
        q = self.enc([query(d) for d in descs])
        sims = q @ self.vecs.T
        idx = sims.argmax(1)
        return [self.vocab[i] for i in idx], sims.max(1)


def fuzzy_top(desc, vocab, k):
    keys = {clean_description(v): v for v in vocab}
    hits = process.extract(clean_description(query(desc)), list(keys), scorer=fuzz.token_set_ratio, limit=k)
    return [keys[h[0]] for h in hits]


def run_slm(descs):
    return [parse_response(_ollama_chat(build_prompt(strip_processor_prefix(d), config.CATEGORIES),
                                        config.SLM_MODEL)).get("canonical_merchant") or query(d)
            for d in descs]


def run_jev(descs, vocab):
    def one(d):
        cands = fuzzy_top(d, vocab, 20)
        crit = {c: None for c in cands}
        crit[NONE_OPTION] = "None of the listed merchants is the one in the description"
        payload = {"state": {"description": d}, "model": JEV_MODEL, "questions": {"merchant": {
            "type": "choice",
            "instructions": "Which known merchant does this bank transaction description refer to?",
            "criteria": crit}}}
        pick = _jev_call(payload).get("answers", {}).get("merchant", {}).get("choice")
        return pick if pick in cands else None  # None = model said "none of these"
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, descs))


ALIASES: dict[str, set[str]] = {}


def accepted(g: str) -> set[str]:
    return {norm(g)} | ALIASES.get(g, set())


def score(pred, gold, n):
    ok = np.array([norm(p or "") in accepted(g) for p, g in zip(pred, gold)])
    per_m = {}
    for o, g in zip(ok, gold):
        per_m.setdefault(g, []).append(o)
    return {"distinct": round(float(ok.mean()), 3),
            "row_weighted": round(float((ok * n).sum() / n.sum()), 3),
            "macro_by_merchant": round(float(np.mean([np.mean(v) for v in per_m.values()])), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/real/moneydata_labels.csv",
                    help="audited labels; moneydata_labels_draft.csv is the pre-audit draft")
    ap.add_argument("--methods", default="derive,fuzzy,embedding,slm,jev")
    ap.add_argument("--confidence", default="high,medium",
                    help="label confidence levels to score; 'low' labels are mostly cleaned literal "
                         "text, which derive/SLM reproduce by construction")
    ap.add_argument("--aliases", default="data/real/moneydata_aliases.csv",
                    help="CSV canonical_merchant,aliases (pipe-separated); a prediction matching "
                         "any alias counts as correct")
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]  # empty = rescore saved predictions only
    if args.aliases:
        for r in pl.read_csv(args.aliases).iter_rows(named=True):
            ALIASES[r["canonical_merchant"]] = {norm(a) for a in (r["aliases"] or "").split("|") if a.strip()}

    all_lab = pl.read_csv(args.labels).filter(pl.col("canonical_merchant") != NOT_MERCHANT)
    lab = all_lab
    if "confidence" in lab.columns:
        lab = lab.filter(pl.col("confidence").is_in(args.confidence.split(",")))
    descs, gold = lab["description"].to_list(), lab["canonical_merchant"].to_list()
    n = lab["n"].to_numpy()
    literal = lab["literal"].to_numpy() if "literal" in lab.columns else np.zeros(len(lab), bool)
    vocab_all = sorted(set(gold))
    # Realistic list: a merchant is "known" if it appears in >= 2 labelled descriptors
    # (any confidence); one-off merchants are the new ones a real list would lack.
    seen = all_lab.group_by("canonical_merchant").len()
    listed = set(seen.filter(pl.col("len") >= 2)["canonical_merchant"])
    held = {m for m in vocab_all if m not in listed}
    vocab_hold = [v for v in vocab_all if v not in held]
    is_held = np.array([g in held for g in gold])
    print(f"{len(descs)} descriptors, {int(n.sum())} rows, {len(vocab_all)} merchants, "
          f"{len(held)} new (not on the realistic list), {int(literal.sum())} literal labels")

    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / f"moneydata_preds_{args.confidence.replace(',', '-')}.parquet"
    preds = lab.select("description", "n", "canonical_merchant")
    if pred_path.exists():  # reuse cached columns, aligned by description (not by position)
        cached = pl.read_parquet(pred_path).drop("n", "canonical_merchant", strict=False)
        preds = preds.join(cached, on="description", how="left")  # left join keeps left order

    def put(name, vals):
        nonlocal preds
        preds = preds.drop(name, strict=False).with_columns(pl.Series(name, vals))

    if "derive" in methods:
        put("derive", [query(d) for d in descs])
    if "fuzzy" in methods:
        put("fuzzy_known", [fuzzy_top(d, vocab_all, 1)[0] for d in descs])
        put("fuzzy_hold", [fuzzy_top(d, vocab_hold, 1)[0] for d in descs])
    if "embedding" in methods:
        for tag, vocab in (("known", vocab_all), ("hold", vocab_hold)):
            top, sim = Embed(vocab).top(descs)
            put(f"embed_{tag}", top)
            put(f"embed_{tag}_cos", sim.astype(float).tolist())
    if "slm" in methods:
        put("slm", run_slm(descs))
    if "jev" in methods:
        put("jev_known", [p or "" for p in run_jev(descs, vocab_all)])
        put("jev_hold", [p or "" for p in run_jev(descs, vocab_hold)])
    preds.write_parquet(pred_path)

    res = {}
    for col in [c for c in preds.columns if c not in ("description", "n", "canonical_merchant") and not c.endswith("_cos")]:
        res[col] = score(preds[col].to_list(), gold, n)
    def held_score(pred):
        return score([x for x, h in zip(pred, is_held) if h], [x for x, h in zip(gold, is_held) if h], n[is_held])

    if {"embed_hold", "slm"} <= set(preds.columns):
        cos = preds["embed_hold_cos"].to_numpy()
        for gate in (0.6, 0.7, 0.8, 0.9):
            accept = cos >= gate
            casc = [e if a else s for e, s, a in zip(preds["embed_hold"], preds["slm"], accept)]
            res[f"cascade_embed{gate}_to_slm"] = {**score(casc, gold, n), "held_subset": held_score(casc),
                                                 "escalated": round(float(1 - accept.mean()), 3),
                                                 "held_wrongly_accepted": round(float(accept[is_held].mean()), 3)}
    if {"jev_hold", "slm"} <= set(preds.columns):
        jev_casc = [j if j else s for j, s in zip(preds["jev_hold"], preds["slm"])]
        res["cascade_jevnone_to_slm"] = {**score(jev_casc, gold, n), "held_subset": held_score(jev_casc),
                                         "escalated": round(float(np.mean([j == "" for j in preds["jev_hold"]])), 3)}
    if "jev_hold" in preds.columns:
        said_none = np.array([p == "" for p in preds["jev_hold"]])
        res["jev_hold_none_rate"] = {"on_held_merchants": round(float(said_none[is_held].mean()), 3),
                                     "on_known_merchants": round(float(said_none[~is_held].mean()), 3)}
    # held-out subset only: who can name a merchant that is not on the list?
    sub = {}
    for col in ("derive", "slm", "embed_hold", "fuzzy_hold"):
        if col in preds.columns:
            p = [x for x, h in zip(preds[col].to_list(), is_held) if h]
            g = [x for x, h in zip(gold, is_held) if h]
            sub[col] = score(p, g, n[is_held])
    res["held_subset_only"] = sub
    nl = ~literal
    res["non_literal"] = {c: score([x for x, k in zip(preds[c].to_list(), nl) if k],
                                   [x for x, k in zip(gold, nl) if k], n[nl])["distinct"]
                          for c in res if c in preds.columns}
    res["non_literal"]["cascade_jevnone_to_slm"] = (
        score([x for x, k in zip(jev_casc, nl) if k], [x for x, k in zip(gold, nl) if k], n[nl])["distinct"]
        if {"jev_hold", "slm"} <= set(preds.columns) else None)
    print(json.dumps(res, indent=2))
    tag = args.confidence.replace(",", "-") + ("_aliased" if args.aliases else "_strict")
    (OUT / f"moneydata_summary_{tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
