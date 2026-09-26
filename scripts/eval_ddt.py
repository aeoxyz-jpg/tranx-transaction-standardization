"""Category-only eval on DoDataThings/us-bank-transaction-categories-v2.

An independently generated synthetic feed (different noise vocabulary than
Tranx's own synth/cleaner), used to check how much of Tranx's scores depend on
the cleaner mirroring the generator. Writes per-row predictions so methods can
be compared with a paired test.

Usage: python scripts/eval_ddt.py --methods embedding,slm,jev [--n 1000]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

import polars as pl
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tranx import config
from tranx.routes.slm_fewshot import _ollama_chat, parse_response
from tranx.routes.jev import _jev_call, JEV_MODEL

SRC = config.DATA_DIR / "real" / "ddt" / "transactions-synthetic.csv"
OUT = config.REPORTS_DIR / "real"


def load(n: int, train_cap: int):
    d = pl.read_csv(SRC).with_row_index("rid")
    test = d.sample(n=n, seed=config.SEED)
    train = d.filter(~pl.col("rid").is_in(test["rid"]))
    if len(train) > train_cap:
        train = train.sample(n=train_cap, seed=config.SEED)
    return train, test


def run_embedding(train, test, cats):
    from tranx.routes.embedding import _default_encoder
    clf = LogisticRegression(max_iter=1000)
    clf.fit(_default_encoder(train["description"].to_list()), train["category"].to_list())
    return list(clf.predict(_default_encoder(test["description"].to_list())))


def _slm_prompt(description: str, cats: list[str]) -> str:
    # Same instructions as the Tranx slm_fewshot prompt, minus the few-shot
    # examples (they use Tranx's category names, which do not exist here).
    return (
        "You categorize bank transactions. Ignore payment-processor prefixes and "
        "reference numbers and choose exactly one category.\n"
        f"Allowed categories: {', '.join(cats)}\n"
        f'Description: "{description}"\n'
        'Reply with ONLY a JSON object with key "category".'
    )


def run_slm(train, test, cats):
    out = []
    for desc in test["description"].to_list():
        out.append(parse_response(_ollama_chat(_slm_prompt(desc, cats), config.SLM_MODEL)).get("category"))
    return out


def run_jev(train, test, cats):
    def one(desc):
        payload = {
            "state": {"description": desc},
            "model": JEV_MODEL,
            "questions": {"category": {
                "type": "choice",
                "instructions": "Which spending category does this bank transaction belong to?",
                "criteria": {c: None for c in cats},
            }},
        }
        return _jev_call(payload).get("answers", {}).get("category", {}).get("choice")
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, test["description"].to_list()))


METHODS = {"embedding": run_embedding, "slm": run_slm, "jev": run_jev}


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="embedding,slm,jev")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--train-cap", type=int, default=20000)
    args = ap.parse_args()

    train, test = load(args.n, args.train_cap)
    cats = sorted(train["category"].unique().to_list())
    gold = test["category"].to_list()
    OUT.mkdir(parents=True, exist_ok=True)
    preds_path = OUT / "ddt_preds.parquet"
    preds = pl.read_parquet(preds_path) if preds_path.exists() else test.select("rid", "description", "category")

    for m in args.methods.split(","):
        p = METHODS[m](train, test, cats)
        preds = preds.drop(m, strict=False).with_columns(pl.Series(m, [x if x is not None else "" for x in p]))
    preds.write_parquet(preds_path)

    summary = {}
    ok = {}
    for m in [c for c in preds.columns if c in METHODS]:
        p = preds[m].to_list()
        ok[m] = [x == g for x, g in zip(p, gold)]
        acc = sum(ok[m]) / len(gold)
        summary[m] = {
            "acc": round(acc, 3),
            "ci95": round(1.96 * sqrt(acc * (1 - acc) / len(gold)), 3),
            "macro_f1": round(float(f1_score(gold, p, average="macro")), 3),
            "invalid": sum(x not in cats for x in p),
        }
        print(m, summary[m])
    names = list(ok)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b, pv = mcnemar(ok[names[i]], ok[names[j]])
            print(f"McNemar {names[i]} vs {names[j]}: only-{names[i]} right={a}, only-{names[j]} right={b}, p={pv:.4f}")
    (OUT / "ddt_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
