"""Record Jev's merchant-choice confidence per row and test whether a confidence
threshold separates right from wrong picks.

Datasets:
  synthetic hard feed, random + unseen model views: read from this run's saved
  jev_merchant predictions (reports/preds, checked against reports/run/manifest.json);
  MoneyData real descriptors, known list + 20%-held-out list (Jev API calls, or with
  --synthetic-only the saved rows in reports/real/jev_confidence_moneydata_rows.parquet).

Writes reports/real/jev_confidence_rows.parquet (per row, gitignored) and
reports/jev_confidence_summary.json.
Usage: python scripts/jev_confidence.py [--synthetic-only] [--extract-moneydata]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from tranx import config
from tranx.cli import eval_rows
from tranx.eval.manifest import ManifestError, file_sha256, load_preds
from tranx.eval.metrics import _norm_merchant as norm, merchant_ok
from tranx.routes.jev import _jev_call, JEV_MODEL, NONE_OPTION, parse
import eval_moneydata as md

THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
REAL = config.REPORTS_DIR / "real"
MONEYDATA_ROWS = REAL / "jev_confidence_moneydata_rows.parquet"


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def train_vocab(split: str, manifest: dict, data_dir: Path) -> set:
    """Normalised train merchants of `split`, rebuilt from the feed/gold the
    manifest was built on (refused if their sha256 differs)."""
    feed_path, gold_path = Path(data_dir) / "bank_feed.parquet", Path(data_dir) / "gold.parquet"
    if (file_sha256(feed_path) != manifest["feed_sha256"]
            or file_sha256(gold_path) != manifest["gold_sha256"]):
        raise ManifestError(f"{data_dir}: feed/gold sha256 differs from the manifest")
    er = eval_rows(split, "model", pl.read_parquet(feed_path), pl.read_parquet(gold_path),
                   caps=manifest["caps"], seed=manifest["seed"])
    return {norm(v) for v in er.train_gold["canonical_merchant"].drop_nulls().unique()}


def synthetic_rows(manifest: dict, preds_dir: Path | None, data_dir: Path) -> pl.DataFrame:
    """Per-row Jev merchant answers on the model view of both splits, from the
    saved jev_merchant predictions (no Jev calls)."""
    out = []
    for split in ("random", "unseen"):
        p = load_preds(split, "model", "jev_merchant", manifest, preds_dir)
        # merchant question only makes sense where the gold has a merchant
        p = p.filter(pl.col("gold_merchant").is_not_null()).sort("txn_id")
        vocab = train_vocab(split, manifest, data_dir)
        for r in p.iter_rows(named=True):
            choice, gold_m = r["jev_choice"], r["gold_merchant"]
            out.append({"dataset": "synthetic", "setting": split, "txn_id": r["txn_id"],
                        "description": r["description"], "gold": gold_m,
                        "choice": choice, "confidence": r["jev_confidence"],
                        "p_choice": r["jev_p_choice"], "p_none": r["jev_p_none"],
                        "correct": choice != NONE_OPTION and merchant_ok(
                            choice, gold_m, r["pred_category"] == r["gold_category"]),
                        "gold_in_list": norm(gold_m) in vocab,
                        "gold_in_cands": norm(gold_m) in {norm(x) for x in (r["candidates"] or [])},
                        "cat_choice": r["pred_category"],
                        "cat_confidence": r["jev_category_confidence"],
                        "cat_correct": r["pred_category"] is not None
                                       and r["pred_category"] == r["gold_category"]})
        print(f"synthetic {split}: {len(p)} rows", flush=True)
    return pl.DataFrame(out, infer_schema_length=None)


def saved_moneydata_rows(path: Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"--synthetic-only needs the saved MoneyData Jev rows at {path}; "
                         "create them with --extract-moneydata before deleting "
                         "reports/real/jev_confidence_rows.parquet")
    return pl.read_parquet(path)


def extract_moneydata(rows_path: Path, out_path: Path) -> Path:
    """Copy the MoneyData rows of an existing per-row parquet to `out_path`."""
    rows = pl.read_parquet(rows_path).filter(pl.col("dataset") == "moneydata")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(out_path)
    print(f"{len(rows)} moneydata rows -> {out_path}")
    return Path(out_path)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=config.RUN_DIR / "manifest.json")
    ap.add_argument("--preds-dir", type=Path, default=config.PREDS_DIR)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--synthetic-only", action="store_true",
                    help="no MoneyData API calls: reuse the saved MoneyData rows")
    ap.add_argument("--extract-moneydata", action="store_true",
                    help="only copy the MoneyData rows of --rows to --moneydata-rows")
    ap.add_argument("--rows", type=Path, default=REAL / "jev_confidence_rows.parquet")
    ap.add_argument("--moneydata-rows", type=Path, default=MONEYDATA_ROWS)
    ap.add_argument("--summary", type=Path, default=config.REPORTS_DIR / "jev_confidence_summary.json")
    args = ap.parse_args(argv)

    if args.extract_moneydata:
        extract_moneydata(args.rows, args.moneydata_rows)
        return
    # Check the MoneyData source before the (slower) synthetic part.
    mrows = saved_moneydata_rows(args.moneydata_rows) if args.synthetic_only else None
    manifest = load_manifest(args.manifest)
    srows = synthetic_rows(manifest, args.preds_dir, args.data_dir)
    if mrows is None:
        mrows = pl.DataFrame(moneydata_rows(), infer_schema_length=None)
    rows = pl.concat([srows, mrows], how="diagonal_relaxed")
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(args.rows)
    summary = {**summarize(rows), "manifest_hash": manifest["manifest_hash"]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
