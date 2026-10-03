import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import inhouse_cascade as ic  # noqa: E402


def test_synthetic_summary_takes_rules_only_when_matched():
    d = pl.DataFrame({
        "gold_merchant": ["Lidl", "Aldi", "Costa"],
        "gold_category": ["Shopping & Retail"] * 2 + ["Food & Dining"],
        "matched": [True, False, False],
        "rules": ["Lidl", "ALDI 12", "COSTA X"], "rules_cat": ["Shopping & Retail"] * 3,
        "slm": ["Tesco", "Aldi", "Costa"], "slm_cat": ["Shopping & Retail"] * 2 + ["Food & Dining"],
        "jev_slm": ["Lidl", "Aldi", "Nero"], "jev_slm_cat": ["Shopping & Retail"] * 2 + ["Food & Dining"],
    }).with_columns(
        pl.when(pl.col("matched")).then(pl.col("rules")).otherwise(pl.col("slm")).alias("cascade"),
        pl.when(pl.col("matched")).then(pl.col("rules_cat")).otherwise(pl.col("slm_cat")).alias("cascade_cat"))
    s = ic.synthetic_summary(d)
    assert s["accuracy"] == {"rules": round(1 / 3, 3), "slm": round(2 / 3, 3),
                             "cascade": 1.0, "jev_slm": round(2 / 3, 3)}
    assert s["fuzzy_matched_share"] == round(1 / 3, 3)


def test_moneydata_summary_gates_fuzzy_and_falls_back_to_slm(tmp_path):
    preds = tmp_path / "md.parquet"
    pl.DataFrame({
        "description": ["LIDL GB NOTTINGHAM", "ZZQ BAKERY LEEDS"],
        "n": [3, 1],
        "canonical_merchant": ["Lidl", "Zzq Bakery"],
        "fuzzy_hold": ["Lidl", "Lidl"],          # second is a low-score top-1 hit
        "embed_hold": ["Lidl", "Lidl"], "embed_hold_cos": [0.9, 0.2],
        "slm": ["Lidl", "Zzq Bakery"],
        "jev_hold": ["Lidl", ""],
    }).write_parquet(preds)
    s = ic.moneydata_summary(preds, tmp_path / "no_aliases.csv")
    assert s["sent_to_slm"] == {"fuzzy_to_slm": 0.5, "embedding_to_slm": 0.5, "jev_to_slm": 0.5}
    assert s["per_descriptor"]["fuzzy_to_slm"] == 1.0
    assert s["per_row"]["slm"] == 1.0
