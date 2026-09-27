import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from eval_moneydata import spend_by_descriptor, score
from cache_sim import running_new_flags


RAW_COLUMNS = "Transaction Date,Transaction Type,Transaction Description,Debit Amount,Credit Amount,Balance"


def _write_raw(tmp_path, rows: list[str]) -> Path:
    p = tmp_path / "raw.csv"
    p.write_text(RAW_COLUMNS + "\n" + "\n".join(rows) + "\n")
    return p


def test_spend_by_descriptor_aggregates_per_description(tmp_path):
    raw = _write_raw(tmp_path, [
        "01/01/2020,DEB,SHOP A,10.00,,100",
        "02/01/2020,DEB,SHOP A,5.00,,95",
        "03/01/2020,DD,SHOP B,20.00,,75",
    ])
    out = spend_by_descriptor(raw).sort("description")
    assert out["description"].to_list() == ["SHOP A", "SHOP B"]
    assert out.filter(pl.col("description") == "SHOP A")["spend"].item() == 15.0
    assert out.filter(pl.col("description") == "SHOP A")["n_rows"].item() == 2
    assert out.filter(pl.col("description") == "SHOP B")["spend"].item() == 20.0


def test_spend_by_descriptor_null_amounts_count_as_zero_and_are_tallied(tmp_path):
    raw = _write_raw(tmp_path, [
        "01/01/2020,DEB,SHOP A,10.00,,100",
        "02/01/2020,DEB,SHOP A,,3.00,103",  # refund row: null Debit Amount
    ])
    out = spend_by_descriptor(raw)
    row = out.filter(pl.col("description") == "SHOP A")
    assert row["spend"].item() == 10.0  # null counted as 0, not the credit amount
    assert row["n_rows"].item() == 2
    assert row["null_amounts"].item() == 1


def test_spend_by_descriptor_filters_to_deb_and_dd(tmp_path):
    raw = _write_raw(tmp_path, [
        "01/01/2020,DEB,SHOP A,10.00,,100",
        "02/01/2020,BP,SAVE THE CHANGE,3.00,,97",
        "03/01/2020,SO,STANDING ORDER,50.00,,47",
    ])
    out = spend_by_descriptor(raw)
    assert set(out["description"].to_list()) == {"SHOP A"}


def test_score_spend_weighted_arithmetic():
    pred = ["Amazon", "Amazon", "Lidl"]
    gold = ["Amazon", "Amazon", "Lidl"]
    n = np.array([1, 1, 1])
    spend = [10.0, 5.0, 100.0]
    res = score(pred, gold, n, spend=spend)
    assert res["spend_weighted"] == 1.0  # all correct
    # one wrong prediction, weighted by spend
    pred_wrong = ["Amazon", "Amazon", "Aldi"]
    res2 = score(pred_wrong, gold, n, spend=spend)
    assert res2["spend_weighted"] == round(15.0 / 115.0, 3)


def test_score_no_spend_key_when_spend_absent():
    res = score(["Amazon"], ["Amazon"], np.array([1]))
    assert "spend_weighted" not in res


def test_running_new_flags_toy_sequence():
    seq = ["A", "B", "A", "C", "B", "B"]
    flags = running_new_flags(seq)
    assert flags == [True, True, False, True, False, False]
