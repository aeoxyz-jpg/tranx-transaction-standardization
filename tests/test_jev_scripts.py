import json
import sys
from pathlib import Path

import polars as pl
import pytest

from tranx.cli import eval_rows
from tranx.eval import manifest as mf
from tranx.eval.manifest import ManifestError
from tranx.routes.jev import NONE_OPTION

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import jev_confidence as jc  # noqa: E402
import jev_threshold_cascade as tc  # noqa: E402

CAPS = {"random": 1000, "unseen": 1000}


def _toy_feed_gold():
    """120 rows over 20 merchants (each repeated, so both splits have train merchants),
    one-off descriptors so the model views are non-empty, and a few merchant-less rows."""
    ids, desc, merch, ttype = [], [], [], []
    for i in range(120):
        ids.append(f"T{i:08d}")
        k = i % 20
        if k < 2:
            desc.append(f"SALARY {k}")
            merch.append(None)
            ttype.append("Salary")
        else:
            desc.append(f"SHOP {k} STORE {i}" if i % 2 else f"SHOP {k}")
            merch.append(f"Shop {k}")
            ttype.append(None)
    feed = pl.DataFrame({"txn_id": ids, "description": desc})
    gold = pl.DataFrame({
        "txn_id": ids, "category": [f"C{i % 3}" for i in range(120)],
        "canonical_merchant": merch, "txn_type": ttype, "direction": ["outgoing"] * 120,
        "origin": [None if m is None else "source" for m in merch],
        "noise_abbrev": [False] * 120, "noise_trunc": [False] * 120,
    }, schema_overrides={"canonical_merchant": pl.Utf8, "txn_type": pl.Utf8, "origin": pl.Utf8})
    return feed, gold


def _jev_answer(i, gold_m):
    """Row pattern by position: right, wrong, none_of_these, repeating."""
    return [(gold_m, 0.8), ("Wrong Shop", 0.4), (NONE_OPTION, 0.9)][i % 3]


@pytest.fixture
def run(tmp_path):
    """Toy data dir, manifest and jev_merchant + slm_fewshot model-view preds."""
    feed, gold = _toy_feed_gold()
    data = tmp_path / "data"
    data.mkdir()
    feed.write_parquet(data / "bank_feed.parquet")
    gold.write_parquet(data / "gold.parquet")
    splits = {s: {v: eval_rows(s, v, feed, gold, caps=CAPS, seed=42).eval_feed["txn_id"].to_list()
                  for v in ("model", "ideal_cache")} for s in ("random", "unseen")}
    m = mf.build_manifest(feed_sha256=mf.file_sha256(data / "bank_feed.parquet"),
                          gold_sha256=mf.file_sha256(data / "gold.parquet"),
                          seed=42, caps=CAPS, splits=splits)
    mpath = tmp_path / "manifest.json"
    mf.write_manifest(m, mpath)
    preds = tmp_path / "preds"
    for s in ("random", "unseen"):
        er = eval_rows(s, "model", feed, gold, caps=CAPS, seed=42)
        g = er.eval_gold.sort("txn_id")
        jrows, srows = [], []
        for i, r in enumerate(g.iter_rows(named=True)):
            choice, conf = _jev_answer(i, r["canonical_merchant"])
            base = {"txn_id": r["txn_id"], "description": f"D{i}", "gold_merchant": r["canonical_merchant"],
                    "gold_category": r["category"], "pred_category": "C0",
                    "candidates": [r["canonical_merchant"] or "x", "Other"]}
            jrows.append({**base, "jev_choice": choice, "jev_confidence": conf, "jev_p_choice": conf,
                          "jev_p_none": 0.1, "jev_category_confidence": 0.5,
                          "pred_merchant": choice if choice != NONE_OPTION else None})
            srows.append({**base, "pred_merchant": "SLM PICK"})
        mf.save_preds(pl.DataFrame(jrows), preds / f"{s}_model_jev_merchant.parquet", m["manifest_hash"])
        mf.save_preds(pl.DataFrame(srows), preds / f"{s}_model_slm_fewshot.parquet", m["manifest_hash"])
    return {"tmp": tmp_path, "data": data, "manifest": m, "mpath": mpath, "preds": preds}


def _expected(split, run):
    rows = [r for r in pl.read_parquet(run["preds"] / f"{split}_model_jev_merchant.parquet")
            .sort("txn_id").iter_rows(named=True) if r["gold_merchant"] is not None]
    right = sum(r["jev_choice"] == r["gold_merchant"] for r in rows)
    none = sum(r["jev_choice"] == NONE_OPTION for r in rows)
    return len(rows), right, none


def test_summary_computed_from_toy_preds(run):
    md_rows = run["tmp"] / "md_rows.parquet"
    pl.DataFrame({"dataset": ["moneydata"] * 2, "setting": ["known"] * 2, "description": ["A", "B"],
                  "gold": ["A", "B"], "choice": ["A", NONE_OPTION], "confidence": [0.9, 0.7],
                  "correct": [True, False], "gold_in_list": [True, True], "n": [3, 1]}
                 ).write_parquet(md_rows)
    out_rows, summary = run["tmp"] / "rows.parquet", run["tmp"] / "summary.json"
    jc.main(["--synthetic-only", "--manifest", str(run["mpath"]), "--preds-dir", str(run["preds"]),
             "--data-dir", str(run["data"]), "--moneydata-rows", str(md_rows),
             "--rows", str(out_rows), "--summary", str(summary)])
    s = json.loads(summary.read_text())
    assert s["manifest_hash"] == run["manifest"]["manifest_hash"]
    for split in ("random", "unseen"):
        n, right, none = _expected(split, run)
        r = s[f"synthetic/{split}"]
        assert r["rows"] == n
        assert r["accepted_right"] == right
        assert r["accepted_wrong"] == n - right - none
        assert r["picked_none"] == round(none / n, 3)
        # right picks at 0.8, wrong at 0.4: perfectly separated
        assert r["auroc_right_vs_wrong"] == 1.0
    # unseen is merchant-disjoint, so no gold merchant is in the train list
    assert s["synthetic/unseen"]["gold_in_list"] == 0.0
    assert s["synthetic/random"]["gold_in_list"] > 0.0
    assert s["moneydata/known"]["rows"] == 2
    assert s["moneydata/known"]["picked_none"] == 0.5
    rows = pl.read_parquet(out_rows)
    assert rows.filter(pl.col("dataset") == "synthetic")["gold_in_cands"].all()


def test_cascade_takes_slm_below_threshold_or_on_none():
    d = pl.DataFrame({"choice": ["Acme", NONE_OPTION, "Acme"], "confidence": [0.9, 0.99, 0.4],
                      "slm": ["Slm", "Beta", "Acme"], "gold": ["Acme", "Beta", "Acme"],
                      "gold_in_list": [True, False, True]})
    by = {r["threshold"]: r for r in tc.sweep(d, lambda p, g, c: p == g)}
    # t=0.5: row 1 accepted (Jev right), row 2 none -> SLM (right), row 3 low -> SLM (right)
    assert by[0.5]["escalated_to_slm"] == round(2 / 3, 3)
    assert by[0.5]["merchant_acc"] == 1.0
    # t=0.95: row 1 now escalated -> SLM "Slm" (wrong)
    assert by[0.95]["escalated_to_slm"] == 1.0
    assert by[0.95]["merchant_acc"] == round(2 / 3, 3)
    assert by["slm_only"]["escalated_to_slm"] == 1.0


def test_cascade_rows_join_jev_and_slm_preds(run):
    rows = tc.synthetic_cascade_rows(run["manifest"], run["preds"], run["data"])
    for split, d in rows.items():
        n, _, _ = _expected(split, run)
        assert len(d) == n
        assert set(d["slm"].to_list()) == {"SLM PICK"}
        res = {r["threshold"]: r for r in tc.sweep(d, lambda p, g, c: p == g)}
        # none_of_these rows and the 0.4-confidence wrong picks go to the SLM at t=0.5
        assert res[0.5]["escalated_to_slm"] == round(1 - d["correct"].mean(), 3)


def test_wrong_manifest_hash_raises(run):
    stale = {**run["manifest"], "manifest_hash": "0" * 64}
    with pytest.raises(ManifestError):
        jc.synthetic_rows(stale, run["preds"], run["data"])
    with pytest.raises(ManifestError):
        tc.synthetic_cascade_rows(stale, run["preds"], run["data"])


def test_changed_feed_raises(run):
    feed = pl.read_parquet(run["data"] / "bank_feed.parquet")
    feed.head(100).write_parquet(run["data"] / "bank_feed.parquet")
    with pytest.raises(ManifestError, match="sha256"):
        jc.synthetic_rows(run["manifest"], run["preds"], run["data"])


def test_synthetic_only_without_moneydata_rows_fails_clearly(run):
    missing = run["tmp"] / "nope.parquet"
    common = ["--synthetic-only", "--manifest", str(run["mpath"]), "--preds-dir", str(run["preds"]),
              "--data-dir", str(run["data"]), "--moneydata-rows", str(missing)]
    with pytest.raises(SystemExit, match="extract-moneydata"):
        jc.main(common + ["--rows", str(run["tmp"] / "r.parquet"),
                          "--summary", str(run["tmp"] / "s.json")])
    with pytest.raises(SystemExit, match="extract-moneydata"):
        tc.main(common + ["--out", str(run["tmp"] / "c.json")])
    assert not (run["tmp"] / "s.json").exists() and not (run["tmp"] / "c.json").exists()


def test_extract_moneydata_keeps_only_moneydata_rows(tmp_path):
    src, dst = tmp_path / "rows.parquet", tmp_path / "md.parquet"
    pl.DataFrame({"dataset": ["synthetic", "moneydata", "moneydata"], "setting": ["random", "known", "hold"],
                  "n": [None, 2, 2]}).write_parquet(src)
    jc.main(["--extract-moneydata", "--rows", str(src), "--moneydata-rows", str(dst)])
    out = pl.read_parquet(dst)
    assert out["dataset"].to_list() == ["moneydata", "moneydata"]
    assert pl.read_parquet(src).height == 3
