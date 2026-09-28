import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from category_routing import routing_rows, summarize  # noqa: E402
from tranx.eval.manifest import build_manifest, save_preds  # noqa: E402


def test_rule_takes_jev_category_only_when_jev_says_none(tmp_path):
    ids = ["T1", "T2", "T3"]
    m = build_manifest(splits={"random": {"model": ids}})
    base = {"txn_id": ids, "gold_category": ["Food", "Food", "Travel"],
            "gold_merchant": ["A", "B", None], "gold_txn_type": [None, None, "Transfer"],
            "origin": ["source", "local", None]}
    save_preds(pl.DataFrame({**base, "pred_category": ["Food", "Shop", "Travel"]}),
               tmp_path / "random_model_embedding.parquet", m["manifest_hash"])
    save_preds(pl.DataFrame({**base, "pred_category": ["Shop", "Food", "Food"],
                             "jev_choice": ["A", "none_of_these", "none_of_these"],
                             "candidates": [["A"], ["A"], ["A"]]}),
               tmp_path / "random_model_jev_merchant.parquet", m["manifest_hash"])
    d = routing_rows("random", m, tmp_path)
    assert d.sort("txn_id")["rule"].to_list() == ["Food", "Food", "Food"]
    s = summarize(d)
    assert s["accuracy"] == {"embedding": round(2 / 3, 3), "jev": round(1 / 3, 3), "rule": round(2 / 3, 3), "rows": 3}
    assert s["by_jev_answer"]["picked"]["rows"] == 1
