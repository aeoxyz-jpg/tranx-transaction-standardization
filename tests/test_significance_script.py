import json
import sys
from pathlib import Path

import polars as pl
import pytest

from tranx.eval import manifest as mf
from tranx.eval.manifest import ManifestError

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import significance as sig  # noqa: E402


def _manifest(ids_by_split):
    splits = {split: {"model": ids, "ideal_cache": ids} for split, ids in ids_by_split.items()}
    return mf.build_manifest(feed_sha256="f" * 64, gold_sha256="g" * 64, seed=42, splits=splits)


def _write_preds(preds_dir, split, route, mhash, rows):
    mf.save_preds(pl.DataFrame(rows), preds_dir / f"{split}_model_{route}.parquet", mhash)


# t3 is a local, abbreviated row: excluded from merchant scoring by D2.
BASE_ROWS = [
    {"txn_id": "t1", "pred_merchant": "Lidl", "gold_merchant": "Lidl", "gold_category": "Shopping & Retail",
     "gold_txn_type": "DEB", "origin": "source", "noise_abbrev": False, "pred_category": "Shopping & Retail"},
    {"txn_id": "t2", "pred_merchant": "Tesco", "gold_merchant": "Aldi", "gold_category": "Shopping & Retail",
     "gold_txn_type": "DEB", "origin": "source", "noise_abbrev": False, "pred_category": "Shopping & Retail"},
    {"txn_id": "t3", "pred_merchant": "Costa", "gold_merchant": "Costa", "gold_category": "Food & Dining",
     "gold_txn_type": "DEB", "origin": "local", "noise_abbrev": True, "pred_category": "Food & Dining"},
]


def test_synthetic_merchant_primary_entry_computed(tmp_path):
    ids = [r["txn_id"] for r in BASE_ROWS]
    m = _manifest({"unseen": ids})
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    rows_a = [{**r, "pred_merchant": r["gold_merchant"]} for r in BASE_ROWS]      # jev_slm: all correct
    rows_b = [{**r, "pred_merchant": "WRONG"} for r in BASE_ROWS]                # slm_fewshot: all wrong
    _write_preds(preds_dir, "unseen", "jev_slm", m["manifest_hash"], rows_a)
    _write_preds(preds_dir, "unseen", "slm_fewshot", m["manifest_hash"], rows_b)

    res = sig.synthetic_merchant(m, preds_dir, "unseen", "jev_slm", "slm_fewshot", "primary")
    assert "skipped" not in res
    assert res["label"] == "primary"
    assert res["n_rows"] == 2       # t3 (local + abbreviated) excluded
    assert res["n_clusters"] == 2   # Lidl, Aldi
    assert res["diff"] == 1.0       # jev_slm right, slm_fewshot wrong, on the two scored rows


def test_local_abbreviated_row_excluded_regardless_of_correctness(tmp_path):
    ids = [r["txn_id"] for r in BASE_ROWS]
    m = _manifest({"unseen": ids})
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    # jev_slm gets t3 wrong; if t3 were scored this would pull the diff down.
    rows_a = [{**r, "pred_merchant": r["gold_merchant"] if r["txn_id"] != "t3" else "WRONG"} for r in BASE_ROWS]
    rows_b = [{**r, "pred_merchant": r["gold_merchant"]} for r in BASE_ROWS]
    _write_preds(preds_dir, "unseen", "jev_slm", m["manifest_hash"], rows_a)
    _write_preds(preds_dir, "unseen", "slm_fewshot", m["manifest_hash"], rows_b)
    res = sig.synthetic_merchant(m, preds_dir, "unseen", "jev_slm", "slm_fewshot", "primary")
    assert res["n_rows"] == 2
    assert res["n_clusters"] == 2


def test_missing_preds_file_is_skipped_not_crashed(tmp_path):
    ids = [r["txn_id"] for r in BASE_ROWS]
    m = _manifest({"unseen": ids})
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    _write_preds(preds_dir, "unseen", "jev_slm", m["manifest_hash"], BASE_ROWS)  # slm_fewshot missing
    res = sig.synthetic_merchant(m, preds_dir, "unseen", "jev_slm", "slm_fewshot", "primary")
    assert res["skipped"]
    assert res["label"] == "primary"


def test_wrong_manifest_hash_raises_manifest_error(tmp_path):
    ids = [r["txn_id"] for r in BASE_ROWS]
    m = _manifest({"unseen": ids})
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    _write_preds(preds_dir, "unseen", "jev_slm", m["manifest_hash"], BASE_ROWS)
    _write_preds(preds_dir, "unseen", "slm_fewshot", "deadbeef", BASE_ROWS)  # scored against a different run
    with pytest.raises(ManifestError):
        sig.synthetic_merchant(m, preds_dir, "unseen", "jev_slm", "slm_fewshot", "primary")


def test_synthetic_category_all_rows_scored_clustered_by_merchant_or_txn_type(tmp_path):
    rows = BASE_ROWS + [{"txn_id": "t4", "pred_merchant": None, "gold_merchant": None,
                        "gold_category": "Income", "gold_txn_type": "FPI", "origin": None,
                        "noise_abbrev": False, "pred_category": "Income"}]
    ids = [r["txn_id"] for r in rows]
    m = _manifest({"random": ids})
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    rows_a = [{**r, "pred_category": r["gold_category"]} for r in rows]
    rows_b = [{**r, "pred_category": "WRONG"} for r in rows]
    _write_preds(preds_dir, "random", "embedding", m["manifest_hash"], rows_a)
    _write_preds(preds_dir, "random", "jev_merchant", m["manifest_hash"], rows_b)
    res = sig.synthetic_category(m, preds_dir, "random", "embedding", "jev_merchant", "exploratory")
    assert res["n_rows"] == 4          # merchant-less row t4 IS scored for category
    assert res["n_clusters"] == 4      # Lidl, Aldi, Costa, FPI (t4's txn_type)


def test_load_manifest_missing_raises_clear_error(tmp_path):
    with pytest.raises(SystemExit):
        sig._load_manifest(tmp_path / "does_not_exist.json")


def _moneydata_fixture(tmp_path):
    preds = pl.DataFrame({
        "description": ["D1", "D2", "D3", "D4"],
        "n": [10, 5, 1, 1],
        "canonical_merchant": ["Lidl", "Lidl", "Tesco", "Amazon"],
        "slm": ["Lidl", "WRONG", "Tesco", "Amazon"],
        "jev_hold": ["Lidl", "Lidl", "", "Amazon"],  # "" -> falls back to slm (Tesco, correct)
        "jev_known": ["Lidl", "Lidl", "Tesco", "Amazon"],
        "embed_known": ["Lidl", "WRONG", "Tesco", "Amazon"],
    })
    preds_path = tmp_path / "moneydata_preds.parquet"
    preds.write_parquet(preds_path)
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("description,confidence\nD1,high\nD2,high\nD3,medium\nD4,low\nD5,low\n")
    aliases_path = tmp_path / "aliases.csv"
    aliases_path.write_text("canonical_merchant,aliases\n")
    return preds_path, aliases_path, labels_path


def test_moneydata_section_primary_and_exploratory(tmp_path):
    preds_path, aliases_path, labels_path = _moneydata_fixture(tmp_path)
    primary, exploratory, note = sig.moneydata_section(preds_path, aliases_path, labels_path)
    assert "skipped" not in primary
    assert primary["label"] == "primary"
    assert primary["diff"] == 0.25  # jev_slm right on all 4, slm wrong only on D2 (1/4)
    assert "row_weighted_point_with_amazon" in primary
    assert "row_weighted_ci_without_amazon" in primary
    assert "lomo_range" in primary
    assert "moneydata_known" in exploratory
    assert "2 low-confidence" in note


def test_moneydata_section_missing_preds_is_skipped(tmp_path):
    primary, exploratory, note = sig.moneydata_section(
        tmp_path / "nope.parquet", tmp_path / "aliases.csv", tmp_path / "labels.csv")
    assert primary["skipped"]
    assert exploratory == {}


def test_ddt_section_computed(tmp_path):
    d = pl.DataFrame({
        "category": ["A", "B", "A", "B"],
        "embedding": ["A", "B", "A", "A"],
        "jev": ["A", "A", "A", "B"],
    })
    path = tmp_path / "ddt_preds.parquet"
    d.write_parquet(path)
    res = sig.ddt_section(path)
    assert "skipped" not in res
    assert res["label"] == "exploratory"
    assert res["n_rows"] == 4
    assert res["n_clusters"] == 4  # clustered by row index, not merchant


def test_ddt_section_missing_file_is_skipped(tmp_path):
    res = sig.ddt_section(tmp_path / "nope.parquet")
    assert res["skipped"]


def test_build_report_end_to_end_json_and_md_serializable(tmp_path):
    ids = [r["txn_id"] for r in BASE_ROWS]
    m = _manifest({"unseen": ids, "random": ids})
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    mf.write_manifest(m, manifest_path)
    # Only jev_slm/slm_fewshot on unseen are written; every other synthetic
    # comparison should come back skipped rather than crash the run.
    rows_a = [{**r, "pred_merchant": r["gold_merchant"]} for r in BASE_ROWS]
    rows_b = [{**r, "pred_merchant": "WRONG"} for r in BASE_ROWS]
    _write_preds(preds_dir, "unseen", "jev_slm", m["manifest_hash"], rows_a)
    _write_preds(preds_dir, "unseen", "slm_fewshot", m["manifest_hash"], rows_b)

    preds_path, aliases_path, labels_path = _moneydata_fixture(tmp_path)
    ddt_path = tmp_path / "ddt_preds.parquet"
    pl.DataFrame({"category": ["A", "B"], "embedding": ["A", "B"], "jev": ["A", "A"]}).write_parquet(ddt_path)

    report = sig.build_report(manifest_path, preds_dir, moneydata_preds_path=preds_path,
                              aliases_path=aliases_path, labels_path=labels_path, ddt_path=ddt_path)
    assert "skipped" not in report["primary"]["synthetic_unseen"]
    assert "skipped" not in report["primary"]["moneydata"]
    assert report["exploratory"]["synthetic_random_merchant"]["skipped"]  # no random preds written
    assert "skipped" not in report["exploratory"]["ddt_category"]
    json.dumps(report)  # no leftover numpy types
    sig.render_md(report)  # does not raise
