import json
import polars as pl
import pytest
from tranx.eval import manifest as mf


def _manifest():
    m = mf.build_manifest(
        feed_sha256="f" * 64, gold_sha256="g" * 64, seed=42,
        splits={"random": {"model": ["a", "b", "c"], "ideal_cache": ["a", "b", "c", "d"]}},
    )
    return m


def test_manifest_hash_excludes_itself_and_is_deterministic():
    m = _manifest()
    assert m["manifest_hash"] == mf.manifest_hash(m)
    # Mutating an unrelated field changes the hash.
    m2 = dict(m)
    m2["seed"] = 43
    assert mf.manifest_hash(m2) != m["manifest_hash"]


def test_write_manifest_roundtrip(tmp_path):
    m = _manifest()
    path = mf.write_manifest(m, tmp_path / "manifest.json")
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == m


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello world")
    assert mf.file_sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def _preds_df(ids):
    return pl.DataFrame({"txn_id": ids, "canonical_merchant": [f"M{i}" for i in ids]})


def test_load_preds_round_trip(tmp_path):
    m = _manifest()
    df = _preds_df(["a", "b", "c"])
    mf.save_preds(df, tmp_path / "random_model_rules.parquet", m["manifest_hash"])
    out = mf.load_preds("random", "model", "rules", m, preds_dir=tmp_path)
    assert set(out["txn_id"].to_list()) == {"a", "b", "c"}
    assert "manifest_hash" in out.columns


def test_load_preds_refuses_wrong_hash(tmp_path):
    m = _manifest()
    df = _preds_df(["a", "b", "c"])
    mf.save_preds(df, tmp_path / "random_model_rules.parquet", "deadbeef")
    with pytest.raises(mf.ManifestError):
        mf.load_preds("random", "model", "rules", m, preds_dir=tmp_path)


def test_load_preds_refuses_duplicate_txn_id(tmp_path):
    m = _manifest()
    df = _preds_df(["a", "b", "b"])  # duplicate row
    mf.save_preds(df, tmp_path / "random_model_rules.parquet", m["manifest_hash"])
    with pytest.raises(mf.ManifestError):
        mf.load_preds("random", "model", "rules", m, preds_dir=tmp_path)


def test_load_preds_refuses_partial_coverage(tmp_path):
    m = _manifest()  # manifest expects {"a", "b", "c"} for random/model
    df = _preds_df(["a", "b"])  # missing "c"
    mf.save_preds(df, tmp_path / "random_model_rules.parquet", m["manifest_hash"])
    with pytest.raises(mf.ManifestError):
        mf.load_preds("random", "model", "rules", m, preds_dir=tmp_path)


def test_load_preds_refuses_extra_coverage(tmp_path):
    m = _manifest()  # manifest expects {"a", "b", "c"} for random/model
    df = _preds_df(["a", "b", "c", "z"])  # extra row not in manifest
    mf.save_preds(df, tmp_path / "random_model_rules.parquet", m["manifest_hash"])
    with pytest.raises(mf.ManifestError):
        mf.load_preds("random", "model", "rules", m, preds_dir=tmp_path)
