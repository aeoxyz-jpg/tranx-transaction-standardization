"""Eval run manifest: pins the exact feed/gold/splits/routes a set of saved
predictions was scored against, so a later script can refuse a stale parquet."""
import hashlib
import json
from pathlib import Path
import polars as pl
from tranx import config


class ManifestError(Exception):
    """Raised when a saved predictions parquet does not match the manifest."""


def manifest_hash(m: dict) -> str:
    """sha256 of the manifest's JSON (sorted keys), excluding manifest_hash itself."""
    body = {k: v for k, v in m.items() if k != "manifest_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def build_manifest(**parts) -> dict:
    """Assemble the manifest dict (schema:1 + given parts) and stamp its hash."""
    m = {"schema": 1, **parts}
    m["manifest_hash"] = manifest_hash(m)
    return m


def write_manifest(m: dict, path: Path = None) -> Path:
    path = Path(path) if path is not None else config.RUN_DIR / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, sort_keys=True))
    return path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_preds(df: pl.DataFrame, path: Path, mhash: str) -> Path:
    """Stamp predictions with the manifest hash and write them to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.with_columns(pl.lit(mhash).alias("manifest_hash")).write_parquet(path)
    return path


def load_preds(split: str, view: str, route: str, manifest: dict,
               preds_dir: Path = None) -> pl.DataFrame:
    """Load reports/preds/{split}_{view}_{route}.parquet, refusing it if it was
    scored against a different manifest, has duplicate txn_ids, or does not cover
    exactly the txn_ids the manifest recorded for this split/view."""
    preds_dir = Path(preds_dir) if preds_dir is not None else config.PREDS_DIR
    path = preds_dir / f"{split}_{view}_{route}.parquet"
    df = pl.read_parquet(path)

    if manifest_hash(manifest) != manifest.get("manifest_hash"):
        raise ManifestError("manifest was edited after it was written (hash does not match)")
    hashes = df["manifest_hash"].unique().to_list()
    if len(hashes) != 1 or hashes[0] != manifest["manifest_hash"]:
        raise ManifestError(f"{path}: manifest_hash mismatch")

    if df["txn_id"].n_unique() != len(df):
        raise ManifestError(f"{path}: duplicate txn_id")

    expected = set(manifest["splits"][split][view])
    got = set(df["txn_id"].to_list())
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise ManifestError(
            f"{path}: coverage mismatch (missing {len(missing)}, extra {len(extra)})")

    return df
