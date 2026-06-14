import numpy as np
import polars as pl
from tranx.routes.embedding import EmbeddingRoute
from tranx.schema import Txn


def _fake_encoder(texts):
    # Deterministic bag-of-chars embedding: vector of 26 letter counts.
    vecs = []
    for t in texts:
        v = np.zeros(26)
        for ch in t.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1.0
        vecs.append(v)
    return np.array(vecs)


def _train():
    feed = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "description": ["McDonald's #111", "BP on Buford Hwy", "Wage"],
        "mcc": [5814, None, None],
        "amount": [-10.0, -40.0, 2000.0],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "category": ["Food & Dining", "Transportation", "Income"],
        "canonical_merchant": ["McDonald's", "BP", "Wage"],
        "direction": ["outgoing", "outgoing", "incoming"],
    })
    return feed, gold


def _txn(desc, amount=-10.0):
    return Txn("Tx", "C1", desc, "O-CC-M", None, amount, "credit_card",
               "2026-01-01", "USA", "USD")


def test_embedding_route_matches_nearest_merchant():
    r = EmbeddingRoute(encoder=_fake_encoder)
    r.fit(*_train())
    out = r.standardize(_txn("McDonald's #888"))
    assert out.canonical_merchant == "McDonald's"


def test_embedding_route_predicts_category():
    r = EmbeddingRoute(encoder=_fake_encoder)
    r.fit(*_train())
    out = r.standardize(_txn("BP @ Pleasant Hills", amount=-30.0))
    assert out.category in {"Transportation", "Food & Dining", "Income"}


def test_embedding_route_direction_from_amount():
    r = EmbeddingRoute(encoder=_fake_encoder)
    r.fit(*_train())
    assert r.standardize(_txn("Wage", amount=2000.0)).direction == "incoming"
