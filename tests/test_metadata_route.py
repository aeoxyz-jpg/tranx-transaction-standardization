import polars as pl
from tranx.routes.metadata import MetadataRoute
from tranx.schema import Txn


def _txn(desc, type_code="O-CC-M", mcc=5814, amount=-10.0):
    return Txn("Tx", "C1", desc, type_code, mcc, amount, "credit_card",
               "2026-01-01", "USA", "USD")


def _train():
    feed = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3", "T4"],
        "transaction_type_code": ["O-CC-M", "O-CC-M", "I-ACH-M", "O-CC-M"],
        "mcc": [5814, 5814, None, 4111],
        "amount": [-10.0, -12.0, 2000.0, -30.0],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3", "T4"],
        "category": ["Food & Dining", "Food & Dining", "Income", "Transportation"],
    })
    return feed, gold


def test_merchant_is_always_none():
    r = MetadataRoute()
    r.fit(*_train())
    out = r.standardize(_txn("McDonald's #111"))
    assert out.canonical_merchant is None


def test_direction_from_amount_sign():
    r = MetadataRoute()
    r.fit(*_train())
    assert r.standardize(_txn("Wage", type_code="I-ACH-M", mcc=None, amount=2000.0)).direction == "incoming"
    assert r.standardize(_txn("BP #5", amount=-30.0)).direction == "outgoing"


def test_learns_type_and_mcc_category_prior():
    r = MetadataRoute()
    r.fit(*_train())
    out = r.standardize(_txn("Some Diner", mcc=5814, amount=-11.0))
    assert out.category == "Food & Dining"


def test_never_reads_description():
    # Identical metadata, wildly different descriptions -> identical prediction.
    r = MetadataRoute()
    r.fit(*_train())
    a = r.standardize(_txn("McDonald's #111", mcc=5814, amount=-11.0))
    b = r.standardize(_txn("Zzyzx Widget Emporium 999", mcc=5814, amount=-11.0))
    assert a.category == b.category
    assert a.canonical_merchant == b.canonical_merchant is None


def test_unseen_mcc_and_type_code_do_not_error():
    r = MetadataRoute()
    r.fit(*_train())
    out = r.standardize(_txn("Something New", type_code="X-NEW-M", mcc=9999, amount=-5.0))
    assert out.category in ["Food & Dining", "Income", "Transportation"]
