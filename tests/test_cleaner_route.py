import polars as pl
from tranx.routes.cleaner import CleanerRoute
from tranx.schema import Txn


def _txn(desc, amount=-10.0):
    return Txn("Tx", "C1", desc, "O-CC-M", None, amount, "credit_card",
               "2026-01-01", "USA", "USD")


def test_fit_is_a_noop():
    r = CleanerRoute()
    r.fit(pl.DataFrame({"txn_id": []}), pl.DataFrame({"txn_id": []}))
    assert r.name == "cleaner"


def test_merchant_is_derived_canonical_over_stripped_description():
    r = CleanerRoute()
    out = r.standardize(_txn("SQ *BLUE BOTTLE #123"))
    assert out.canonical_merchant == "BLUE BOTTLE"


def test_category_is_none():
    r = CleanerRoute()
    out = r.standardize(_txn("McDonald's #111"))
    assert out.category is None


def test_direction_from_amount_sign():
    r = CleanerRoute()
    assert r.standardize(_txn("Wage", amount=2000.0)).direction == "incoming"
    assert r.standardize(_txn("McDonald's #5", amount=-10.0)).direction == "outgoing"
