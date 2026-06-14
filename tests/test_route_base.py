import polars as pl
from tranx.routes.base import Route
from tranx.schema import Txn, Standardized


class _Echo(Route):
    name = "echo"

    def fit(self, train_feed, train_gold):
        self.fitted = True

    def standardize(self, txn: Txn) -> Standardized:
        return Standardized(canonical_merchant=txn.description,
                            category="Income", direction="incoming")


def test_route_requires_name_fit_standardize():
    r = _Echo()
    r.fit(pl.DataFrame(), pl.DataFrame())
    assert r.fitted is True
    out = r.standardize(Txn("t1", "C1", "Wage", "I-ACH-N", None, 100.0,
                            "ach", "2026-01-01", "USA", "USD"))
    assert isinstance(out, Standardized)
    assert out.category == "Income"


def test_route_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        Route()
