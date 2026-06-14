import polars as pl
from tranx.eval.harness import run_route
from tranx.routes.rules import RulesRoute


def _data():
    feed = pl.DataFrame({
        "txn_id": ["T1", "T2"],
        "customer_id": ["C1", "C1"],
        "description": ["McDonald's #111", "BP on Buford Hwy"],
        "transaction_type_code": ["O-CC-M", "O-DC-M"],
        "mcc": [5814, None],
        "amount": [-10.0, -40.0],
        "payment_method": ["credit_card", "debit_card"],
        "posted_date": ["2026-01-01", "2026-01-02"],
        "country": ["USA", "USA"],
        "currency": ["USD", "USD"],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2"],
        "category": ["Food & Dining", "Transportation"],
        "canonical_merchant": ["McDonald's", "BP"],
        "direction": ["outgoing", "outgoing"],
    })
    return feed, gold


def test_run_route_returns_predictions_and_timing():
    feed, gold = _data()
    route = RulesRoute()
    route.fit(feed, gold)
    preds, timing = run_route(route, feed)
    assert set(preds.columns) >= {"txn_id", "canonical_merchant", "category", "direction"}
    assert len(preds) == 2
    assert timing["n"] == 2
    assert timing["avg_ms"] >= 0.0
