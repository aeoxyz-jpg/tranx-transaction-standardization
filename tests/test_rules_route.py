import polars as pl
from tranx.routes.rules import RulesRoute
from tranx.schema import Txn


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


def _txn(desc, amount=-10.0, mcc=None):
    return Txn("Tx", "C1", desc, "O-CC-M", mcc, amount, "credit_card",
               "2026-01-01", "USA", "USD")


def test_fuzzy_matches_known_merchant():
    r = RulesRoute()
    r.fit(*_train())
    out = r.standardize(_txn("McDonald's #999"))
    assert out.canonical_merchant == "McDonald's"


def test_fuzzy_matches_through_hard_noise():
    # Embedded store-id + city + state (hard-mode dirt) should still resolve to
    # the known merchant via token-set matching, without enumerating city names.
    r = RulesRoute()
    r.fit(*_train())
    assert r.standardize(_txn("BP F1234 ATLANTA GA", amount=-30.0)).canonical_merchant == "BP"


def test_fuzzy_does_not_match_on_shared_city_token():
    # A different merchant that merely shares a city token must NOT match.
    r = RulesRoute()
    r.fit(*_train())
    out = r.standardize(_txn("Wendys ATLANTA GA", amount=-9.0))
    assert out.canonical_merchant != "BP"


def test_direction_from_amount_sign():
    r = RulesRoute()
    r.fit(*_train())
    assert r.standardize(_txn("Wage", amount=2000.0)).direction == "incoming"
    assert r.standardize(_txn("McDonald's #5", amount=-10.0)).direction == "outgoing"


def test_category_from_learned_mcc_prior():
    # mcc 5814 was seen in train labeled Food & Dining, so the learned
    # mcc -> category prior applies even to an unknown merchant string.
    r = RulesRoute()
    r.fit(*_train())
    out = r.standardize(_txn("Some New Diner", amount=-12.0, mcc=5814))
    assert out.category == "Food & Dining"


def test_unknown_mcc_falls_back_to_merchant_then_default():
    r = RulesRoute()
    r.fit(*_train())
    # mcc 9999 never seen in train -> fall back to merchant-majority category.
    # "BP #5" strips to "BP", which matches the learned canonical "BP".
    out = r.standardize(_txn("BP #5", amount=-30.0, mcc=9999))
    assert out.category == "Transportation"


def test_transaction_type_rows_still_teach_category():
    import polars as pl
    from tranx.routes.rules import RulesRoute
    feed = pl.DataFrame({"txn_id": ["a", "b"], "description": ["SALARY #1", "Starbucks #2"],
                         "mcc": [None, 5814]}, schema_overrides={"mcc": pl.Int64})
    gold = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": [None, "Starbucks"],
                         "txn_type": ["Salary", None], "category": ["Income", "Food & Dining"],
                         "direction": ["incoming", "outgoing"]},
                        schema_overrides={"canonical_merchant": pl.Utf8, "txn_type": pl.Utf8})
    r = RulesRoute()
    r.fit(feed, gold)
    assert r.standardize(_txn("SALARY #99", mcc=None, amount=2500.0)).category == "Income"
    # type labels are not offered as merchants (e.g. as Jev candidates)
    assert "Salary" not in r._canon_clean_to_name.values()
