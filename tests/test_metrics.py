import polars as pl
from tranx.eval import metrics


def _pred_gold():
    pred = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3", "T4"],
        "canonical_merchant": ["McDonald's", "McDonald's", "BP", "BP"],
        "category": ["Food & Dining", "Food & Dining", "Transportation", "Shopping & Retail"],
        "direction": ["outgoing", "outgoing", "outgoing", "outgoing"],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3", "T4"],
        "canonical_merchant": ["McDonald's", "McDonald's", "BP", "BP"],
        "category": ["Food & Dining", "Food & Dining", "Transportation", "Transportation"],
        "direction": ["outgoing", "outgoing", "outgoing", "outgoing"],
    })
    return pred, gold


def test_category_accuracy():
    pred, gold = _pred_gold()
    assert metrics.category_accuracy(pred, gold) == 0.75


def test_merchant_exact_match():
    pred, gold = _pred_gold()
    assert metrics.merchant_exact_match(pred, gold) == 1.0


def test_merchant_normalized_match_credits_case_and_punctuation():
    pred = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "canonical_merchant": ["PARAMEDIC", "Raising Canes", "AMZN"],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "canonical_merchant": ["Paramedic", "Raising Cane's", "Fair"],
    })
    assert metrics.merchant_exact_match(pred, gold) == 0.0
    assert metrics.merchant_normalized_match(pred, gold) == 2 / 3


def test_direction_accuracy():
    pred, gold = _pred_gold()
    assert metrics.direction_accuracy(pred, gold) == 1.0


def test_macro_f1_between_0_and_1():
    pred, gold = _pred_gold()
    f1 = metrics.category_macro_f1(pred, gold)
    assert 0.0 <= f1 <= 1.0


def test_dedup_ratio():
    pred = pl.DataFrame({"canonical_merchant": ["McDonald's", "McDonald's", "BP"]})
    # 3 rows collapse to 2 distinct merchants -> ratio 2/3
    assert metrics.dedup_ratio(pred, raw_distinct=3) == 2 / 3


def test_merchant_spend_kpi_perfect_when_merchants_correct():
    feed = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "customer_id": ["C1", "C1", "C1"],
        "amount": [-10.0, -12.0, -40.0],
    })
    pred = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "canonical_merchant": ["McDonald's", "McDonald's", "BP"],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "canonical_merchant": ["McDonald's", "McDonald's", "BP"],
    })
    res = metrics.merchant_spend_kpi(feed, pred, gold, tolerance=0.01)
    assert res["mae"] == 0.0
    assert res["within_tolerance"] == 1.0


def test_merchant_spend_kpi_penalizes_misassignment():
    feed = pl.DataFrame({
        "txn_id": ["T1", "T2"],
        "customer_id": ["C1", "C1"],
        "amount": [-10.0, -40.0],
    })
    # predicts both as McDonald's -> BP bucket missing, McDonald's inflated
    pred = pl.DataFrame({
        "txn_id": ["T1", "T2"],
        "canonical_merchant": ["McDonald's", "McDonald's"],
    })
    gold = pl.DataFrame({
        "txn_id": ["T1", "T2"],
        "canonical_merchant": ["McDonald's", "BP"],
    })
    res = metrics.merchant_spend_kpi(feed, pred, gold, tolerance=0.01)
    assert res["mae"] > 0.0
    assert res["within_tolerance"] < 1.0


def test_null_gold_merchant_rows_are_not_scored():
    import polars as pl
    from tranx.eval import metrics
    gold = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["Starbucks", None],
                         "category": ["Food & Dining", "Income"]},
                        schema_overrides={"canonical_merchant": pl.Utf8})
    pred = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["STARBUCKS", "Salary"],
                         "category": ["Food & Dining", "Income"]})
    assert metrics.merchant_exact_match(pred, gold) == 0.0
    assert metrics.merchant_normalized_match(pred, gold) == 1.0
    assert metrics.category_accuracy(pred, gold) == 1.0


def test_spend_kpi_uses_normalized_names():
    import polars as pl
    from tranx.eval import metrics
    feed = pl.DataFrame({"txn_id": ["a", "b"], "customer_id": ["C1", "C1"], "amount": [-5.0, -7.0]})
    gold = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["Starbucks", "Starbucks"]})
    pred = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["STARBUCKS", "Starbucks"]})
    assert metrics.merchant_spend_kpi(feed, pred, gold)["within_tolerance"] == 1.0
