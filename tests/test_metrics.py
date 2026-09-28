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


def test_merchant_subsets_splits_recoverable_abbreviated_truncated():
    pred = pl.DataFrame({
        "txn_id": ["a", "b", "c", "d"],
        "canonical_merchant": ["Joe's Diner", "Joe's Diner", "WRONG", "Joe's Diner"],
    })
    gold = pl.DataFrame({
        "txn_id": ["a", "b", "c", "d"],
        "canonical_merchant": ["Joe's Diner", "Joe's Diner", "Joe's Diner", "Joe's Diner"],
        "noise_abbrev": [False, True, True, False],
        "noise_trunc": [False, False, False, True],
        "origin": ["source", "local", "local", "source"],
    })
    res = metrics.merchant_subsets(pred, gold)
    # recoverable: rows a, d (neither noise fired) -> both correct -> 1.0
    assert res["recoverable"] == 1.0
    # abbreviated: rows b, c -> b correct, c wrong -> 0.5
    assert res["abbreviated"] == 0.5
    # truncated: row d only -> correct -> 1.0
    assert res["truncated"] == 1.0
    assert res["origin"]["source"] == 1.0  # rows a, d both correct
    assert res["origin"]["local"] == 0.5  # rows b, c: b correct, c wrong


def test_merchant_subsets_none_when_subset_empty():
    pred = pl.DataFrame({"txn_id": ["a"], "canonical_merchant": ["Joe's Diner"]})
    gold = pl.DataFrame({
        "txn_id": ["a"], "canonical_merchant": ["Joe's Diner"],
        "noise_abbrev": [False], "noise_trunc": [False], "origin": ["source"],
    })
    res = metrics.merchant_subsets(pred, gold)
    assert res["abbreviated"] is None
    assert res["truncated"] is None
    assert res["recoverable"] == 1.0


def test_retrieval_recall_share_of_gold_in_candidates():
    gold_merchants = ["Starbucks", "BP", "Unknown Merchant"]
    candidate_lists = [["Starbucks", "Costa"], ["Shell", "Texaco"], ["Foo", "Bar"]]
    # only the first row's gold is in its candidate list -> 1/3
    assert metrics.retrieval_recall(gold_merchants, candidate_lists) == 1 / 3


def test_retrieval_recall_normalized_match():
    gold_merchants = ["Raising Cane's"]
    candidate_lists = [["RAISING CANES"]]
    assert metrics.retrieval_recall(gold_merchants, candidate_lists) == 1.0


def test_retrieval_recall_empty_is_zero():
    assert metrics.retrieval_recall([], []) == 0.0


def test_spend_kpi_uses_normalized_names():
    import polars as pl
    from tranx.eval import metrics
    feed = pl.DataFrame({"txn_id": ["a", "b"], "customer_id": ["C1", "C1"], "amount": [-5.0, -7.0]})
    gold = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["Starbucks", "Starbucks"]})
    pred = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["STARBUCKS", "Starbucks"]})
    assert metrics.merchant_spend_kpi(feed, pred, gold)["within_tolerance"] == 1.0


def test_merchant_ok_credits_parent_brand_by_line_of_business():
    from tranx.eval.metrics import merchant_ok
    assert merchant_ok("DOUBLETREE BY HILTON", "DoubleTree by Hilton")
    assert merchant_ok("Hilton", "DoubleTree by Hilton")                 # same line: always
    assert not merchant_ok("Uber", "Uber Eats")                          # other line, category unknown
    assert not merchant_ok("Uber", "Uber Eats", category_ok=False)
    assert merchant_ok("Uber", "Uber Eats", category_ok=True)
    assert not merchant_ok("DoubleTree by Hilton", "Hilton")             # child for parent is not credited
    assert not merchant_ok("Hilton", None)


def test_normalized_match_uses_row_category_for_parent_credit():
    from tranx.eval.metrics import merchant_normalized_match
    gold = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["Walmart Pharmacy"] * 2,
                         "category": ["Healthcare & Medical"] * 2})
    pred = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["Walmart"] * 2,
                         "category": ["Healthcare & Medical", "Shopping & Retail"]})
    assert merchant_normalized_match(pred, gold) == 0.5
