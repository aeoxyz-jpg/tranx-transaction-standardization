import polars as pl
from tranx.synth.feed import build_feed
from tranx import config


def test_hard_mode_changes_descriptions_but_keeps_gold(fixture_df):
    feed, gold = build_feed(fixture_df, seed=42)
    hfeed, hgold = build_feed(fixture_df, seed=42, hard=True)
    # gold labels are unchanged: derived from the original clean description
    assert gold.sort("txn_id").to_dicts() == hgold.sort("txn_id").to_dicts()
    # but the feed descriptions are dirtied
    clean_desc = set(feed["description"].to_list())
    hard_desc = set(hfeed["description"].to_list())
    assert clean_desc != hard_desc


def test_build_feed_splits_and_columns(fixture_df):
    feed, gold = build_feed(fixture_df, seed=42)
    assert len(feed) == len(gold) == len(fixture_df)
    feed_cols = set(feed.columns)
    # Leakage guard: feed must not expose labels
    assert "category" not in feed_cols
    assert "canonical_merchant" not in feed_cols
    assert {"txn_id", "customer_id", "amount", "payment_method",
            "transaction_type_code", "mcc", "posted_date"} <= feed_cols
    assert {"txn_id", "category", "canonical_merchant", "direction"} == set(gold.columns)


def test_feed_is_deterministic(fixture_df):
    f1, g1 = build_feed(fixture_df, seed=42)
    f2, g2 = build_feed(fixture_df, seed=42)
    assert f1.to_dicts() == f2.to_dicts()
    assert g1.to_dicts() == g2.to_dicts()


def test_income_is_incoming(fixture_df):
    feed, gold = build_feed(fixture_df, seed=42)
    joined = feed.join(gold, on="txn_id")
    income = joined.filter(pl.col("category") == "Income")
    # Income rows are incoming except the small refund-flip fraction
    assert (income["direction"] == "incoming").sum() >= int(0.9 * len(income))


def test_mcc_coverage_is_partial(fixture_df):
    feed, gold = build_feed(fixture_df, seed=42)
    frac_present = feed["mcc"].is_not_null().sum() / len(feed)
    assert frac_present < 1.0  # only some rows carry an MCC


def test_customer_count_bounded(fixture_df):
    feed, _ = build_feed(fixture_df, seed=42, n_customers=5)
    assert feed["customer_id"].n_unique() <= 5
