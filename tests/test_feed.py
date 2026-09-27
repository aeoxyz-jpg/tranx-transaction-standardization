import polars as pl
from tranx.synth.feed import build_feed
from tranx import config


def test_hard_mode_changes_descriptions_but_keeps_gold(fixture_df):
    feed, gold = build_feed(fixture_df, seed=42)
    hfeed, hgold = build_feed(fixture_df, seed=42, hard=True)
    # gold labels are unchanged: derived from the original clean description
    labels = ["txn_id", "category", "canonical_merchant", "txn_type", "direction",
              "origin"]
    assert (gold.select(labels).sort("txn_id").to_dicts()
            == hgold.select(labels).sort("txn_id").to_dicts())
    # the non-label feed fields do not move either
    cols = ["txn_id", "customer_id", "amount", "mcc", "posted_date", "payment_method"]
    assert feed.select(cols).to_dicts() == hfeed.select(cols).to_dicts()
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
    assert gold.columns == config.GOLD_COLUMNS
    # merchant and transaction type are mutually exclusive, exactly one is set
    assert (gold["canonical_merchant"].is_null() != gold["txn_type"].is_null()).all()
    # the type code carries direction and method only (no category-determined flag)
    assert feed["transaction_type_code"].str.contains(r"^[IO]-[A-Z]+$").all()


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


def test_local_merchants_reassigned_within_category(fixture_df):
    from tranx.synth.local import local_merchants
    feed, gold = build_feed(fixture_df, seed=42)
    merchant = gold.filter(pl.col("canonical_merchant").is_not_null())
    # origin is set exactly on merchant rows
    assert (gold["origin"].is_null() == gold["canonical_merchant"].is_null()).all()
    local = gold.filter(pl.col("origin") == "local")
    share = len(local) / len(merchant)
    assert 0.15 < share < 0.35
    # a local keeps the row's category
    source = sorted(set(merchant.filter(pl.col("origin") == "source")["canonical_merchant"])
                    | set(gold["txn_type"].drop_nulls()))
    cat_of = dict(local_merchants(source, 42, config.LOCAL_MERCHANTS_N))
    for name, cat in local.select("canonical_merchant", "category").iter_rows():
        assert cat_of[name] == cat
    # never Income / Financial Services
    assert not set(local["category"]) & {"Income", "Financial Services"}
    # every source label keeps its first row (the fixture cycles 10 descriptions)
    head = gold.filter(pl.col("txn_id") < "T00000010", pl.col("origin").is_not_null())
    assert (head["origin"] == "source").all()
    assert set(merchant.filter(pl.col("origin") == "source")["canonical_merchant"]) == {
        "McDonald's", "BP on Buford Hwy", "BP @ Pleasant Hills", "Amazon", "Walgreens",
        "Starbucks", "Disney+"}
    # non-hard: a local row's description is the local name; flags are False
    joined = feed.join(gold, on="txn_id").filter(pl.col("origin") == "local")
    assert (joined["description"] == joined["canonical_merchant"]).all()
    assert not gold["noise_abbrev"].any() and not gold["noise_trunc"].any()


def test_hard_mode_repeats_descriptors_per_location(fixture_df):
    import math
    from tranx.synth.feed import location_descriptor
    feed, gold = build_feed(fixture_df, seed=42, hard=True)
    # repeat structure: far fewer descriptors than rows
    assert feed["description"].n_unique() < 0.6 * len(feed)
    joined = feed.join(gold, on="txn_id").with_columns(
        name=pl.coalesce("canonical_merchant", "txn_type"))
    counts = joined.group_by("name").len()
    # no merged labels in the fixture, so the descriptor text is the gold name
    for name, n in counts.iter_rows():
        n_loc = min(config.LOCATIONS_CAP, math.ceil(n / config.LOCATIONS_ROWS_PER))
        allowed = {location_descriptor(name, loc, name, "", "")
                   for loc in range(n_loc)}
        got = set(joined.filter(pl.col("name") == name)["description"])
        assert got <= {d for d, _, _ in allowed}, name
    # the noise flags come from the same (name, location) draw
    assert gold["noise_abbrev"].dtype == pl.Boolean


def test_location_descriptor_is_stable_across_processes():
    import os
    import subprocess
    import sys
    from tranx.synth.feed import location_descriptor
    code = ("from tranx.synth.feed import location_descriptor as f;"
            "print(repr(f('Blue Heron Bakery', 3, 'Blue Heron Bakery', 'Food & Dining', 'USA')))")
    here = repr(location_descriptor("Blue Heron Bakery", 3, "Blue Heron Bakery",
                                    "Food & Dining", "USA"))
    for hs in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, check=True).stdout.strip()
        assert out == here
