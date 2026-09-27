import polars as pl
from tranx.cli import build_parser, _cap_eval, _split_unseen


def test_parser_has_subcommands():
    parser = build_parser()
    args = parser.parse_args(["synth", "--n", "100"])
    assert args.command == "synth"
    assert args.n == 100
    assert args.hard is False


def test_synth_accepts_hard_flag():
    parser = build_parser()
    args = parser.parse_args(["synth", "--hard"])
    assert args.hard is True


def test_run_requires_route():
    parser = build_parser()
    args = parser.parse_args(["run", "--route", "rules"])
    assert args.command == "run"
    assert args.route == "rules"
    assert args.split == "random"


def test_run_accepts_unseen_split():
    parser = build_parser()
    args = parser.parse_args(["run", "--route", "rules", "--split", "unseen"])
    assert args.split == "unseen"


def test_run_accepts_slm_model():
    parser = build_parser()
    args = parser.parse_args(["run", "--route", "slm_fewshot", "--slm-model", "gemma2:2b"])
    assert args.slm_model == "gemma2:2b"


def test_eval_subcommand():
    parser = build_parser()
    args = parser.parse_args(["eval"])
    assert args.command == "eval"
    assert args.cap_random is None and args.cap_unseen is None  # config.EVAL_CAPS


def test_eval_cap_flags_override():
    args = build_parser().parse_args(["eval", "--cap-random", "5", "--cap-unseen", "7"])
    assert (args.cap_random, args.cap_unseen) == (5, 7)


def test_cap_eval_subsamples_and_aligns():
    feed = pl.DataFrame({"txn_id": [f"T{i}" for i in range(10)]})
    gold = pl.DataFrame({"txn_id": [f"T{i}" for i in range(10)], "v": list(range(10))})
    capped_feed, capped_gold = _cap_eval(feed, gold, cap=4, seed=42)
    assert len(capped_feed) == 4
    # gold is filtered to exactly the capped txn_ids
    assert set(capped_gold["txn_id"]) == set(capped_feed["txn_id"])


def test_cap_eval_noop_when_cap_zero_or_larger():
    feed = pl.DataFrame({"txn_id": ["T0", "T1"]})
    gold = pl.DataFrame({"txn_id": ["T0", "T1"]})
    assert len(_cap_eval(feed, gold, cap=0, seed=42)[0]) == 2
    assert len(_cap_eval(feed, gold, cap=99, seed=42)[0]) == 2


def test_split_unseen_has_disjoint_merchants():
    feed = pl.DataFrame({"txn_id": [f"T{i}" for i in range(10)]})
    gold = pl.DataFrame({
        "txn_id": [f"T{i}" for i in range(10)],
        "canonical_merchant": [f"M{i % 5}" for i in range(10)],  # 5 distinct merchants
    })
    tf, tg, ef, eg = _split_unseen(feed, gold, seed=42, frac=0.4)
    train_merchants = set(tg["canonical_merchant"])
    eval_merchants = set(eg["canonical_merchant"])
    # the whole point: no merchant appears in both train and eval
    assert train_merchants.isdisjoint(eval_merchants)
    assert len(eval_merchants) >= 1
    # feed/gold stay aligned by txn_id
    assert set(ef["txn_id"]) == set(eg["txn_id"])


def test_brand_families_group_sibling_labels():
    from tranx.cli import brand_families
    f = brand_families(["Walmart", "Walmart Pharmacy", "Bank", "Community Bank",
                        "Saks Fifth Avenue", "Saks Off 5th", "Starbucks"])
    assert f["Walmart"] == f["Walmart Pharmacy"]
    assert f["Bank"] == f["Community Bank"]
    assert f["Saks Fifth Avenue"] == f["Saks Off 5th"]
    assert f["Starbucks"] not in {f["Walmart"], f["Bank"]}


def test_unseen_assignment_is_stable_under_label_changes():
    from tranx.cli import _in_eval
    keys = [f"Merchant {i}" for i in range(500)]
    before = {k for k in keys if _in_eval(k, 0.2)}
    after = {k for k in keys[:300] if _in_eval(k, 0.2)}  # other labels removed
    assert after == {k for k in before if k in keys[:300]}
    assert 60 <= len(before) <= 140



def test_generic_hub_label_does_not_chain_brands():
    from tranx.cli import brand_families
    f = brand_families(["Pharmacy", "Kroger Pharmacy", "Safeway Pharmacy", "Walmart Pharmacy",
                        "Walmart", "Local Church", "Local Food Bank"])
    assert f["Walmart"] == f["Walmart Pharmacy"]          # real sibling
    assert f["Kroger Pharmacy"] != f["Walmart Pharmacy"]  # not chained through "Pharmacy"
    assert f["Local Church"] != f["Local Food Bank"]      # a shared generic first word is not a brand


# --- eval_rows (spec D4) -------------------------------------------------------

def _toy_feed_gold():
    """200 rows: 40 repeating descriptors, 40 one-off ones, one conflicting descriptor
    ("SHARED DESC" maps to two merchants) and some merchant-less rows."""
    ids, desc, merch, ttype, cat = [], [], [], [], []
    for i in range(200):
        ids.append(f"T{i:08d}")
        k = i % 40
        if i % 50 == 7:
            desc.append("SHARED DESC")
            merch.append("Alpha" if i < 100 else "Beta")
            ttype.append(None)
        elif i >= 120 and i % 2 == 0:  # one-off descriptors (cache misses)
            desc.append(f"SHOP {k} STORE {i}")
            merch.append(f"Shop {k}")
            ttype.append(None)
        elif k < 5:
            desc.append(f"SALARY {k}")
            merch.append(None)
            ttype.append("Salary")
        else:
            desc.append(f"SHOP {k}")
            merch.append(f"Shop {k}")
            ttype.append(None)
        cat.append(f"C{k % 3}")
    feed = pl.DataFrame({"txn_id": ids, "description": desc})
    gold = pl.DataFrame({
        "txn_id": ids, "category": cat, "canonical_merchant": merch, "txn_type": ttype,
        "direction": ["outgoing"] * 200,
        "origin": [None if m is None else "source" for m in merch],
        "noise_abbrev": [False] * 200, "noise_trunc": [False] * 200,
    }, schema_overrides={"canonical_merchant": pl.Utf8, "txn_type": pl.Utf8,
                         "origin": pl.Utf8})
    return feed, gold


def _eval_rows(view, **kw):
    from tranx.cli import eval_rows
    feed, gold = _toy_feed_gold()
    return eval_rows("random", view, feed=feed, gold=gold,
                     caps=kw.get("caps", {"random": 1000, "unseen": 2000}), seed=42)


def test_eval_rows_model_view_excludes_verbatim_and_counts_them():
    from tranx.cli import _split_random
    feed, gold = _toy_feed_gold()
    tf, _, ef, _ = _split_random(feed, gold, 42)
    train_desc = set(tf["description"])
    er = _eval_rows("model")
    assert not set(er.eval_feed["description"]) & train_desc
    assert er.stats["removed_verbatim"] == sum(d in train_desc for d in ef["description"])
    assert er.stats["removed_verbatim"] > 0
    assert er.stats["n_eval"] == len(ef)


def test_eval_rows_model_view_one_row_per_description_min_txn_id():
    from tranx.cli import _split_random
    feed, gold = _toy_feed_gold()
    _, _, ef, _ = _split_random(feed, gold, 42)
    er = _eval_rows("model")
    assert er.eval_feed["description"].n_unique() == len(er.eval_feed)
    for tid, d in er.eval_feed.select("txn_id", "description").iter_rows():
        assert tid == min(ef.filter(pl.col("description") == d)["txn_id"])
    assert er.eval_gold["txn_id"].to_list() == er.eval_feed["txn_id"].to_list()


def test_eval_rows_drops_conflicting_descriptors():
    for view in ("model", "ideal_cache"):
        er = _eval_rows(view)
        assert "SHARED DESC" not in set(er.eval_feed["description"])
        assert er.stats["conflicts"] == {"descriptors": 1, "rows": 4,
                                         "eval_rows": er.stats["conflicts"]["eval_rows"]}


def test_eval_rows_deterministic_and_capped():
    a = _eval_rows("model", caps={"random": 3})
    b = _eval_rows("model", caps={"random": 3})
    assert a.eval_feed["txn_id"].to_list() == b.eval_feed["txn_id"].to_list()
    assert len(a.eval_feed) == 3 and a.stats["capped_n"] == 3
    assert a.eval_feed["txn_id"].to_list() == sorted(a.eval_feed["txn_id"].to_list())


def test_eval_rows_ideal_view_hits_take_train_gold():
    from tranx.cli import _split_random
    feed, gold = _toy_feed_gold()
    tf, tg, _, _ = _split_random(feed, gold, 42)
    train = tf.join(tg, on="txn_id").sort("txn_id")
    er = _eval_rows("ideal_cache")
    model = _eval_rows("model")
    c = er.cache
    assert c["txn_id"].to_list() == er.eval_feed["txn_id"].to_list()
    hits = c.filter(pl.col("cache_hit")).join(er.eval_feed, on="txn_id")
    assert len(hits) and er.stats["cache_hit_rate"] == round(len(hits) / er.stats["n_eval"], 4)
    for row in hits.iter_rows(named=True):
        first = train.filter(pl.col("description") == row["description"]).row(0, named=True)
        assert row["rep_txn_id"] == first["txn_id"]
        assert row["cache_merchant"] == first["canonical_merchant"]
        assert row["cache_category"] == first["category"]
    # Misses point at the model-view row of their descriptor.
    misses = c.filter(~pl.col("cache_hit"))
    assert set(misses["rep_txn_id"]) <= set(model.eval_feed["txn_id"])


# --- eval end to end (manifest + preds) ------------------------------------------

def _fake_encoder(texts):
    import numpy as np
    vecs = []
    for t in texts:
        v = np.zeros(26)
        for ch in t.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1.0
        vecs.append(v)
    return np.array(vecs)


def test_eval_writes_manifest_and_loadable_preds(fixture_df, tmp_path, monkeypatch):
    import json
    from tranx import config
    from tranx.cli import main
    from tranx.routes import embedding
    from tranx.synth.feed import build_feed
    from tranx.eval.manifest import load_preds
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "RUN_DIR", tmp_path / "reports" / "run")
    monkeypatch.setattr(config, "PREDS_DIR", tmp_path / "reports" / "preds")
    monkeypatch.setattr(embedding, "_default_encoder", _fake_encoder)
    feed, gold = build_feed(fixture_df, hard=True)
    (tmp_path / "data").mkdir()
    feed.write_parquet(tmp_path / "data" / "bank_feed.parquet")
    gold.write_parquet(tmp_path / "data" / "gold.parquet")

    routes = ["rules", "embedding", "cleaner", "metadata"]
    main(["eval", "--routes", ",".join(routes), "--cap-random", "10", "--no-plot"])

    manifest = json.loads((tmp_path / "reports" / "run" / "manifest.json").read_text())
    assert manifest["caps"]["random"] == 10
    assert len(manifest["splits"]["random"]["model"]) == 10
    for split in ("random", "unseen"):
        for view in ("model", "ideal_cache"):
            assert manifest["splits"][split][view]
            for route in routes:
                df = load_preds(split, view, route, manifest)
                assert ("cache_hit" in df.columns) == (view == "ideal_cache")
                assert df["candidates"].null_count() == len(df)  # no Jev route here
    stats = json.loads((tmp_path / "reports" / "run" / "eval_stats.json").read_text())
    assert set(stats["random"]) == {"model", "ideal_cache"}
    results = json.loads((tmp_path / "reports" / "leaderboard.json").read_text())
    by = {(r["split"], r["view"], r["route"]): r for r in results}
    assert by[("random", "model", "cleaner")]["category_acc"] == "-"
    assert by[("random", "model", "metadata")]["merchant_norm"] == "-"
    assert isinstance(by[("random", "model", "rules")]["no_match_rate"], float)
    assert "kpi_within_tol" not in by[("random", "model", "rules")]
    assert isinstance(by[("random", "ideal_cache", "rules")]["kpi_within_tol"], float)
    # D2: abbreviated local-merchant rows leave the model-view headline merchant score only.
    ids = set(manifest["splits"]["random"]["model"])
    g = gold.filter(pl.col("txn_id").is_in(ids) & pl.col("canonical_merchant").is_not_null())
    n_excl = g.filter((pl.col("origin") == "local") & pl.col("noise_abbrev")).height
    assert by[("random", "model", "rules")]["headline_excluded"] == n_excl
    assert by[("random", "model", "rules")]["merchant_rows"] == g.height - n_excl
    assert by[("random", "ideal_cache", "rules")]["headline_excluded"] == 0
