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
    assert args.eval_cap == 1000


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
