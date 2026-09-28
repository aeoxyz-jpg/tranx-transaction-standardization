import argparse
import hashlib
import json
import random
from dataclasses import dataclass
import polars as pl
from tranx import config
from tranx.synth.sample import load_sample
from tranx.synth.feed import build_feed
from tranx.synth.canonical import strip_coverage
from tranx.synth.families import _STOP, _tokens, _FAMILY_LINKS, brand_families, _in_eval  # noqa: F401
from tranx.eval.harness import run_route
from tranx.eval import metrics
from tranx.eval.report import save_leaderboard, plot_customer_spend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tranx")
    sub = parser.add_subparsers(dest="command", required=True)

    p_synth = sub.add_parser("synth", help="download, sample, synthesize feed")
    p_synth.add_argument("--n", type=int, default=config.SAMPLE_SIZE)
    p_synth.add_argument("--hard", action="store_true",
                         help="dirty the feed descriptors into card-network-style noise")

    p_run = sub.add_parser("run", help="score one route on one split's model view (as eval does)")
    p_run.add_argument("--route", required=True,
                       choices=["rules", "embedding", "slm_fewshot", "slm_lora", "jev", "jev_merchant",
                                "jev_slm", "cleaner", "metadata"])
    p_run.add_argument("--adapter-path", default="adapters/qwen-hard",
                       help="LoRA adapter dir for the slm_lora route")
    p_run.add_argument("--base-model", default=config.MLX_BASE,
                       help="MLX base model for the slm_lora route")
    p_run.add_argument("--split", choices=["random", "unseen"], default="random",
                       help="random 80/20 split, or merchant-disjoint (unseen merchants)")
    p_run.add_argument("--eval-cap", type=int, default=0,
                       help="cap model-view descriptors (0 = config.EVAL_CAPS)")
    p_run.add_argument("--slm-model", default=config.SLM_MODEL,
                       help="Ollama model for the slm_fewshot route")

    p_eval = sub.add_parser("eval", help="run all routes on both splits, write leaderboard")
    p_eval.add_argument("--cap-random", type=int, default=None,
                        help="model-view descriptor cap on the random split "
                             f"(default {config.EVAL_CAPS['random']})")
    p_eval.add_argument("--cap-unseen", type=int, default=None,
                        help="model-view descriptor cap on the unseen split "
                             f"(default {config.EVAL_CAPS['unseen']})")
    p_eval.add_argument("--routes", default="rules,embedding,slm_fewshot",
                        help="comma-separated routes to score")
    p_eval.add_argument("--name", default="leaderboard_hard.md",
                        help="leaderboard file name under reports/")
    p_eval.add_argument("--also", default="",
                        help="optional extra leaderboard: NAME=route1,route2 (subset of --routes)")
    p_eval.add_argument("--no-plot", action="store_true", help="skip the spend plot")

    p_ft = sub.add_parser("finetune-export",
                          help="export MLX LoRA training data from the unseen-split "
                               "train portion (eval merchants stay unseen by training)")
    p_ft.add_argument("--out", default=str(config.DATA_DIR / "ft"))
    p_ft.add_argument("--cap", type=int, default=4000,
                      help="max training rows (sampled from the unseen-split train)")
    return parser


def _ids_split(feed, gold, eval_ids):
    keep = pl.col("txn_id").is_in(eval_ids)
    return (feed.filter(~keep), gold.filter(~keep),
            feed.filter(keep), gold.filter(keep))


def _split_random(feed: pl.DataFrame, gold: pl.DataFrame, seed: int):
    """Random 80/20 split. Train and eval share merchants (in-distribution)."""
    shuffled = feed.sample(fraction=1.0, shuffle=True, seed=seed)
    cut = int(0.8 * len(shuffled))
    eval_ids = set(shuffled["txn_id"][cut:].to_list())
    return _ids_split(feed, gold, eval_ids)


def _split_unseen(feed: pl.DataFrame, gold: pl.DataFrame, seed: int, frac: float = 0.2):
    """Merchant-disjoint split: ~frac of distinct merchants are held out, so the
    eval set contains only merchants the training set never saw. This is the
    robustness test — learned-canonical routes cannot memorize their way through."""
    merchants = sorted(gold["canonical_merchant"].drop_nulls().unique().to_list())
    family = brand_families(merchants)
    # Whole brand families are held out together, so no eval brand has a sibling
    # label ("Walmart" for "Walmart Pharmacy") in the training list.
    eval_merchants = {m for m in merchants if _in_eval(family[m], frac)}
    eval_ids = set(
        gold.filter(pl.col("canonical_merchant").is_in(eval_merchants))["txn_id"].to_list())
    # Transaction-type rows have no merchant; send the same fraction to eval so the
    # category mix of the unseen split stays comparable.
    no_merchant = gold.filter(pl.col("canonical_merchant").is_null())["txn_id"].to_list()
    eval_ids |= {t for t in no_merchant if _in_eval(t, frac)}
    return _ids_split(feed, gold, eval_ids)


_SPLITTERS = {"random": _split_random, "unseen": _split_unseen}


VIEWS = ("model", "ideal_cache")


@dataclass
class EvalRows:
    split: str
    view: str
    train_feed: pl.DataFrame
    train_gold: pl.DataFrame
    eval_feed: pl.DataFrame
    eval_gold: pl.DataFrame
    cache: pl.DataFrame | None  # ideal_cache view only, aligned with eval_feed
    stats: dict


def _conflicting_descriptors(feed: pl.DataFrame, gold: pl.DataFrame) -> tuple[set, int]:
    """Descriptions that map to more than one (canonical_merchant, txn_type) across
    the whole feed, and how many rows carry them."""
    key = pl.concat_str([pl.col("canonical_merchant").fill_null(""),
                         pl.col("txn_type").fill_null("")], separator="\x1f").alias("key")
    j = feed.select("txn_id", "description").join(gold.select("txn_id", key), on="txn_id")
    g = j.group_by("description").agg(pl.col("key").n_unique().alias("k"), pl.len().alias("n"))
    bad = g.filter(pl.col("k") > 1)
    return set(bad["description"].to_list()), int(bad["n"].sum())


def _composition(gold: pl.DataFrame) -> dict:
    cats = gold["category"].value_counts()
    origins = gold["origin"].fill_null("none").value_counts()
    return {
        "rows": len(gold),
        "merchants": gold["canonical_merchant"].drop_nulls().n_unique(),
        "category": dict(sorted(zip(cats["category"].to_list(), cats["count"].to_list()))),
        "origin": dict(sorted(zip(origins["origin"].to_list(), origins["count"].to_list()))),
    }


def _build_views(split: str, feed: pl.DataFrame, gold: pl.DataFrame,
                 caps: dict, seed: int) -> dict[str, EvalRows]:
    tf, tg, ef, eg = _SPLITTERS[split](feed, gold, seed)
    ef, eg = ef.sort("txn_id"), eg.sort("txn_id")
    conflict_desc, conflict_rows = _conflicting_descriptors(feed, gold)
    train_desc = set(tf["description"].to_list())
    in_train = pl.col("description").is_in(train_desc)
    conflict = pl.col("description").is_in(conflict_desc)

    # Model view: cache misses only, one row per descriptor (min txn_id), capped.
    removed_verbatim = ef.filter(in_train).height
    miss = ef.filter(~in_train & ~conflict)
    reps = miss.group_by("description").agg(pl.col("txn_id").min())["txn_id"]
    pool = miss.filter(pl.col("txn_id").is_in(reps)).sort("txn_id")
    cap = caps.get(split, 0)
    model_feed = pool
    if cap and len(pool) > cap:
        model_feed = pool.sample(n=cap, seed=seed).sort("txn_id")
    model_ids = model_feed["txn_id"]
    model_gold = eg.filter(pl.col("txn_id").is_in(model_ids)).sort("txn_id")

    # Ideal-cache view: every cache hit (descriptor seen in train, not conflicting)
    # takes the train gold of that descriptor; every eval row whose descriptor is in
    # the capped model view takes that descriptor's model-view prediction.
    train_rep = (tf.select("txn_id", "description").filter(~conflict)
                 .join(tg.select("txn_id", "canonical_merchant", "category"), on="txn_id")
                 .sort("txn_id").group_by("description", maintain_order=True).first()
                 .rename({"txn_id": "rep_txn_id", "canonical_merchant": "cache_merchant",
                          "category": "cache_category"}))
    hits = (ef.filter(in_train & ~conflict).select("txn_id", "description")
            .join(train_rep, on="description").with_columns(pl.lit(True).alias("cache_hit")))
    model_rep = model_feed.select("description", pl.col("txn_id").alias("rep_txn_id"))
    misses = (ef.select("txn_id", "description").join(model_rep, on="description")
              .with_columns(pl.lit(False).alias("cache_hit"),
                            pl.lit(None, dtype=pl.Utf8).alias("cache_merchant"),
                            pl.lit(None, dtype=pl.Utf8).alias("cache_category")))
    cols = ["txn_id", "cache_hit", "rep_txn_id", "cache_merchant", "cache_category"]
    cache = pl.concat([hits.select(cols), misses.select(cols)]).sort("txn_id")
    ideal_feed = ef.filter(pl.col("txn_id").is_in(cache["txn_id"])).sort("txn_id")
    ideal_gold = eg.filter(pl.col("txn_id").is_in(cache["txn_id"])).sort("txn_id")

    base = {
        "n_eval": len(ef),
        "removed_verbatim": removed_verbatim,
        "cache_hit_rate": round(len(hits) / max(1, len(ef)), 4),
        "conflicts": {"descriptors": len(conflict_desc), "rows": conflict_rows,
                      "eval_rows": ef.filter(conflict).height},
        "model_view_n": len(pool),
        "capped_n": len(model_feed),
    }
    before = _composition(eg)
    return {
        "model": EvalRows(split, "model", tf, tg, model_feed, model_gold, None, {
            **base, "view_n": len(model_feed),
            "composition": {"before": before, "after": _composition(model_gold)}}),
        "ideal_cache": EvalRows(split, "ideal_cache", tf, tg, ideal_feed, ideal_gold, cache, {
            **base, "view_n": len(ideal_feed), "ideal_hit_rows": len(hits),
            "composition": {"before": before, "after": _composition(ideal_gold)}}),
    }


def _load_feed_gold():
    return (pl.read_parquet(config.DATA_DIR / "bank_feed.parquet"),
            pl.read_parquet(config.DATA_DIR / "gold.parquet"))


def eval_rows(split: str, view: str, feed: pl.DataFrame | None = None,
              gold: pl.DataFrame | None = None, caps: dict = config.EVAL_CAPS,
              seed: int = config.SEED) -> EvalRows:
    """The rows every eval script scores (spec D4). view "model": eval rows whose
    description is not verbatim in train and not a conflicting descriptor, one row
    per description (min txn_id), seeded cap. view "ideal_cache": cache hits plus
    every eval row whose description is in the capped model view."""
    if feed is None or gold is None:
        feed, gold = _load_feed_gold()
    return _build_views(split, feed, gold, caps, seed)[view]


def _make_route(name: str, slm_model: str | None = None,
                adapter_path: str | None = None, base_model: str | None = None):
    if name == "rules":
        from tranx.routes.rules import RulesRoute
        return RulesRoute()
    if name == "embedding":
        from tranx.routes.embedding import EmbeddingRoute
        return EmbeddingRoute()
    if name == "cleaner":
        from tranx.routes.cleaner import CleanerRoute
        return CleanerRoute()
    if name == "metadata":
        from tranx.routes.metadata import MetadataRoute
        return MetadataRoute()
    if name in ("jev", "jev_merchant"):
        from tranx.routes.jev import JevRoute
        return JevRoute(merchant_choice=(name == "jev_merchant"))
    if name == "jev_slm":
        from tranx.routes.jev import JevRoute
        from tranx.routes.slm_fewshot import SlmFewshotRoute
        return JevRoute(fallback=SlmFewshotRoute(model=slm_model))
    from tranx.routes.slm_fewshot import SlmFewshotRoute
    if name == "slm_lora":
        from tranx.finetune.mlx_backend import make_mlx_chat
        from tranx.finetune.data import finetune_prompt
        chat = make_mlx_chat(base_model or config.MLX_BASE, adapter_path)
        # LoRA model was trained on raw hard descriptors, so do not pre-strip.
        return SlmFewshotRoute(chat_fn=chat, prompt_fn=finetune_prompt,
                               strip_prefix=False, name="slm_lora")
    return SlmFewshotRoute(model=slm_model)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _eval_manifest(views: dict, feed_path, gold_path, caps: dict, seed: int) -> dict:
    import inspect
    from tranx.eval.manifest import build_manifest, file_sha256
    from tranx.routes import slm_fewshot, jev
    from tranx.routes.rules import RulesRoute
    cand = {}
    for split, vs in views.items():
        # Jev's merchant candidates are retrieved from the rules route's known list.
        r = RulesRoute()
        r.fit(vs["model"].train_feed, vs["model"].train_gold)
        cand[split] = _sha256_text(json.dumps(sorted(set(r._canon_clean_to_name.values()))))
    # The few-shot block lives outside build_prompt, so it is hashed with it.
    prompt_src = inspect.getsource(slm_fewshot.build_prompt) + slm_fewshot._FEWSHOT
    return build_manifest(
        feed_sha256=file_sha256(feed_path),
        gold_sha256=file_sha256(gold_path),
        seed=seed,
        caps=dict(caps),
        filter={"drop_verbatim": True, "dedupe": True, "drop_conflicts": True},
        splits={split: {v: vs[v].eval_feed["txn_id"].to_list() for v in VIEWS}
                for split, vs in views.items()},
        slm={"model": config.SLM_MODEL, "prompt_sha256": _sha256_text(prompt_src)},
        jev={"model": jev.JEV_MODEL, "candidates_k": jev.MERCHANT_CANDIDATES},
        candidate_list_sha256=cand,
    )


_JEV_FIELDS = {"jev_choice": pl.Utf8, "jev_confidence": pl.Float64, "jev_p_choice": pl.Float64,
               "jev_p_none": pl.Float64, "jev_category_confidence": pl.Float64}


def _model_preds(route, er: EvalRows, preds: pl.DataFrame) -> pl.DataFrame:
    """Per-row predictions on the model view in the preds-parquet contract shape
    (without manifest_hash, which save_preds stamps)."""
    info = getattr(route, "row_info", {}) or {}
    ids = er.eval_feed["txn_id"].to_list()
    extra = pl.DataFrame(
        {"txn_id": ids,
         "candidates": [(info.get(t) or {}).get("candidates") for t in ids],
         **{f: [(info.get(t) or {}).get(f) for t in ids] for f in _JEV_FIELDS}},
        schema={"txn_id": pl.Utf8, "candidates": pl.List(pl.Utf8), **_JEV_FIELDS})
    p = preds.select("txn_id",
                     pl.col("canonical_merchant").cast(pl.Utf8).alias("pred_merchant"),
                     pl.col("category").cast(pl.Utf8).alias("pred_category"))
    g = er.eval_gold.select("txn_id", pl.col("canonical_merchant").alias("gold_merchant"),
                            pl.col("category").alias("gold_category"),
                            pl.col("txn_type").alias("gold_txn_type"),
                            "origin", "noise_abbrev", "noise_trunc")
    return (er.eval_feed.select("txn_id", "description").join(g, on="txn_id", how="left")
            .join(p, on="txn_id", how="left").join(extra, on="txn_id", how="left")
            .select("txn_id", "description", "gold_merchant", "gold_category", "gold_txn_type",
                    "pred_merchant", "pred_category", "origin", "noise_abbrev", "noise_trunc",
                    "candidates", *_JEV_FIELDS).sort("txn_id"))


def _ideal_preds(model_pdf: pl.DataFrame, er: EvalRows) -> pl.DataFrame:
    """Broadcast model-view predictions per descriptor; cache hits take the cache."""
    route_cols = ["pred_merchant", "pred_category", "candidates", *_JEV_FIELDS]
    rep = model_pdf.select(pl.col("txn_id").alias("rep_txn_id"), *route_cols)
    g = er.eval_gold.select("txn_id", pl.col("canonical_merchant").alias("gold_merchant"),
                            pl.col("category").alias("gold_category"),
                            pl.col("txn_type").alias("gold_txn_type"),
                            "origin", "noise_abbrev", "noise_trunc")
    hit = pl.col("cache_hit")
    return (er.eval_feed.select("txn_id", "description").join(g, on="txn_id", how="left")
            .join(er.cache, on="txn_id", how="left").join(rep, on="rep_txn_id", how="left")
            .with_columns(
                pl.when(hit).then(pl.col("cache_merchant")).otherwise(pl.col("pred_merchant"))
                .alias("pred_merchant"),
                pl.when(hit).then(pl.col("cache_category")).otherwise(pl.col("pred_category"))
                .alias("pred_category"))
            .select("txn_id", "description", "gold_merchant", "gold_category", "gold_txn_type",
                    "pred_merchant", "pred_category", "origin", "noise_abbrev", "noise_trunc",
                    "candidates", *_JEV_FIELDS, "cache_hit").sort("txn_id"))


def _score(name: str, route, er: EvalRows, pdf: pl.DataFrame, model_pdf: pl.DataFrame,
           timing: dict) -> dict:
    """Leaderboard row for one (split, view, route). Metrics a route cannot produce
    (cleaner: category, metadata: merchant) are shown as "-"."""
    has_cat = model_pdf["pred_category"].null_count() < len(model_pdf)
    has_m = model_pdf["pred_merchant"].null_count() < len(model_pdf)
    gold = er.eval_gold
    feed = er.eval_feed
    p = pdf.select("txn_id", pl.col("pred_merchant").alias("canonical_merchant"),
                   pl.col("pred_category").alias("category"))
    m_all = gold.filter(pl.col("canonical_merchant").is_not_null())
    # Spec D2: abbreviated local-merchant rows are out of the model-view headline
    # merchant score (the full name is not in the input); they stay in the subsets.
    excluded = pl.lit(False)
    if er.view == "model":
        excluded = (pl.col("origin") == "local") & pl.col("noise_abbrev").fill_null(False)
    m_gold = m_all.filter(~excluded)
    m_ids = m_gold["txn_id"]
    m_preds = p.filter(pl.col("txn_id").is_in(m_ids))
    m_feed = feed.filter(pl.col("txn_id").is_in(m_ids))
    dash = "-"
    r = {"route": name, "split": er.split, "view": er.view, "rows": len(gold),
         "merchant_rows": len(m_ids), "headline_excluded": len(m_all) - len(m_gold),
         "cache_hit_rate": er.stats["cache_hit_rate"], "avg_ms": timing["avg_ms"]}
    # Fill nulls so the CI helper never compares against a missing prediction.
    p_filled = p.with_columns(pl.col("canonical_merchant").fill_null(""),
                              pl.col("category").fill_null(""))
    if has_cat and len(gold):
        r["category_acc"] = metrics.category_accuracy(p, gold)
        r["macro_f1"] = metrics.category_macro_f1(p_filled, gold)
        r["category_ci"] = metrics.cluster_bootstrap_ci(p_filled, gold, seed=config.SEED)["category"]
    else:
        r["category_acc"] = r["macro_f1"] = r["category_ci"] = dash
    if has_m and len(m_ids):
        r["merchant_acc"] = metrics.merchant_exact_match(m_preds, m_gold)
        r["merchant_norm"] = metrics.merchant_normalized_match(m_preds, m_gold)
        r["merchant_norm_ci"] = metrics.cluster_bootstrap_ci(
            p_filled.filter(pl.col("txn_id").is_in(m_ids)), m_gold,
            seed=config.SEED)["merchant_norm"]
        n_desc = m_feed["description"].n_unique()
        r["dedup_ratio"] = metrics.dedup_ratio(m_preds, n_desc)
        r["gold_dedup_ratio"] = metrics.dedup_ratio(m_gold, n_desc)
        r["subsets"] = metrics.merchant_subsets(p, m_all)
        if er.view == "ideal_cache":
            all_ids = m_all["txn_id"]
            kpi = metrics.merchant_spend_kpi(feed.filter(pl.col("txn_id").is_in(all_ids)),
                                             p.filter(pl.col("txn_id").is_in(all_ids)), m_all)
            r["kpi_within_tol"] = kpi["within_tolerance"]
            r["kpi_mae"] = kpi["mae"]
    else:
        r["merchant_acc"] = r["merchant_norm"] = r["merchant_norm_ci"] = dash
        r["dedup_ratio"] = r["gold_dedup_ratio"] = dash
        if er.view == "ideal_cache":
            r["kpi_within_tol"] = r["kpi_mae"] = dash

    # No-match and retrieval recall are model-view facts over rows with a merchant.
    r["no_match_rate"] = r["retrieval_recall"] = dash
    if getattr(route, "_fallback", None) is not None:
        r["escalated"] = dash
    if er.view == "model" and len(m_all):
        m_all_ids = set(m_all["txn_id"].to_list())
        if name == "rules":
            r["no_match_rate"] = round(len(route.no_match_ids & m_all_ids) / len(m_all_ids), 4)
        elif name in ("jev_merchant", "jev_slm"):
            r["no_match_rate"] = round(len(route.none_ids & m_all_ids) / len(m_all_ids), 4)
            rows = pdf.filter(pl.col("txn_id").is_in(m_all_ids))
            r["retrieval_recall"] = round(metrics.retrieval_recall(
                rows["gold_merchant"].to_list(), rows["candidates"].to_list()), 4)
        if getattr(route, "_fallback", None) is not None:
            esc = getattr(route, "escalated_ids", set())
            r["escalated"] = round(len(esc & m_all_ids) / len(m_all_ids), 3)
    if getattr(route, "input_tokens", 0):
        r["input_tokens"] = route.input_tokens
    return r


_LEADERBOARD_NOTE = (
    "Model view: eval rows whose description is verbatim in train are removed, the rest "
    "deduplicated to one row per descriptor and capped (cache-miss accuracy; not "
    "comparable to earlier leaderboards). Ideal-cache view: cache hits take the train "
    "gold, the model view's descriptors take the route's prediction broadcast per "
    "descriptor (rules' per-row MCC category is approximated). Spend KPI: ideal-cache "
    "view only.")


def _synth_stats(feed: pl.DataFrame, gold: pl.DataFrame, mode: str) -> dict:
    """Merchant and brand-family counts by origin, and how many land in the model
    views after the cap (spec D1)."""
    m = gold.filter(pl.col("canonical_merchant").is_not_null())
    merchants = sorted(m["canonical_merchant"].unique().to_list())
    family = brand_families(merchants)

    def by_origin(g: pl.DataFrame) -> dict:
        out = {}
        for o in ("source", "local"):
            names = set(g.filter(pl.col("origin") == o)["canonical_merchant"].to_list())
            out[o] = {"merchants": len(names), "families": len({family[n] for n in names}),
                      "rows": g.filter(pl.col("origin") == o).height}
        return out

    views = {s: _build_views(s, feed, gold, config.EVAL_CAPS, config.SEED) for s in _SPLITTERS}
    return {
        "rows": len(feed), "mode": mode,
        "distinct_descriptions": feed["description"].n_unique(),
        "distinct_per_row": round(feed["description"].n_unique() / max(1, len(feed)), 4),
        "all": by_origin(m),
        **{f"{s}_model_view": {"model_view_n": v["model"].stats["model_view_n"],
                               "capped_n": v["model"].stats["capped_n"],
                               **by_origin(v["model"].eval_gold.filter(
                                   pl.col("canonical_merchant").is_not_null()))}
           for s, v in views.items()},
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    feed_path = config.DATA_DIR / "bank_feed.parquet"
    gold_path = config.DATA_DIR / "gold.parquet"

    if args.command == "synth":
        sample = load_sample(n=args.n)
        cov = strip_coverage(sample["transaction_description"].to_list())
        feed, gold = build_feed(sample, hard=args.hard)
        feed.write_parquet(feed_path)
        gold.write_parquet(gold_path)
        mode = "hard" if args.hard else "standard"
        print(f"wrote {feed_path} and {gold_path} ({len(feed)} rows, {mode} mode)")
        print(f"canonical strip coverage (clean descriptions): {cov:.3f}")
        stats = _synth_stats(feed, gold, mode)
        print(f"distinct descriptions / rows: {stats['distinct_per_row']}")
        for key in ("all", "random_model_view", "unseen_model_view"):
            print(key, {k: v for k, v in stats[key].items()})
        config.RUN_DIR.mkdir(parents=True, exist_ok=True)
        (config.RUN_DIR / "synth_stats.json").write_text(json.dumps(stats, indent=2))
        return

    feed = pl.read_parquet(feed_path)
    gold = pl.read_parquet(gold_path)

    if args.command == "finetune-export":
        from tranx.finetune.data import export
        tf, tg, _, _ = _split_unseen(feed, gold, config.SEED)
        if len(tf) > args.cap:
            tf = tf.sample(n=args.cap, seed=config.SEED)
            tg = tg.filter(pl.col("txn_id").is_in(set(tf["txn_id"].to_list())))
        categories = sorted(gold["category"].unique().to_list())
        out = export(tf, tg, args.out, categories=categories, seed=config.SEED)
        print(f"wrote {out}/train.jsonl and {out}/valid.jsonl from {len(tf)} rows")
        return

    if args.command == "run":
        # Same rows and scoring as eval's model view, for one route and one split;
        # nothing is written, so it cannot mix with a published run.
        caps = dict(config.EVAL_CAPS)
        if args.eval_cap > 0:
            caps[args.split] = args.eval_cap
        er = eval_rows(args.split, "model", feed, gold, caps=caps, seed=config.SEED)
        route = _make_route(args.route, slm_model=args.slm_model,
                            adapter_path=args.adapter_path, base_model=args.base_model)
        route.fit(er.train_feed, er.train_gold)
        preds, timing = run_route(route, er.eval_feed)
        pdf = _model_preds(route, er, preds)
        result = {k: v for k, v in _score(args.route, route, er, pdf, pdf, timing).items()
                  if k != "subsets"}
        if getattr(route, "confidences", None):
            result["mean_confidence"] = round(
                sum(route.confidences) / len(route.confidences), 3)
            result["input_tokens"] = route.input_tokens
        print(result)
        return

    if args.command == "eval":
        from tranx.eval.manifest import write_manifest, save_preds
        routes = args.routes.split(",")
        caps = dict(config.EVAL_CAPS)
        if args.cap_random is not None:
            caps["random"] = args.cap_random
        if args.cap_unseen is not None:
            caps["unseen"] = args.cap_unseen
        views = {s: _build_views(s, feed, gold, caps, config.SEED) for s in _SPLITTERS}
        for s, vs in views.items():
            print(f"{s}: model view {vs['model'].stats['view_n']} rows "
                  f"(of {vs['model'].stats['model_view_n']} descriptors), ideal-cache view "
                  f"{vs['ideal_cache'].stats['view_n']} rows, "
                  f"cache_hit_rate {vs['model'].stats['cache_hit_rate']}", flush=True)
            # A cap below the model view drops the other misses from the ideal-cache view too.
            st = vs["model"].stats
            if st["capped_n"] < st["model_view_n"]:
                print(f"WARNING {s}: cap keeps {st['capped_n']} of {st['model_view_n']} "
                      "model-view descriptors; the ideal-cache view omits the rest", flush=True)
        # The manifest pins data, splits and prompts before any prediction is saved.
        manifest = _eval_manifest(views, feed_path, gold_path, caps, config.SEED)
        write_manifest(manifest)
        mhash = manifest["manifest_hash"]
        config.RUN_DIR.mkdir(parents=True, exist_ok=True)
        (config.RUN_DIR / "eval_stats.json").write_text(json.dumps(
            {"manifest_hash": mhash,
             **{s: {v: vs[v].stats for v in VIEWS} for s, vs in views.items()}}, indent=2))
        results = []
        by = {}  # (split, view, route) -> preds parquet frame
        for split_name, vs in views.items():
            mv = vs["model"]
            for name in routes:
                route = _make_route(name)
                route.fit(mv.train_feed, mv.train_gold)
                preds, timing = run_route(route, mv.eval_feed)
                model_pdf = _model_preds(route, mv, preds)
                for view in VIEWS:
                    er = vs[view]
                    pdf = model_pdf if view == "model" else _ideal_preds(model_pdf, er)
                    save_preds(pdf, config.PREDS_DIR / f"{split_name}_{view}_{name}.parquet",
                               mhash)
                    result = _score(name, route, er, pdf, model_pdf, timing)
                    result["manifest_hash"] = mhash
                    results.append(result)
                    by[(split_name, view, name)] = pdf
                    print({k: v for k, v in result.items() if k != "subsets"}, flush=True)
        results.sort(key=lambda r: (r["split"] != "random", VIEWS.index(r["view"]),
                                    routes.index(r["route"]) if r["route"] in routes else 99))
        path = save_leaderboard(results, config.REPORTS_DIR, name=args.name,
                                preamble=_LEADERBOARD_NOTE)
        (config.REPORTS_DIR / (args.name.rsplit(".", 1)[0] + ".json")).write_text(
            json.dumps(results, indent=2))
        if args.also:
            also_name, also_routes = args.also.split("=")
            keep = set(also_routes.split(","))
            save_leaderboard([r for r in results if r["route"] in keep],
                             config.REPORTS_DIR, name=also_name, preamble=_LEADERBOARD_NOTE)

        def best(split):
            rows = [r for r in results if r["split"] == split and r["view"] == "model"
                    and isinstance(r["merchant_norm"], float)]
            return max(rows, key=lambda r: r["merchant_norm"])["route"] if rows else None

        print(f"wrote {path}; best model-view merchant_norm on random: {best('random')}, "
              f"on unseen: {best('unseen')}")
        if args.no_plot:
            return
        # Spend plot (resolution 13): jev_slm if scored, else the best model-view
        # merchant_norm route; drawn from the random split's ideal-cache view.
        plot_route = "jev_slm" if "jev_slm" in routes else best("random")
        if plot_route is None:
            return
        iv = views["random"]["ideal_cache"]
        pdf = by[("random", "ideal_cache", plot_route)]
        preds = pdf.select("txn_id", pl.col("pred_merchant").alias("canonical_merchant"))
        customer = iv.eval_feed["customer_id"].mode().sort().to_list()[0]
        plot_customer_spend(iv.eval_feed, preds, iv.eval_gold, customer, config.REPORTS_DIR)
        print(f"spend plot: route {plot_route}, customer {customer}")

if __name__ == "__main__":
    main()
