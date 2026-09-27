import argparse
import random
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

    p_run = sub.add_parser("run", help="run one route over the feed")
    p_run.add_argument("--route", required=True,
                       choices=["rules", "embedding", "slm_fewshot", "slm_lora", "jev", "jev_merchant", "jev_slm"])
    p_run.add_argument("--adapter-path", default="adapters/qwen-hard",
                       help="LoRA adapter dir for the slm_lora route")
    p_run.add_argument("--base-model", default=config.MLX_BASE,
                       help="MLX base model for the slm_lora route")
    p_run.add_argument("--split", choices=["random", "unseen"], default="random",
                       help="random 80/20 split, or merchant-disjoint (unseen merchants)")
    p_run.add_argument("--eval-cap", type=int, default=0,
                       help="cap eval rows (0 = no cap)")
    p_run.add_argument("--slm-model", default=config.SLM_MODEL,
                       help="Ollama model for the slm_fewshot route")

    p_eval = sub.add_parser("eval", help="run all routes on both splits, write leaderboard")
    p_eval.add_argument("--eval-cap", type=int, default=1000,
                        help="common eval-set cap so all routes are scored on the "
                             "same subset and the slow SLM route stays feasible")
    p_eval.add_argument("--routes", default="rules,embedding,slm_fewshot",
                        help="comma-separated routes to score")
    p_eval.add_argument("--name", default="leaderboard.md",
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


def _cap_eval(eval_feed: pl.DataFrame, eval_gold: pl.DataFrame, cap: int, seed: int):
    """Deterministically subsample the eval set so all routes score the same rows."""
    if cap <= 0 or len(eval_feed) <= cap:
        return eval_feed, eval_gold
    capped = eval_feed.sample(n=cap, seed=seed)
    keep = set(capped["txn_id"].to_list())
    return capped, eval_gold.filter(pl.col("txn_id").is_in(keep))


def _make_route(name: str, slm_model: str | None = None,
                adapter_path: str | None = None, base_model: str | None = None):
    if name == "rules":
        from tranx.routes.rules import RulesRoute
        return RulesRoute()
    if name == "embedding":
        from tranx.routes.embedding import EmbeddingRoute
        return EmbeddingRoute()
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


def _evaluate(route, train_feed, train_gold, eval_feed, eval_gold):
    route.fit(train_feed, train_gold)
    preds, timing = run_route(route, eval_feed)
    # Merchant metrics are scored only where the gold has a merchant; transaction-type
    # rows (salary, transfer, ...) count for category only.
    m_ids = eval_gold.filter(pl.col("canonical_merchant").is_not_null())["txn_id"]
    m_feed = eval_feed.filter(pl.col("txn_id").is_in(m_ids))
    m_preds = preds.filter(pl.col("txn_id").is_in(m_ids))
    m_gold = eval_gold.filter(pl.col("txn_id").is_in(m_ids))
    kpi = metrics.merchant_spend_kpi(m_feed, m_preds, m_gold)
    ci = metrics.cluster_bootstrap_ci(preds, eval_gold, seed=config.SEED)
    extra = {}
    if getattr(route, "_fallback", None) is not None:
        esc = getattr(route, "escalated_ids", set())
        extra["escalated"] = round(sum(t in esc for t in m_ids) / max(1, len(m_ids)), 3)
    return {
        "route": route.name,
        "category_acc": metrics.category_accuracy(preds, eval_gold),
        "macro_f1": metrics.category_macro_f1(preds, eval_gold),
        "merchant_acc": metrics.merchant_exact_match(m_preds, m_gold),
        "merchant_norm": metrics.merchant_normalized_match(m_preds, m_gold),
        "dedup_ratio": metrics.dedup_ratio(m_preds, m_feed["description"].n_unique()),
        "kpi_within_tol": kpi["within_tolerance"],
        "kpi_mae": kpi["mae"],
        "avg_ms": timing["avg_ms"],
        "merchant_rows": len(m_ids),
        "category_ci": ci["category"],
        "merchant_norm_ci": ci["merchant_norm"],
        "gold_dedup_ratio": metrics.dedup_ratio(m_gold, m_feed["description"].n_unique()),
        **extra,
    }, preds


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
        tf, tg, ef, eg = _SPLITTERS[args.split](feed, gold, config.SEED)
        ef, eg = _cap_eval(ef, eg, args.eval_cap, config.SEED)
        route = _make_route(args.route, slm_model=args.slm_model,
                            adapter_path=args.adapter_path, base_model=args.base_model)
        result, _ = _evaluate(route, tf, tg, ef, eg)
        if getattr(route, "confidences", None):
            result["mean_confidence"] = round(
                sum(route.confidences) / len(route.confidences), 3)
            result["input_tokens"] = route.input_tokens
        print(result)
        return

    if args.command == "eval":
        import json
        routes = args.routes.split(",")
        results = []
        by = {}  # (split, route) -> (preds, eval_feed, eval_gold)
        for split_name, splitter in _SPLITTERS.items():
            tf, tg, ef, eg = splitter(feed, gold, config.SEED)
            ef, eg = _cap_eval(ef, eg, args.eval_cap, config.SEED)
            for name in routes:
                route = _make_route(name)
                result, preds = _evaluate(route, tf, tg, ef, eg)
                result["split"] = split_name
                if getattr(route, "input_tokens", 0):
                    result["input_tokens"] = route.input_tokens
                results.append(result)
                by[(split_name, name)] = (preds, ef, eg)
                print(result, flush=True)
        results.sort(key=lambda r: (r["split"] != "random", routes.index(r["route"])
                                    if r["route"] in routes else 99))
        path = save_leaderboard(results, config.REPORTS_DIR, name=args.name)
        (config.REPORTS_DIR / (args.name.rsplit(".", 1)[0] + ".json")).write_text(
            json.dumps(results, indent=2))
        if args.also:
            also_name, also_routes = args.also.split("=")
            keep = set(also_routes.split(","))
            save_leaderboard([r for r in results if r["route"] in keep],
                             config.REPORTS_DIR, name=also_name)
        if args.no_plot:
            print(f"wrote {path}")
            return
        # Money shot from the random split (richest per-customer merchant mix).
        rand = [r for r in results if r["split"] == "random"]
        best = max(rand, key=lambda r: r["kpi_within_tol"])
        preds, ef, eg = by[("random", best["route"])]
        customer = ef["customer_id"].mode().to_list()[0]
        plot_customer_spend(ef, preds, eg, customer, config.REPORTS_DIR)
        unseen_best = max([r for r in results if r["split"] == "unseen"],
                          key=lambda r: r["kpi_within_tol"])
        print(f"wrote {path}; best on random: {best['route']}, "
              f"best on unseen: {unseen_best['route']}")

if __name__ == "__main__":
    main()
