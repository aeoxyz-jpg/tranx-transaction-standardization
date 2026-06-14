import json
import random
from pathlib import Path
import polars as pl
from tranx import config


def finetune_prompt(descriptor: str, categories: list[str]) -> str:
    """Instruction prompt used identically at train and inference time."""
    cats = ", ".join(categories)
    return (
        "Standardize this bank transaction descriptor. Strip processor prefixes, "
        "store numbers, locations and noise to recover the real merchant in its "
        "proper brand spelling, and choose exactly one category.\n"
        f"Allowed categories: {cats}\n"
        f'Descriptor: "{descriptor}"\n'
        'Reply with ONLY a JSON object with keys "canonical_merchant" and "category".'
    )


def to_example(descriptor: str, canonical: str, category: str,
               categories: list[str]) -> dict:
    """One MLX chat-format training record (user prompt + JSON assistant answer)."""
    completion = json.dumps({"canonical_merchant": canonical, "category": category})
    return {"messages": [
        {"role": "user", "content": finetune_prompt(descriptor, categories)},
        {"role": "assistant", "content": completion},
    ]}


def build_examples(feed: pl.DataFrame, gold: pl.DataFrame,
                   categories: list[str]) -> list[dict]:
    j = feed.select(["txn_id", "description"]).join(
        gold.select(["txn_id", "canonical_merchant", "category"]), on="txn_id")
    return [to_example(r["description"], r["canonical_merchant"], r["category"], categories)
            for r in j.iter_rows(named=True)]


def _write(path: Path, examples: list[dict]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in examples))


def export(feed: pl.DataFrame, gold: pl.DataFrame, out_dir, categories=None,
           valid_frac: float = 0.1, seed: int = config.SEED) -> Path:
    """Write train.jsonl / valid.jsonl in MLX chat format to out_dir."""
    categories = categories or sorted(gold["category"].unique().to_list())
    examples = build_examples(feed, gold, categories)
    rng = random.Random(seed)
    rng.shuffle(examples)
    cut = int((1 - valid_frac) * len(examples))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "train.jsonl", examples[:cut])
    _write(out / "valid.jsonl", examples[cut:])
    return out
