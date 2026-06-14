import json
import polars as pl
from tranx.finetune.data import finetune_prompt, to_example, export


def test_finetune_prompt_has_descriptor_and_categories():
    p = finetune_prompt("SQ *MCDONALDS ATL", ["Food & Dining", "Income"])
    assert "SQ *MCDONALDS ATL" in p
    assert "Food & Dining" in p


def test_to_example_is_chat_with_json_completion():
    ex = to_example("SQ *MCDONALDS", "McDonald's", "Food & Dining", ["Food & Dining"])
    assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
    payload = json.loads(ex["messages"][1]["content"])
    assert payload == {"canonical_merchant": "McDonald's", "category": "Food & Dining"}


def test_export_writes_train_and_valid(tmp_path):
    feed = pl.DataFrame({
        "txn_id": [f"T{i}" for i in range(20)],
        "description": [f"SQ *SHOP{i}" for i in range(20)],
    })
    gold = pl.DataFrame({
        "txn_id": [f"T{i}" for i in range(20)],
        "canonical_merchant": [f"Shop{i}" for i in range(20)],
        "category": ["Shopping & Retail"] * 20,
    })
    out = export(feed, gold, tmp_path, valid_frac=0.2, seed=42)
    train_lines = (out / "train.jsonl").read_text().splitlines()
    valid_lines = (out / "valid.jsonl").read_text().splitlines()
    assert len(train_lines) == 16 and len(valid_lines) == 4
    # every line is valid chat JSON
    rec = json.loads(train_lines[0])
    assert "messages" in rec
