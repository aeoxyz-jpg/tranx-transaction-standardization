import json
import polars as pl
from tranx.routes.slm_fewshot import SlmFewshotRoute, build_prompt, parse_response
from tranx.schema import Txn


def test_build_prompt_includes_description_and_categories():
    p = build_prompt("McDonald's #111", ["Food & Dining", "Income"])
    assert "McDonald's #111" in p
    assert "Food & Dining" in p


def test_route_honors_prompt_fn_strip_and_name():
    seen = {}

    def fake_chat(prompt):
        seen["prompt"] = prompt
        return json.dumps({"canonical_merchant": "X", "category": "Income"})

    def my_prompt(descriptor, categories):
        return f"PROMPT::{descriptor}"

    feed = pl.DataFrame({"txn_id": ["T1"], "description": ["AMZN MKTP FOO"],
                         "category": ["Income"], "canonical_merchant": ["Foo"],
                         "direction": ["outgoing"]})
    r = SlmFewshotRoute(chat_fn=fake_chat, prompt_fn=my_prompt,
                        strip_prefix=False, name="slm_lora")
    r.fit(feed, feed)
    txn = Txn("T1", "C1", "AMZN MKTP FOO", "O-CC-M", None, -5.0,
              "credit_card", "2026-01-01", "USA", "USD")
    r.standardize(txn)
    assert r.name == "slm_lora"
    # strip_prefix=False -> descriptor passed through untouched to the custom prompt
    assert seen["prompt"] == "PROMPT::AMZN MKTP FOO"


def test_parse_response_extracts_json():
    raw = 'Sure!\n{"canonical_merchant": "McDonald\'s", "category": "Food & Dining"}\nDone'
    out = parse_response(raw)
    assert out["canonical_merchant"] == "McDonald's"
    assert out["category"] == "Food & Dining"


def test_parse_response_bad_json_returns_empty():
    assert parse_response("no json here") == {}


def test_slm_route_uses_chat_fn():
    def fake_chat(prompt: str) -> str:
        return json.dumps({"canonical_merchant": "BP", "category": "Transportation"})

    feed = pl.DataFrame({"txn_id": ["T1"], "description": ["BP #1"],
                         "category": ["Transportation"], "canonical_merchant": ["BP"],
                         "direction": ["outgoing"]})
    r = SlmFewshotRoute(chat_fn=fake_chat)
    r.fit(feed, feed)
    txn = Txn("T1", "C1", "BP @ Pleasant Hills", "O-DC-M", None, -30.0,
              "debit_card", "2026-01-01", "USA", "USD")
    out = r.standardize(txn)
    assert out.canonical_merchant == "BP"
    assert out.category == "Transportation"
    assert out.direction == "outgoing"
