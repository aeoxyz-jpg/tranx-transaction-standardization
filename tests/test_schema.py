from dataclasses import fields
from tranx.schema import Txn, Gold, Standardized


def test_txn_has_no_label_fields():
    # Leakage guard: the pipeline input must not carry the answers.
    names = {f.name for f in fields(Txn)}
    assert "category" not in names
    assert "canonical_merchant" not in names
    assert {"txn_id", "customer_id", "description", "amount"} <= names


def test_standardized_fields():
    s = Standardized(canonical_merchant="McDonald's", category="Food & Dining", direction="outgoing")
    assert s.canonical_merchant == "McDonald's"
    assert s.direction == "outgoing"


def test_gold_fields():
    g = Gold(txn_id="t1", category="Income", canonical_merchant="Salary", direction="incoming")
    assert g.txn_id == "t1"
