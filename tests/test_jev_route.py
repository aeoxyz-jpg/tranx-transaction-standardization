import polars as pl
from tranx.routes.jev import JevRoute, build_request, NONE_OPTION
from tranx.schema import Txn


def _txn(desc="SQ *CANES 47486", amount=-12.5, mcc=5814):
    return Txn(txn_id="t1", customer_id="c1", description=desc,
               transaction_type_code="POS", mcc=mcc, amount=amount,
               payment_method="credit_card", posted_date="2024-01-01",
               country="USA", currency="USD")


def test_build_request_shape():
    req = build_request(_txn(), ["Income", "Food & Dining"], "jev-latest")
    assert req["model"] == "jev-latest"
    # description only: the same input the SLM gets
    assert req["state"] == {"description": "CANES 47486"}
    q = req["questions"]["category"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"Income", "Food & Dining"}


def test_jev_route_uses_call_fn_and_falls_back_on_unknown_choice():
    seen = []

    def fake(payload):
        seen.append(payload)
        return {"answers": {"category": {"type": "choice", "choice": "Food & Dining",
                                         "confidence": 0.9}}}

    feed = pl.DataFrame({"txn_id": ["a"], "description": ["Cane's"], "mcc": [5814]})
    gold = pl.DataFrame({"txn_id": ["a"], "canonical_merchant": ["Cane's"],
                         "category": ["Food & Dining"], "direction": ["outgoing"]})
    r = JevRoute(call_fn=fake)
    r.fit(feed, gold)
    out = r.standardize(_txn(desc="Cane's #111"))
    assert out.category == "Food & Dining"
    assert out.direction == "outgoing"
    assert out.canonical_merchant == "Cane's"
    assert seen and set(seen[0]["questions"]["category"]["criteria"]) == {"Food & Dining"}
    assert r.confidences == [0.9]

    r2 = JevRoute(call_fn=lambda p: {"answers": {"category": {"choice": "Nonsense"}}})
    r2.fit(feed, gold)
    assert r2.standardize(_txn()).category == "Food & Dining"


def test_jev_merchant_choice_picks_candidate_or_falls_back():
    feed = pl.DataFrame({"txn_id": ["a", "b"], "description": ["Cane's #1", "BP #2"],
                         "mcc": [5814, 5541]})
    gold = pl.DataFrame({"txn_id": ["a", "b"], "canonical_merchant": ["Cane's", "BP"],
                         "category": ["Food & Dining", "Transportation"],
                         "direction": ["outgoing", "outgoing"]})
    seen = []

    def fake(payload):
        seen.append(payload)
        return {"answers": {"category": {"choice": "Food & Dining", "confidence": 0.5},
                            "merchant": {"choice": "BP"}},
                "usage": {"input_tokens": 42}}

    r = JevRoute(call_fn=fake, merchant_choice=True)
    r.fit(feed, gold)
    assert r.name == "jev_merchant"
    out = r.standardize(_txn(desc="SQ *CANES 47486"))
    crit = seen[0]["questions"]["merchant"]["criteria"]
    assert NONE_OPTION in crit and "Cane's" in crit and "BP" in crit
    assert out.canonical_merchant == "BP"
    assert r.input_tokens == 42

    r2 = JevRoute(call_fn=lambda p: {"answers": {"merchant": {"choice": NONE_OPTION},
                                                 "category": {"choice": "Income"}}},
                  merchant_choice=True)
    r2.fit(feed, gold)
    out2 = r2.standardize(_txn(desc="Unknown Shop #9"))
    assert out2.canonical_merchant not in ("Cane's", "BP")


def test_jev_slm_escalates_only_on_none():
    from tranx.schema import Standardized

    class FakeSlm:
        fitted = False
        def fit(self, f, g): FakeSlm.fitted = True
        def standardize(self, txn):
            return Standardized(canonical_merchant="FromSLM", category="Income", direction="outgoing")

    feed = pl.DataFrame({"txn_id": ["a"], "description": ["Cane's #1"], "mcc": [5814]})
    gold = pl.DataFrame({"txn_id": ["a"], "canonical_merchant": ["Cane's"],
                         "category": ["Food & Dining"], "direction": ["outgoing"]})
    answers = iter([NONE_OPTION, "Cane's"])
    r = JevRoute(call_fn=lambda p: {"answers": {"merchant": {"choice": next(answers)},
                                                "category": {"choice": "Food & Dining"}}},
                 fallback=FakeSlm())
    r.fit(feed, gold)
    assert r.name == "jev_slm" and FakeSlm.fitted
    out1 = r.standardize(_txn(desc="Unknown Shop #9"))
    assert out1.canonical_merchant == "FromSLM" and out1.category == "Food & Dining"
    assert r.standardize(_txn(desc="Cane's #2")).canonical_merchant == "Cane's"
    assert r.escalated == 1


def test_jev_call_retries_transient_5xx(monkeypatch):
    import requests
    from tranx.routes import jev

    class R:
        def __init__(self, code): self.status_code = code
        def raise_for_status(self):
            if self.status_code >= 400: raise requests.HTTPError(str(self.status_code))
        def json(self): return {"ok": True}

    codes = iter([520, 429, 200])
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(next(codes)))
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    assert jev._jev_call({}) == {"ok": True}
