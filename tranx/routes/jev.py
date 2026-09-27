import os
import time
import polars as pl
from tranx.routes.base import Route
from tranx.routes.rules import RulesRoute
from tranx.schema import Txn, Standardized
from tranx.pipeline.clean import strip_processor_prefix, clean_description
from tranx.synth.canonical import derive_canonical
from tranx import config
from rapidfuzz import process, fuzz

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
NONE_OPTION = "none_of_these"
MERCHANT_CANDIDATES = 20
_RETRY_STATUSES = {429, 529}


def build_request(txn: Txn, categories: list[str], model: str,
                  merchant_candidates: list[str] | None = None) -> dict:
    """Choice question over the category list; optionally a second Choice over
    fuzzy-retrieved known merchants (Jev cannot emit free text, so unseen
    merchants can only resolve to NONE_OPTION). State is the structured txn."""
    # Description only, the same input the SLM and embedding routes get. The
    # synthetic type code / amount / payment method are category-conditioned by
    # construction and would hand Jev (only) a partial label oracle.
    state = {"description": strip_processor_prefix(txn.description)}
    questions = {
        "category": {
            "type": "choice",
            "instructions": "Which spending category does this bank transaction belong to?",
            "criteria": {c: None for c in categories},
        }
    }
    if merchant_candidates:
        criteria = {m: None for m in merchant_candidates}
        criteria[NONE_OPTION] = "None of the listed merchants is the one in the description"
        questions["merchant"] = {
            "type": "choice",
            "instructions": "Which known merchant does this transaction description refer to?",
            "criteria": criteria,
        }
    return {"state": state, "model": model, "questions": questions}


def parse(ans: dict, key: str) -> dict:
    """Pull choice/confidence/probabilities for one Choice question out of a raw
    Jev response. `key` is "merchant" or "category"."""
    a = ans.get("answers", {}).get(key, {})
    probs = a.get("probabilities", {}) or {}
    choice = a.get("choice")
    return {"choice": choice, "confidence": a.get("confidence"),
            "p_choice": probs.get(choice), "p_none": probs.get(NONE_OPTION)}


def _jev_call(payload: dict) -> dict:
    import requests
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(6):
        last = attempt == 5
        try:
            resp = requests.post(JEV_URL, json=payload, headers=headers, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            if last:
                raise
            time.sleep(0.5 * 2 ** attempt)
            continue
        # 429 rate limit, 529 overloaded, and transient 5xx (e.g. 520 from the CDN)
        if (resp.status_code in _RETRY_STATUSES or resp.status_code >= 500) and not last:
            time.sleep(0.5 * 2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


class JevRoute(Route):
    """Category via TypeSafe Jev (Choice question); merchant via the learned
    rules matcher, since Jev has no free-text output."""
    name = "jev"

    def __init__(self, call_fn=None, model: str = JEV_MODEL, merchant_choice: bool = False,
                 fallback: Route | None = None):
        self._call = call_fn or _jev_call
        self._model = model
        self._merchant_choice = merchant_choice or fallback is not None
        self._fallback = fallback  # merchant route used when Jev answers none_of_these
        if fallback is not None:
            self.name = "jev_slm"
        elif merchant_choice:
            self.name = "jev_merchant"
        self.escalated = 0
        self.escalated_ids: set[str] = set()
        # txn_ids where the merchant Choice returned none_of_these or anything not
        # in the candidate list (retrieval miss or a truly new merchant).
        self.none_ids: set[str] = set()
        # Per txn_id: candidates offered, merchant Choice fields, category confidence.
        self.row_info: dict[str, dict] = {}
        self._rules = RulesRoute()
        self._categories = config.CATEGORIES
        self._default_category = "Shopping & Retail"
        self.confidences: list[float] = []
        self.input_tokens = 0

    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        self._rules.fit(train_feed, train_gold)
        if self._fallback is not None:
            self._fallback.fit(train_feed, train_gold)
        cats = sorted(train_gold["category"].unique().to_list())
        if cats:
            self._categories = cats
            self._default_category = train_gold["category"].mode().to_list()[0]

    def _candidates(self, stripped: str) -> list[str]:
        keys = list(self._rules._canon_clean_to_name.keys())
        if not keys:
            return []
        query = clean_description(derive_canonical(stripped))
        hits = process.extract(query, keys, scorer=fuzz.token_set_ratio,
                               limit=MERCHANT_CANDIDATES)
        return [self._rules._canon_clean_to_name[k] for k, _, _ in hits]

    def standardize(self, txn: Txn) -> Standardized:
        direction = "incoming" if txn.amount > 0 else "outgoing"
        stripped = strip_processor_prefix(txn.description)
        candidates = self._candidates(stripped) if self._merchant_choice else None
        answer = self._call(build_request(txn, self._categories, self._model, candidates))
        answers = answer.get("answers", {})
        self.input_tokens += int(answer.get("usage", {}).get("input_tokens", 0))
        m_info = parse(answer, "merchant") if candidates else {
            "choice": None, "confidence": None, "p_choice": None, "p_none": None}
        if candidates:
            pick = m_info["choice"]
            if pick in candidates:
                merchant = pick
            else:
                self.none_ids.add(txn.txn_id)
                if self._fallback is not None:
                    self.escalated += 1
                    self.escalated_ids.add(txn.txn_id)
                    merchant = self._fallback.standardize(txn).canonical_merchant
                else:
                    merchant = derive_canonical(stripped)
        else:
            merchant = self._rules._match_merchant(txn.description)
        ans = answers.get("category", {})
        category = ans.get("choice")
        if category not in self._categories:
            category = self._default_category
        if "confidence" in ans:
            self.confidences.append(float(ans["confidence"]))
        self.row_info[txn.txn_id] = {
            "candidates": candidates if candidates else None,
            "jev_choice": m_info["choice"],
            "jev_confidence": m_info["confidence"],
            "jev_p_choice": m_info["p_choice"],
            "jev_p_none": m_info["p_none"],
            "jev_category_confidence": ans.get("confidence"),
        }
        return Standardized(canonical_merchant=merchant, category=category,
                            direction=direction)
