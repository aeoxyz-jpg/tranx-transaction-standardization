import json
import re
import polars as pl
from tranx.routes.base import Route
from tranx.schema import Txn, Standardized
from tranx import config

_JSON = re.compile(r"\{.*\}", re.DOTALL)

_FEWSHOT = (
    'Examples:\n'
    'Description: "McDonald\'s #111" -> {"canonical_merchant": "McDonald\'s", "category": "Food & Dining"}\n'
    'Description: "BP on Buford Hwy" -> {"canonical_merchant": "BP", "category": "Transportation"}\n'
    'Description: "Salary - Rush Hour" -> {"canonical_merchant": "Salary", "category": "Income"}\n'
    'Description: "HOME DEPOT #4521 ATLANTA GA" -> {"canonical_merchant": "Home Depot", "category": "Shopping & Retail"}\n'
    'Description: "SQ *CANES 47486" -> {"canonical_merchant": "Raising Cane\'s", "category": "Food & Dining"}\n'
)


def build_prompt(description: str, categories: list[str]) -> str:
    cats = ", ".join(categories)
    return (
        "You standardize bank transactions. Strip store numbers, locations and noise "
        "to get the canonical merchant, and choose exactly one category.\n"
        "Ignore payment-processor prefixes (SQ*, TST*, PP*, PAYPAL*, SP*, "
        "POS DEBIT) and return the real merchant that follows them, using its proper "
        "brand spelling and capitalization.\n"
        f"Allowed categories: {cats}\n"
        f"{_FEWSHOT}"
        f'Now do this one. Description: "{description}"\n'
        'Reply with ONLY a JSON object with keys "canonical_merchant" and "category".'
    )


def parse_response(raw: str) -> dict:
    m = _JSON.search(raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _ollama_chat(prompt: str, model: str) -> str:
    import requests
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


class SlmFewshotRoute(Route):
    """Local small-LLM few-shot route via Ollama (or an injected chat_fn)."""
    name = "slm_fewshot"

    def __init__(self, chat_fn=None, model=None, prompt_fn=None,
                 strip_prefix=True, name=None):
        self._model = model or config.SLM_MODEL
        self._chat = chat_fn or (lambda p: _ollama_chat(p, self._model))
        self._prompt_fn = prompt_fn or build_prompt
        self._strip = strip_prefix
        if name:
            self.name = name
        elif self._model == config.SLM_MODEL:
            self.name = "slm_fewshot"
        else:
            self.name = f"slm_fewshot:{self._model}"
        self._categories = config.CATEGORIES
        self._default_category = "Shopping & Retail"

    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        cats = sorted(train_gold["category"].unique().to_list())
        if cats:
            self._categories = cats
            self._default_category = train_gold["category"].mode().to_list()[0]

    def standardize(self, txn: Txn) -> Standardized:
        from tranx.synth.canonical import derive_canonical
        from tranx.pipeline.clean import strip_processor_prefix
        cleaned = strip_processor_prefix(txn.description) if self._strip else txn.description
        prompt = self._prompt_fn(cleaned, self._categories)
        parsed = parse_response(self._chat(prompt))
        merchant = parsed.get("canonical_merchant") or derive_canonical(cleaned)
        category = parsed.get("category")
        if category not in self._categories:
            category = self._default_category
        direction = "incoming" if txn.amount > 0 else "outgoing"
        return Standardized(canonical_merchant=merchant, category=category,
                            direction=direction)
