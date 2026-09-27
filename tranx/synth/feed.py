import math
import random
import polars as pl
from tranx import config
from tranx.synth.canonical import derive_canonical, gold_merchant
from tranx.synth.mcc import mcc_for
from tranx.synth.hard import hard_descriptor

# Category-conditioned payment-method priors. Keys are method names; the synth
# picks a method by sampling from the category's distribution.
_METHOD_PRIORS = {
    "Income": {"ach": 0.5, "transfer": 0.3, "check": 0.2},
    "Utilities & Services": {"ach": 0.6, "credit_card": 0.3, "debit_card": 0.1},
    "Financial Services": {"wire": 0.4, "transfer": 0.3, "ach": 0.3},
    "Food & Dining": {"credit_card": 0.5, "debit_card": 0.4, "cash_app": 0.1},
    "Shopping & Retail": {"credit_card": 0.6, "debit_card": 0.3, "cash_app": 0.1},
    "Healthcare & Medical": {"credit_card": 0.5, "debit_card": 0.3, "check": 0.2},
    "Entertainment & Recreation": {"credit_card": 0.6, "debit_card": 0.3, "cash_app": 0.1},
    "Charity & Donations": {"credit_card": 0.4, "ach": 0.3, "check": 0.3},
    "Transportation": {"credit_card": 0.5, "debit_card": 0.4, "cash_app": 0.1},
    "Government & Legal": {"ach": 0.4, "check": 0.3, "debit_card": 0.3},
}
# Median absolute amount per category (in currency units) for a lognormal draw.
_AMOUNT_MEDIAN = {
    "Income": 2500.0, "Food & Dining": 25.0, "Healthcare & Medical": 120.0,
    "Shopping & Retail": 60.0, "Utilities & Services": 90.0,
    "Entertainment & Recreation": 40.0, "Financial Services": 500.0,
    "Charity & Donations": 50.0, "Transportation": 35.0, "Government & Legal": 150.0,
}
_REFUND_RATE = 0.03


def _pick_method(category: str, rng: random.Random) -> str:
    priors = _METHOD_PRIORS.get(category, {m: 1.0 for m in config.PAYMENT_METHODS})
    methods = list(priors.keys())
    weights = list(priors.values())
    return rng.choices(methods, weights=weights, k=1)[0]


def _type_code(method: str, direction: str) -> str:
    """Synthetic bank type code: <DIR>-<METHOD>, e.g. 'O-CC'.

    There is deliberately no merchant/non-merchant flag: it was a pure function of
    the category (Income / Financial Services) and handed any route that read it
    the label for those two classes.
    """
    d = "I" if direction == "incoming" else "O"
    m = {
        "credit_card": "CC", "debit_card": "DC", "ach": "ACH", "wire": "WIR",
        "check": "CHK", "cash_app": "CSH", "transfer": "TRF",
    }[method]
    return f"{d}-{m}"


def _amount(category: str, rng: random.Random) -> float:
    median = _AMOUNT_MEDIAN.get(category, 50.0)
    mu = math.log(median)
    return round(math.exp(rng.gauss(mu, 0.6)), 2)


def build_feed(df: pl.DataFrame, seed: int = config.SEED,
               n_customers: int = config.N_CUSTOMERS,
               hard: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Synthesize a realistic bank feed + gold labels from the raw dataset.

    With hard=True the feed descriptions are dirtied into card-network-style
    descriptors; the gold canonical_merchant is still derived from the original
    clean description, so the labels stay reliable while the inputs get hard.

    Returns (feed_df, gold_df) keyed by txn_id. The feed carries no labels.
    """
    rng = random.Random(seed)
    # Separate stream so dirtying descriptors never shifts the label-affecting
    # draws (direction, amount, ...) — gold stays identical with or without hard.
    hrng = random.Random(seed + 1)
    feed_rows, gold_rows = [], []

    for i, row in enumerate(df.iter_rows(named=True)):
        desc = row["transaction_description"]
        category = row["category"]
        canonical = derive_canonical(desc)
        feed_desc = hard_descriptor(canonical, category, row["country"], hrng) if hard else desc

        direction = "incoming" if category == "Income" else "outgoing"
        if rng.random() < _REFUND_RATE and direction == "outgoing":
            direction = "incoming"

        is_merchant = category not in ("Income", "Financial Services")
        method = _pick_method(category, rng)
        amount = _amount(category, rng)
        signed = amount if direction == "incoming" else -amount

        mcc = None
        if is_merchant and rng.random() < config.MCC_COVERAGE:
            mcc_category = category
            if rng.random() < config.MCC_NOISE:
                mcc_category = rng.choice([c for c in config.CATEGORIES if c != category])
            mcc = mcc_for(mcc_category, rng)

        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        customer_id = f"C{rng.randint(0, n_customers - 1):05d}"
        txn_id = f"T{i:08d}"

        feed_rows.append({
            "txn_id": txn_id,
            "customer_id": customer_id,
            "description": feed_desc,
            "transaction_type_code": _type_code(method, direction),
            "mcc": mcc,
            "amount": signed,
            "payment_method": method,
            "posted_date": f"2026-{month:02d}-{day:02d}",
            "country": row["country"],
            "currency": row["currency"],
        })
        merchant, txn_type = gold_merchant(canonical, category)
        gold_rows.append({
            "txn_id": txn_id,
            "category": category,
            "canonical_merchant": merchant,
            "txn_type": txn_type,
            "direction": direction,
        })

    feed = pl.DataFrame(feed_rows, schema_overrides={"mcc": pl.Int64})
    gold = pl.DataFrame(gold_rows, schema_overrides={"canonical_merchant": pl.Utf8,
                                                     "txn_type": pl.Utf8})
    return feed, gold
