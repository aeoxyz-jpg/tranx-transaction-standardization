import hashlib
import math
import random
import polars as pl
from tranx import config
from tranx.synth.canonical import derive_canonical, gold_merchant
from tranx.synth.families import _in_eval
from tranx.synth.local import local_merchants
from tranx.synth.mcc import mcc_for
from tranx.synth.hard import hard_descriptor_flags

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
# Zipf exponent for picking a local merchant within its category (a few busy locals,
# a long tail of rarely seen ones).
_LOCAL_ZIPF_S = 1.0


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


def _zipf_weights(n: int, s: float) -> list[float]:
    return [1.0 / (k + 1) ** s for k in range(n)]


def location_descriptor(name: str, loc: int, text: str, category: str,
                        country: str) -> tuple[str, bool, bool]:
    # Seeded by sha1, not hash(): the same (name, location) must give the same
    # descriptor in every process, so repeats are byte-identical.
    rng = random.Random(int(hashlib.sha1(f"{name}|{loc}".encode()).hexdigest(), 16))
    return hard_descriptor_flags(text, category, country, rng)


def _assign_origin(rows: list[dict], seed: int) -> list[tuple[str, str, str | None]]:
    """Per row (gold label, descriptor text, origin). A stable-hash share of merchant
    rows moves to a same-category local merchant; a label's first row never moves,
    so no source label disappears from the gold."""
    base = []
    for row in rows:
        canonical = derive_canonical(row["transaction_description"])
        merchant, txn_type = gold_merchant(canonical, row["category"])
        base.append((merchant, txn_type, canonical))
    source_labels = sorted({t if m is None else m for m, t, _ in base})
    by_cat: dict[str, list[str]] = {}
    for name, cat in local_merchants(source_labels, seed, config.LOCAL_MERCHANTS_N):
        by_cat.setdefault(cat, []).append(name)
    weights = {c: _zipf_weights(len(v), _LOCAL_ZIPF_S) for c, v in by_cat.items()}
    first = {}
    for i, (merchant, _, _) in enumerate(base):
        if merchant is not None:
            first.setdefault(merchant, i)
    # Own stream: local picks never shift the label draws or the descriptor noise.
    lrng = random.Random(seed + 2)
    out = []
    for i, ((merchant, txn_type, canonical), row) in enumerate(zip(base, rows)):
        cat = row["category"]
        if merchant is None:
            out.append((txn_type, canonical, None))
        elif (cat in by_cat and first[merchant] != i
              and _in_eval(f"local|T{i:08d}", config.LOCAL_MERCHANT_SHARE)):
            name = lrng.choices(by_cat[cat], weights=weights[cat], k=1)[0]
            out.append((name, name, "local"))
        else:
            out.append((merchant, canonical, "source"))
    return out


def _hard_descriptors(rows: list[dict], origin: list[tuple], seed: int) -> list[tuple]:
    """Per row (descriptor, noise_abbrev, noise_trunc). A name gets
    L = min(LOCATIONS_CAP, ceil(rows / LOCATIONS_ROWS_PER)) locations, each row draws
    one Zipf-weighted, and each (name, location) has one fixed descriptor, so busy
    names repeat descriptors and the tail stays one-off."""
    counts: dict[str, int] = {}
    for name, _, _ in origin:
        counts[name] = counts.get(name, 0) + 1
    locrng = random.Random(seed + 1)
    wcache: dict[int, list[float]] = {}
    cache: dict[tuple[str, int], tuple] = {}
    out = []
    for row, (name, text, _) in zip(rows, origin):
        n_loc = min(config.LOCATIONS_CAP, math.ceil(counts[name] / config.LOCATIONS_ROWS_PER))
        if n_loc not in wcache:
            wcache[n_loc] = _zipf_weights(n_loc, config.LOCATIONS_ZIPF_S)
        loc = locrng.choices(range(n_loc), weights=wcache[n_loc], k=1)[0]
        if (name, loc) not in cache:
            cache[(name, loc)] = location_descriptor(name, loc, text, row["category"],
                                                     row["country"])
        out.append(cache[(name, loc)])
    return out


def build_feed(df: pl.DataFrame, seed: int = config.SEED,
               n_customers: int = config.N_CUSTOMERS,
               hard: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Synthesize a realistic bank feed + gold labels from the raw dataset.

    With hard=True the feed descriptions are dirtied into card-network-style
    descriptors; the gold canonical_merchant is still derived from the original
    clean description, so the labels stay reliable while the inputs get hard.
    A share of merchant rows is reassigned to fictional local merchants in both
    modes; without hard, a local row's description is the local name.

    Returns (feed_df, gold_df) keyed by txn_id. The feed carries no labels.
    """
    rng = random.Random(seed)
    rows = list(df.iter_rows(named=True))
    origin = _assign_origin(rows, seed)
    # Descriptor noise uses its own streams, so the label-affecting draws
    # (direction, amount, ...) and the gold are identical with or without hard.
    noise = _hard_descriptors(rows, origin, seed) if hard else None
    feed_rows, gold_rows = [], []

    for i, row in enumerate(rows):
        desc = row["transaction_description"]
        category = row["category"]
        name, _, org = origin[i]
        if hard:
            feed_desc, abbrev, trunc = noise[i]
        else:
            feed_desc, abbrev, trunc = (name if org == "local" else desc), False, False

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
        gold_rows.append({
            "txn_id": txn_id,
            "category": category,
            "canonical_merchant": name if org else None,
            "txn_type": None if org else name,
            "direction": direction,
            "origin": org,
            "noise_abbrev": abbrev,
            "noise_trunc": trunc,
        })

    feed = pl.DataFrame(feed_rows, schema_overrides={"mcc": pl.Int64})
    gold = pl.DataFrame(gold_rows, schema_overrides={"canonical_merchant": pl.Utf8,
                                                     "txn_type": pl.Utf8,
                                                     "origin": pl.Utf8,
                                                     "noise_abbrev": pl.Boolean,
                                                     "noise_trunc": pl.Boolean})
    gold = gold.select(config.GOLD_COLUMNS)
    return feed, gold
