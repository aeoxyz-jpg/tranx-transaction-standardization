import re
import random
from tranx import config

_WS = re.compile(r"\s+")


def hard_descriptor(canonical: str, category: str, country: str,
                    rng: random.Random) -> str:
    """Synthesize a realistically dirty card-network descriptor from a clean
    canonical merchant name.

    Applies the noise real descriptors carry — punctuation loss, uppercasing,
    embedded store/auth numbers, city/state tokens, aggregator prefixes,
    truncation to legacy 22-char limits, and space collapsing — while keeping
    the merchant's leading characters so a capable parser can still recover it.
    """
    m = canonical
    if rng.random() < 0.5:
        m = re.sub(r"[',.&]", "", m)
    if rng.random() < 0.8:
        m = m.upper()

    parts = [m]
    if rng.random() < 0.5:
        parts.append(rng.choice(["#", "F", "S", ""]) + str(rng.randint(100, 99999)))
    if rng.random() < 0.5:
        parts.append(rng.choice(config.HARD_CITIES))
        if rng.random() < 0.6:
            parts.append(rng.choice(config.HARD_REGIONS))
    if rng.random() < 0.15:
        parts.append(f"{rng.randint(200, 999)}{rng.randint(1000000, 9999999)}")

    desc = " ".join(parts)
    # Truncate before the prefix is added, so the merchant always keeps its
    # leading characters (truncating after left some rows as a bare
    # "DEBIT CARD PURCHASE" with no merchant text at all).
    if rng.random() < 0.5:
        desc = desc[: rng.randint(20, 25)]
    if rng.random() < 0.45:
        desc = rng.choice(config.HARD_PREFIXES) + desc
    if rng.random() < 0.2:
        desc = _WS.sub("", desc)
    return desc.strip()
