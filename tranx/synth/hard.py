import re
import random
from tranx import config

_WS = re.compile(r"\s+")
_VOWELS = re.compile(r"[aeiouAEIOU]")


def _abbreviate(m: str) -> str:
    # Words longer than 4 chars keep their first letter and drop the vowels after it
    # (BLUE HERON BAKERY -> BLUE HRN BKRY), like legacy fixed-width merchant fields.
    return " ".join(w[0] + _VOWELS.sub("", w[1:]) if len(w) > 4 else w
                    for w in m.split(" "))


def hard_descriptor(canonical: str, category: str, country: str,
                    rng: random.Random) -> str:
    return hard_descriptor_flags(canonical, category, country, rng)[0]


def hard_descriptor_flags(canonical: str, category: str, country: str,
                          rng: random.Random) -> tuple[str, bool, bool]:
    """Synthesize a realistically dirty card-network descriptor from a clean
    canonical merchant name.

    Applies the noise real descriptors carry — punctuation loss, uppercasing,
    embedded store/auth numbers, city/state tokens, aggregator prefixes,
    truncation to legacy 22-char limits, and space collapsing — while keeping
    the merchant's leading characters so a capable parser can still recover it.
    With probability ABBREV_P long words first lose the vowels after their first letter.

    Returns (descriptor, noise_abbrev, noise_trunc): whether abbreviation changed the
    name, and whether truncation cut into the name (not just trailing ids/city).
    """
    m = canonical
    if rng.random() < 0.5:
        m = re.sub(r"[',.&]", "", m)
    if rng.random() < 0.8:
        m = m.upper()
    abbrev = False
    if rng.random() < config.ABBREV_P:
        short = _abbreviate(m)
        abbrev = short != m
        m = short

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
    trunc = False
    if rng.random() < 0.5:
        cut = rng.randint(20, 25)
        trunc = cut < len(m)
        desc = desc[:cut]
    if rng.random() < 0.45:
        desc = rng.choice(config.HARD_PREFIXES) + desc
    if rng.random() < 0.2:
        desc = _WS.sub("", desc)
    return desc.strip(), abbrev, trunc
