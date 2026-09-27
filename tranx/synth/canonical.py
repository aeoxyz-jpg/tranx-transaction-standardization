import re
from tranx import config

_PAREN = re.compile(r"\([^)]*\)")
_IDS = re.compile(r"(#\d+|TXN\d+)", re.IGNORECASE)
_COUNTRY = re.compile(r"\b(" + "|".join(config.COUNTRIES) + r")\b", re.IGNORECASE)
_TIME = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in config.TIME_PHRASES) + r")\b",
    re.IGNORECASE,
)
_FORMAT = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in config.FORMAT_WORDS) + r")\b",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_TRAILING_HOSPITAL = re.compile(r"(?<=\S)\s+Hospital$", re.IGNORECASE)
_DANGLING = re.compile(r"(^[\s\-]+|[\s\-]+$)")


def derive_canonical(description: str) -> str:
    """Best-effort recovery of the canonical merchant from a noisy description.

    Strips parentheticals, store/txn ids, country tags, time-of-day phrases and
    format words, then collapses whitespace. Returns a silver-standard label.
    """
    s = description
    s = _PAREN.sub(" ", s)
    s = _IDS.sub(" ", s)
    # Protected names contain noise words; park them as placeholders while stripping.
    parked = {}
    for i, name in enumerate(config.PROTECTED_NAMES):
        token = f"\x00{i}\x00"
        s, n = re.subn(re.escape(name), token, s, flags=re.IGNORECASE)
        if n:
            parked[token] = name
    s = _TIME.sub(" ", s)
    s = _FORMAT.sub(" ", s)
    s = _COUNTRY.sub(" ", s)
    for token, name in parked.items():
        s = s.replace(token, name)
    # Drop leftover lone dashes used as separators
    s = s.replace(" - ", " ")
    s = _WS.sub(" ", s)
    s = _DANGLING.sub("", s)
    s = s.strip()
    if s not in config.HOSPITAL_MERCHANTS:
        s = _TRAILING_HOSPITAL.sub("", s)
    return s


def gold_merchant(canonical: str, category: str) -> tuple[str | None, str | None]:
    """Gold (merchant, non-merchant label) for a derived canonical name.

    Transaction types (salary, transfer, ...) and purchased items (MRI, Toll,
    Broadband, ...) have no merchant: returns (None, label). Otherwise applies label
    merges and category-based splits and returns (merchant, None).
    """
    if canonical in config.NON_MERCHANT_LABELS:
        return None, canonical
    split = config.CATEGORY_SPLIT_LABELS.get(canonical)
    if split and category in split:
        return split[category], None
    return config.LABEL_MERGES.get(canonical, canonical), None


def strip_coverage(descriptions: list[str]) -> float:
    """Fraction of descriptions whose canonical form has no residual noise tokens."""
    if not descriptions:
        return 0.0
    clean = 0
    for d in descriptions:
        c = derive_canonical(d)
        has_noise = bool(_IDS.search(c) or _PAREN.search(c) or _COUNTRY.search(c))
        if c and not has_noise:
            clean += 1
    return clean / len(descriptions)
