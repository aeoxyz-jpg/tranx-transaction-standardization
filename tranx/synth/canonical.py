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
    s = _TIME.sub(" ", s)
    s = _FORMAT.sub(" ", s)
    s = _COUNTRY.sub(" ", s)
    # Drop leftover lone dashes used as separators
    s = s.replace(" - ", " ")
    s = _WS.sub(" ", s)
    s = _DANGLING.sub("", s)
    s = s.strip()
    if s not in config.HOSPITAL_MERCHANTS:
        s = _TRAILING_HOSPITAL.sub("", s)
    return s


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
