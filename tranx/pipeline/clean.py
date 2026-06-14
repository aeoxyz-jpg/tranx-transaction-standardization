import re

_IDS = re.compile(r"(#\d+|TXN\d+)", re.IGNORECASE)
_WS = re.compile(r"\s+")

# Known payment-processor / aggregator prefixes (real-world + synth hard mode).
# A real enrichment pipeline maintains this list; the actual merchant follows it.
# Allow optional internal spaces so space-collapsed descriptors are still caught.
_PROCESSOR = re.compile(
    r"^\s*(SQ\s*\*|TST\s*\*|PP\s*\*|PAYPAL\s*\*|SP\s*\*|"
    r"POS\s*DEBIT|DEBIT\s*CARD\s*PURCHASE|PURCHASE|AMZN\s*MKTP|ACH)\s*",
    re.IGNORECASE,
)


def clean_description(description: str) -> str:
    """Normalize a raw description: drop ids, lowercase, collapse whitespace."""
    s = _IDS.sub(" ", description)
    s = s.lower()
    s = _WS.sub(" ", s)
    return s.strip()


def strip_processor_prefix(description: str) -> str:
    """Remove a leading payment-processor/aggregator prefix so the real merchant
    is exposed (e.g. 'AMZN MKTP HOME DEPOT' -> 'HOME DEPOT')."""
    return _PROCESSOR.sub("", description, count=1).strip()
