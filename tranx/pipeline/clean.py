import re

_IDS = re.compile(r"(#\d+|TXN\d+)", re.IGNORECASE)
_WS = re.compile(r"\s+")

# Known payment-processor / aggregator prefixes; the actual merchant follows them.
# Star-delimited wrappers carry their own separator, so they may be fused to the
# merchant ("PP*DOUGHNOTTS"). Word prefixes must be followed by a non-alphanumeric
# boundary, otherwise real names get truncated ("ACHILLES" -> "ILLES").
# "AMZN MKTP" is deliberately absent: on real statements it is Amazon's own
# descriptor ("AMZNMktplace", "AMZN Mktp UK*MI5TU"), not a wrapper around another
# merchant; stripping it turned 390 real Amazon rows into "lace" / "UK*...".
_PROCESSOR = re.compile(
    r"^\s*(?:(?:SQ|TST|PP|PAYPAL|SP)\s*\*"
    r"|(?:POS\s*DEBIT|DEBIT\s*CARD\s*PURCHASE|PURCHASE|ACH)(?![A-Za-z0-9]))\s*",
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
    is exposed (e.g. 'SQ *BLUE BOTTLE' -> 'BLUE BOTTLE')."""
    return _PROCESSOR.sub("", description, count=1).strip()
