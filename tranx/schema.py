from dataclasses import dataclass


@dataclass(frozen=True)
class Txn:
    """One row of the synthetic bank feed — the pipeline input. Carries no labels."""
    txn_id: str
    customer_id: str
    description: str
    transaction_type_code: str
    mcc: int | None
    amount: float
    payment_method: str
    posted_date: str
    country: str
    currency: str


@dataclass(frozen=True)
class Gold:
    """Ground-truth labels for one transaction."""
    txn_id: str
    category: str
    canonical_merchant: str
    direction: str  # "incoming" | "outgoing"


@dataclass(frozen=True)
class Standardized:
    """A route's prediction for one transaction."""
    canonical_merchant: str
    category: str
    direction: str
