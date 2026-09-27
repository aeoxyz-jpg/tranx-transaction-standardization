import polars as pl
from tranx.routes.base import Route
from tranx.schema import Txn, Standardized
from tranx.pipeline.clean import strip_processor_prefix
from tranx.synth.canonical import derive_canonical


class CleanerRoute(Route):
    """String-cleaning baseline, no learning: derive_canonical over the stripped
    description. Every other route's lift is measured against this floor."""
    name = "cleaner"

    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        pass

    def standardize(self, txn: Txn) -> Standardized:
        merchant = derive_canonical(strip_processor_prefix(txn.description))
        direction = "incoming" if txn.amount > 0 else "outgoing"
        return Standardized(canonical_merchant=merchant, category=None, direction=direction)
