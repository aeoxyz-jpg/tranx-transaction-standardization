from abc import ABC, abstractmethod
import polars as pl
from tranx.schema import Txn, Standardized


class Route(ABC):
    """A pluggable standardization approach. Subclasses set `name`."""
    name: str = "base"

    @abstractmethod
    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        """Learn any state needed (canonical lists, classifiers) from labeled train data."""

    @abstractmethod
    def standardize(self, txn: Txn) -> Standardized:
        """Predict canonical merchant, category and direction for one transaction."""
