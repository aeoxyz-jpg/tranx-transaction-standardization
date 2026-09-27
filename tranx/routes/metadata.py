import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from tranx.routes.base import Route
from tranx.schema import Txn, Standardized


class MetadataRoute(Route):
    """Category from transaction metadata only — never the description.
    Reimplementation of the ~0.68 metadata-only baseline: logistic regression on
    one-hot(transaction_type_code, mcc or "none") + log1p(|amount|). Merchant is
    unknowable from metadata alone, so it is always None."""
    name = "metadata"

    def __init__(self):
        self._encoder: OneHotEncoder | None = None
        self._clf: LogisticRegression | None = None
        self._default_category = "Shopping & Retail"

    def _cat_input(self, feed: pl.DataFrame) -> list[tuple[str, str]]:
        mccs = ["none" if m is None else str(m) for m in feed["mcc"].to_list()]
        return list(zip(feed["transaction_type_code"].to_list(), mccs))

    def _features(self, feed: pl.DataFrame) -> np.ndarray:
        cat_feats = self._encoder.transform(self._cat_input(feed)).toarray()
        amounts = np.log1p(np.abs(feed["amount"].to_numpy()))[:, None]
        return np.hstack([cat_feats, amounts])

    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        joined = train_feed.join(train_gold, on="txn_id")
        self._encoder = OneHotEncoder(handle_unknown="ignore")
        self._encoder.fit(self._cat_input(joined))
        X = self._features(joined)
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X, joined["category"].to_list())
        cats = train_gold["category"].mode().to_list()
        if cats:
            self._default_category = cats[0]

    def standardize(self, txn: Txn) -> Standardized:
        row = pl.DataFrame({"transaction_type_code": [txn.transaction_type_code],
                            "mcc": [txn.mcc], "amount": [txn.amount]})
        X = self._features(row)
        category = self._clf.predict(X)[0]
        direction = "incoming" if txn.amount > 0 else "outgoing"
        return Standardized(canonical_merchant=None, category=category, direction=direction)
