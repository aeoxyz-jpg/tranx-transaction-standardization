import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from tranx.routes.base import Route
from tranx.schema import Txn, Standardized
from tranx.synth.canonical import derive_canonical
from tranx.pipeline.clean import strip_processor_prefix
from tranx import config


_MODEL = None


def _default_encoder(texts: list[str]) -> np.ndarray:
    # Cache the model so it loads once, not once per transaction.
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(config.EMBED_MODEL)
    return _MODEL.encode(texts, normalize_embeddings=True)


class EmbeddingRoute(Route):
    """Embedding nearest-canonical for merchants + logistic regression for category."""
    name = "embedding"

    def __init__(self, encoder=None):
        self._encoder = encoder or _default_encoder
        self._canon_names: list[str] = []
        self._canon_vecs: np.ndarray | None = None
        self._clf: LogisticRegression | None = None

    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        joined = train_feed.join(train_gold, on="txn_id")
        self._canon_names = sorted(set(joined["canonical_merchant"].drop_nulls().to_list()))
        self._canon_vecs = self._encoder(self._canon_names)

        desc_vecs = self._encoder(joined["description"].to_list())
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(desc_vecs, joined["category"].to_list())

    def standardize(self, txn: Txn) -> Standardized:
        query = derive_canonical(strip_processor_prefix(txn.description))
        qvec = self._encoder([query])[0]
        sims = self._canon_vecs @ qvec
        merchant = self._canon_names[int(np.argmax(sims))]

        cat_vec = self._encoder([txn.description])
        category = self._clf.predict(cat_vec)[0]

        direction = "incoming" if txn.amount > 0 else "outgoing"
        return Standardized(canonical_merchant=merchant, category=category,
                            direction=direction)
