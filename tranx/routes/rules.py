import polars as pl
from rapidfuzz import process, fuzz
from tranx.routes.base import Route
from tranx.schema import Txn, Standardized
from tranx.pipeline.clean import clean_description, strip_processor_prefix
from tranx.synth.canonical import derive_canonical


def _majority(counts: dict[str, int]) -> str:
    return max(counts, key=counts.get)


class RulesRoute(Route):
    """Baseline: fuzzy-match to a learned canonical list + learned MCC/merchant category.

    Category is predicted from priors *learned from training data* — a
    mcc -> majority-category map and a merchant -> majority-category map — never
    from the synthetic generative MCC table. Because real (and our synthetic)
    MCCs are noisy and overlap categories, the MCC prior is strong but imperfect.
    """
    name = "rules"

    def __init__(self, score_cutoff: int = 85):
        self.score_cutoff = score_cutoff
        self._canon_clean_to_name: dict[str, str] = {}
        self._merchant_to_category: dict[str, str] = {}
        self._mcc_to_category: dict[int, str] = {}
        self._default_category = "Shopping & Retail"

    def fit(self, train_feed: pl.DataFrame, train_gold: pl.DataFrame) -> None:
        joined = train_feed.join(train_gold, on="txn_id")
        merchant_counts: dict[str, dict[str, int]] = {}
        mcc_counts: dict[int, dict[str, int]] = {}
        for row in joined.iter_rows(named=True):
            canon = row["canonical_merchant"]
            category = row["category"]
            self._canon_clean_to_name[clean_description(canon)] = canon
            merchant_counts.setdefault(canon, {})
            merchant_counts[canon][category] = merchant_counts[canon].get(category, 0) + 1
            mcc = row["mcc"]
            if mcc is not None:
                mcc_counts.setdefault(mcc, {})
                mcc_counts[mcc][category] = mcc_counts[mcc].get(category, 0) + 1
        self._merchant_to_category = {m: _majority(c) for m, c in merchant_counts.items()}
        self._mcc_to_category = {m: _majority(c) for m, c in mcc_counts.items()}
        all_cats = train_gold["category"].value_counts(sort=True)
        if len(all_cats):
            self._default_category = all_cats.row(0)[0]

    def _match_merchant(self, description: str) -> str:
        stripped = strip_processor_prefix(description)
        query = clean_description(derive_canonical(stripped))
        keys = list(self._canon_clean_to_name.keys())
        if not keys:
            return derive_canonical(stripped)
        # token_set_ratio tolerates embedded city/state/store-id tokens that hard
        # descriptors carry, without enumerating any noise vocabulary, while still
        # rejecting merchants that merely share a token (e.g. a common city).
        match = process.extractOne(query, keys, scorer=fuzz.token_set_ratio,
                                   score_cutoff=self.score_cutoff)
        if match:
            return self._canon_clean_to_name[match[0]]
        return derive_canonical(stripped)

    def standardize(self, txn: Txn) -> Standardized:
        merchant = self._match_merchant(txn.description)
        direction = "incoming" if txn.amount > 0 else "outgoing"
        if txn.mcc is not None and txn.mcc in self._mcc_to_category:
            category = self._mcc_to_category[txn.mcc]
        else:
            category = self._merchant_to_category.get(merchant, self._default_category)
        return Standardized(canonical_merchant=merchant, category=category,
                            direction=direction)
