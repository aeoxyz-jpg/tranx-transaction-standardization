import time
import polars as pl
from tranx.routes.base import Route
from tranx.schema import Txn

_TXN_FIELDS = ["txn_id", "customer_id", "description", "transaction_type_code",
               "mcc", "amount", "payment_method", "posted_date", "country", "currency"]


def run_route(route: Route, feed: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Run a fitted route over the feed; return prediction df and timing stats."""
    rows = []
    start = time.perf_counter()
    for r in feed.select(_TXN_FIELDS).iter_rows(named=True):
        txn = Txn(**r)
        out = route.standardize(txn)
        rows.append({
            "txn_id": txn.txn_id,
            "customer_id": txn.customer_id,
            "canonical_merchant": out.canonical_merchant,
            "category": out.category,
            "direction": out.direction,
        })
    elapsed = time.perf_counter() - start
    n = len(rows)
    timing = {"n": n, "total_s": round(elapsed, 4),
              "avg_ms": round(1000 * elapsed / n, 3) if n else 0.0}
    return pl.DataFrame(rows), timing
