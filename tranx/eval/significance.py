"""Paired merchant-cluster bootstrap for comparing two routes on the same rows.

Rows of one merchant (or txn_type, for merchant-less rows) are correlated, so an
iid CI would be too narrow. Clusters are resampled with replacement (the same
resample for both routes); the diff per resample is the resampled weighted
numerator over the resampled weighted denominator, computed via per-cluster sums
(np.bincount) rather than by concatenating row indices.
"""
import numpy as np
from tranx import config


def _cluster_sums(a_ok, b_ok, clusters, weights):
    a = np.asarray(a_ok, dtype=float)
    b = np.asarray(b_ok, dtype=float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(uniq)
    num_sum = np.bincount(inv, weights=(a - b) * w, minlength=k)
    den_sum = np.bincount(inv, weights=w, minlength=k)
    return num_sum, den_sum, k


def paired_bootstrap(a_ok, b_ok, clusters, weights=None, n_boot: int = config.BOOTSTRAP_N,
                     seed: int = 0) -> dict:
    num_sum, den_sum, k = _cluster_sums(a_ok, b_ok, clusters, weights)
    total_den = den_sum.sum()
    diff = float(num_sum.sum() / total_den) if total_den else 0.0

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, k, k), minlength=k)
        d = (counts * den_sum).sum()
        diffs[i] = (counts * num_sum).sum() / d if d else 0.0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": diff, "lo": float(lo), "hi": float(hi),
            "n_clusters": int(k), "n_rows": int(len(np.asarray(a_ok)))}


def leave_one_cluster_out(a_ok, b_ok, clusters, weights=None) -> tuple:
    """Drop each merchant cluster in turn and recompute the diff on the rest;
    return (min, max) over clusters — a sensitivity range, not a CI."""
    num_sum, den_sum, k = _cluster_sums(a_ok, b_ok, clusters, weights)
    total_num, total_den = num_sum.sum(), den_sum.sum()
    diffs = []
    for i in range(k):
        den = total_den - den_sum[i]
        if den == 0:
            continue
        diffs.append((total_num - num_sum[i]) / den)
    return (min(diffs), max(diffs))
