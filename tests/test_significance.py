import numpy as np
from tranx.eval.significance import paired_bootstrap, leave_one_cluster_out


def test_identical_predictions_give_zero_width_interval_at_zero():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 200).astype(float)
    clusters = rng.integers(0, 20, 200)
    res = paired_bootstrap(a, a.copy(), clusters, n_boot=500, seed=0)
    assert res["diff"] == 0.0
    assert res["lo"] == 0.0
    assert res["hi"] == 0.0
    assert res["n_clusters"] == len(set(clusters))
    assert res["n_rows"] == 200


def test_swapping_routes_flips_the_sign():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 2, 300).astype(float)
    b = rng.integers(0, 2, 300).astype(float)
    clusters = rng.integers(0, 15, 300)
    fwd = paired_bootstrap(a, b, clusters, n_boot=1000, seed=0)
    rev = paired_bootstrap(b, a, clusters, n_boot=1000, seed=0)
    assert np.isclose(rev["diff"], -fwd["diff"])
    # Same resamples run in reverse: the interval mirrors around zero.
    assert np.isclose(rev["lo"], -fwd["hi"])
    assert np.isclose(rev["hi"], -fwd["lo"])


def test_row_weighted_diff_matches_hand_computed_toy():
    # Two clusters (merchants), unequal weights. a beats b on cluster 1 (weight 3),
    # loses on cluster 2 (weight 1): weighted diff = (1*3 + -1*1) / (3+1) = 0.5
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    w = np.array([3.0, 1.0])
    clusters = np.array(["m1", "m2"])
    res = paired_bootstrap(a, b, clusters, weights=w, n_boot=10, seed=0)
    assert np.isclose(res["diff"], 0.5)


def test_paired_bootstrap_result_keys():
    a = np.array([1.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 1.0, 0.0])
    clusters = np.array(["m1", "m1", "m2", "m2"])
    res = paired_bootstrap(a, b, clusters, n_boot=50, seed=0)
    assert set(res) == {"diff", "lo", "hi", "n_clusters", "n_rows"}
    assert res["n_clusters"] == 2
    assert res["n_rows"] == 4
    assert res["lo"] <= res["diff"] <= res["hi"]


def test_leave_one_cluster_out_hand_computed():
    # cluster1: a-b=+1, cluster2: a-b=+1, cluster3: a-b=-1 (unit weights).
    a = np.array([1.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    clusters = np.array(["m1", "m2", "m3"])
    # Drop m1 -> (1-1)/2=0; drop m2 -> (1-1)/2=0; drop m3 -> (1+1)/2=1.
    lo, hi = leave_one_cluster_out(a, b, clusters)
    assert np.isclose(lo, 0.0)
    assert np.isclose(hi, 1.0)
