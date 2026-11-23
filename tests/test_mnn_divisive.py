"""Basic tests for mNN-Divisive.  Run:  python -m pytest tests"""
import os, sys
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mnn_divisive as M


def data(seed=0):
    X, y = make_blobs([60, 50, 40], centers=[[0, 0], [5, 0], [2, 4]], cluster_std=0.6, random_state=seed)
    return X, y


def test_recovers_separated_blobs():
    X, y = data()
    assert adjusted_rand_score(y, M.MNNDivisive(3).fit_predict(X)) > 0.95


def test_moons():
    X, y = make_moons(400, noise=0.06, random_state=0)
    assert adjusted_rand_score(y, M.MNNDivisive(2).fit_predict(X)) > 0.95


def test_deterministic():
    X, _ = data(1)
    a = M.MNNDivisive(3).fit_predict(X)
    b = M.MNNDivisive(3).fit_predict(X)
    assert np.array_equal(a, b)


def test_scale_invariance():
    X, _ = data(2)
    D = pairwise_distances(X)
    a = M.MNNDivisive(3, metric="precomputed").fit_predict(D)
    b = M.MNNDivisive(3, metric="precomputed").fit_predict(7.3 * D)
    assert np.array_equal(a, b)


def test_precomputed_equals_euclidean():
    X, _ = data(3)
    a = M.MNNDivisive(3).fit_predict(X)
    b = M.MNNDivisive(3, metric="precomputed").fit_predict(pairwise_distances(X))
    assert np.array_equal(a, b)


def test_hierarchy_is_nested():
    X, _ = data(4)
    model = M.MNNDivisive(5).fit(X)
    for k in range(1, 5):
        coarse, fine = model.labels_at(k), model.labels_at(k + 1)
        for c in np.unique(fine):                 # every finer cluster lies inside one coarser cluster
            assert len(np.unique(coarse[fine == c])) == 1
        assert len(np.unique(fine)) == k + 1


@pytest.mark.skipif(not M._HAS_NUMBA, reason="numba not installed")
def test_backends_agree():
    X, _ = data(5)
    D = pairwise_distances(X)
    a = M.MNNDivisive(3, metric="precomputed", backend="numpy").fit_predict(D)
    b = M.MNNDivisive(3, metric="precomputed", backend="numba").fit_predict(D)
    assert adjusted_rand_score(a, b) == 1.0


def test_tiny_inputs():
    for n in (1, 2, 3, 5):
        X = np.arange(n, dtype=float).reshape(-1, 1)
        lab = M.MNNDivisive(min(2, n)).fit_predict(X)
        assert len(lab) == n
