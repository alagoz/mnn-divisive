"""
mNN-Divisive: divisive hierarchical clustering with group-wise m-nearest-neighbour growing.

FROZEN DEFINITION (do not change after benchmarking)
  Leaf selection : split the current leaf with the most points.
  Seeds          : density-trimmed farthest pair - drop the `trim` (5 %) of points with the
                   largest distance to their `knn`-th (5th) neighbour, then take the
                   farthest pair among the rest.
  Growing order  : the next point to label is the unlabelled point with the smallest mean
                   distance to the m most recently added members of either group ("recent").
  Assignment     : the point joins the group whose m nearest labelled members are closer
                   on average (group-wise m-NN). Ties -> nearer seed.
  m              : 10.

  Input is only a dissimilarity matrix, so any metric can be used (metric="precomputed").
  Time  O(n^2) per split level (worst case O(n^2 K)), memory O(n^2) for the matrix.

BACKENDS
  "numba" : compiled kernel (pip install numba) - fastest.
  "numpy" : vectorised fallback, no extra dependency.
  "auto"  : numba if importable, otherwise numpy.  Both give the same partitions
            (up to floating-point ties).

USAGE
  from mnn_divisive import MNNDivisive
  model = MNNDivisive(n_clusters=3).fit(X)          # Euclidean on X
  model.labels_                                     # K-cluster partition
  model.labels_at(2)                                # coarser level of the same hierarchy
  MNNDivisive(3, metric="precomputed").fit(D)       # any distance matrix
"""
import numpy as np
from sklearn.metrics import pairwise_distances

__all__ = ["MNNDivisive", "mnn_split", "trimmed_seeds"]

# --------------------------------------------------------------------------- seeds
def trimmed_seeds(D, trim=0.05, knn=5):
    m = len(D)
    k = min(knn, m - 1)
    kd = np.partition(D, k, axis=1)[:, k]
    keep = np.flatnonzero(kd <= np.quantile(kd, 1 - trim))
    if len(keep) < 2:
        keep = np.arange(m)
    sub = D[np.ix_(keep, keep)]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    return int(keep[i]), int(keep[j])

# ------------------------------------------------------------------- numpy backend
def _split_numpy(D, m, s0, s1):
    n = len(D)
    lab = np.full(n, -1); lab[s0], lab[s1] = 0, 1
    mem = [np.empty(n, np.int64), np.empty(n, np.int64)]
    mem[0][0], mem[1][0] = s0, s1
    cnt = [1, 1]
    rsum = [D[:, s0].astype(float), D[:, s1].astype(float)]   # sum of distances to last-m members
    rsum[0][[s0, s1]] = np.inf; rsum[1][[s0, s1]] = np.inf      # labelled points never chosen
    for _ in range(n - 2):
        key = np.minimum(rsum[0] / min(m, cnt[0]), rsum[1] / min(m, cnt[1]))
        p = int(np.argmin(key))
        sc = []
        for g in (0, 1):
            d = D[p, mem[g][:cnt[g]]]
            mg = min(m, cnt[g])
            if mg < len(d):
                d = np.partition(d, mg - 1)[:mg]
            sc.append(np.sort(d).mean())
        g = int(sc[1] < sc[0]) if sc[0] != sc[1] else int(D[p, s1] < D[p, s0])
        lab[p] = g
        mem[g][cnt[g]] = p; cnt[g] += 1
        rsum[g] += D[:, p]
        if cnt[g] > m:
            rsum[g] -= D[:, mem[g][cnt[g] - m - 1]]
        rsum[0][p] = np.inf; rsum[1][p] = np.inf
    return lab

# ------------------------------------------------------------------- numba backend
try:
    from numba import njit

    @njit(cache=True)
    def _split_numba_kernel(D, m, s0, s1):
        n = D.shape[0]
        lab = np.full(n, -1, np.int64); lab[s0] = 0; lab[s1] = 1
        mem = np.empty((2, n), np.int64); mem[0, 0] = s0; mem[1, 0] = s1
        cnt = np.array([1, 1], np.int64)
        rs = np.empty((2, n))
        for i in range(n):
            rs[0, i] = D[i, s0]; rs[1, i] = D[i, s1]
        rs[0, s0] = np.inf; rs[1, s0] = np.inf; rs[0, s1] = np.inf; rs[1, s1] = np.inf
        best = np.empty(m)
        for _ in range(n - 2):
            w0 = min(m, cnt[0]); w1 = min(m, cnt[1])
            p = -1; kv = np.inf
            for i in range(n):
                a = rs[0, i] / w0; b = rs[1, i] / w1
                v = a if a < b else b
                if v < kv:
                    kv = v; p = i
            sc = np.empty(2)
            for g in range(2):
                mg = min(m, cnt[g]); filled = 0
                for t in range(cnt[g]):                      # insertion into sorted m-smallest
                    d = D[p, mem[g, t]]
                    if filled < mg:
                        j = filled; filled += 1
                    elif d < best[mg - 1]:
                        j = mg - 1
                    else:
                        continue
                    while j > 0 and best[j - 1] > d:
                        best[j] = best[j - 1]; j -= 1
                    best[j] = d
                s = 0.0
                for t in range(mg):
                    s += best[t]
                sc[g] = s / mg
            if sc[0] != sc[1]:
                g = 1 if sc[1] < sc[0] else 0
            else:
                g = 1 if D[p, s1] < D[p, s0] else 0
            lab[p] = g
            mem[g, cnt[g]] = p; cnt[g] += 1
            old = mem[g, cnt[g] - m - 1] if cnt[g] > m else -1
            for i in range(n):
                rs[g, i] += D[i, p]
                if old >= 0:
                    rs[g, i] -= D[i, old]
            rs[0, p] = np.inf; rs[1, p] = np.inf
        return lab

    _HAS_NUMBA = True
except ImportError:                                           # pragma: no cover
    _HAS_NUMBA = False

def mnn_split(D, m=10, trim=0.05, knn=5, backend="auto"):
    """Split the points of distance matrix D into two groups (labels 0/1)."""
    n = len(D)
    if n < 4:
        lab = np.zeros(n, int); lab[n // 2:] = 1; return lab
    s0, s1 = trimmed_seeds(D, trim, knn)
    use_numba = (backend == "numba") or (backend == "auto" and _HAS_NUMBA)
    if use_numba:
        if not _HAS_NUMBA:
            raise ImportError("numba is not installed (pip install numba)")
        return _split_numba_kernel(np.ascontiguousarray(D, dtype=np.float64), m, s0, s1).astype(int)
    return _split_numpy(D, m, s0, s1)

# --------------------------------------------------------------------------- model
class MNNDivisive:
    """Divisive hierarchical clustering with group-wise m-NN growing (frozen version)."""

    def __init__(self, n_clusters=2, m=10, trim=0.05, knn=5, metric="euclidean", backend="auto"):
        self.n_clusters, self.m, self.trim, self.knn = n_clusters, m, trim, knn
        self.metric, self.backend = metric, backend

    def fit(self, X, y=None):
        D = np.asarray(X, float) if self.metric == "precomputed" else pairwise_distances(X, metric=self.metric)
        n = len(D)
        leaves = [np.arange(n)]
        self.splits_ = []                      # (index of split leaf, left members, right members)
        while len(leaves) < self.n_clusters:
            cand = [i for i, L in enumerate(leaves) if len(L) >= 2]
            if not cand:
                break
            i = max(cand, key=lambda t: len(leaves[t]))
            L = leaves.pop(i)
            lab = mnn_split(D[np.ix_(L, L)], self.m, self.trim, self.knn, self.backend)
            if lab.min() == lab.max():         # degenerate split -> halve
                lab = np.zeros(len(L), int); lab[len(L) // 2:] = 1
            leaves += [L[lab == 0], L[lab == 1]]
            self.splits_.append((L, L[lab == 0], L[lab == 1]))
        self.n_samples_ = n
        self.labels_ = self.labels_at(len(leaves))
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def labels_at(self, k):
        """Partition with k clusters from the same hierarchy (1 <= k <= n_clusters)."""
        leaves = [np.arange(self.n_samples_)]
        for parent, left, right in self.splits_[:max(0, k - 1)]:
            idx = next(i for i, L in enumerate(leaves) if len(L) == len(parent) and np.array_equal(L, parent))
            leaves.pop(idx); leaves += [left, right]
        lab = np.empty(self.n_samples_, int)
        for c, L in enumerate(leaves):
            lab[L] = c
        return lab
