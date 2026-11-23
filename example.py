"""
Example run of mNN-Divisive.

    python example.py

Shows: (1) clustering a non-convex 2-D dataset, (2) reading several levels of the same
hierarchy on a real dataset (Wine), (3) using your own dissimilarity matrix.
Saves docs/example.png.
"""
import time
import numpy as np
from sklearn.datasets import make_moons, load_wine
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from mnn_divisive import MNNDivisive, _HAS_NUMBA

print(f"backend: {'numba' if _HAS_NUMBA else 'numpy'}\n")

# 1) Two moons: non-convex clusters ------------------------------------------------
X, y = make_moons(n_samples=600, noise=0.07, random_state=0)
t = time.perf_counter()
model = MNNDivisive(n_clusters=2, m=10).fit(X)
print(f"[moons]  n={len(X)}  ARI={adjusted_rand_score(y, model.labels_):.3f}  "
      f"time={1000 * (time.perf_counter() - t):.0f} ms")

# 2) Wine: one fit gives every level of the hierarchy up to K ----------------------
Xi, yi = load_wine(return_X_y=True)
Xi = StandardScaler().fit_transform(Xi)
wine = MNNDivisive(n_clusters=3).fit(Xi)
for k in (1, 2, 3):
    lab = wine.labels_at(k)
    print(f"[wine]   k={k}  cluster sizes={np.bincount(lab).tolist()}  "
          f"ARI={adjusted_rand_score(yi, lab):.3f}")

# 3) Any dissimilarity: pass a precomputed matrix -----------------------------------
D = pairwise_distances(Xi, metric="manhattan")
lab = MNNDivisive(n_clusters=3, metric="precomputed").fit_predict(D)
print(f"[wine, Manhattan distance]  ARI={adjusted_rand_score(yi, lab):.3f}")

# 4) Figure --------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    for a, lab, title in [(ax[0], y, "Reference labels"), (ax[1], model.labels_, "mNN-Divisive (m = 10)")]:
        a.scatter(X[:, 0], X[:, 1], c=np.where(lab == lab[0], "#2a78d6", "#eb6834"), s=8, lw=0)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig("docs/example.png", dpi=150)
    print("\nfigure saved to docs/example.png")
except ImportError:
    pass
