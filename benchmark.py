"""
Final benchmark: mNN-Divisive (frozen: recent order, m = 10, trimmed seeds) versus reference
clustering algorithms on UCI, Gagolewski's benchmark suite and 4 synthetic sets.

Needs mnn_divisive.py in the same folder.  Optional: pip install numba genieclust ucimlrepo psutil

PROTOCOL  R stratified subsamples (80 %, at most 2,000 points) per dataset; z-scored features;
          Euclidean distance; k = number of reference classes; ARI against the reference labels
          (Gagolewski: noise points excluded, best-matching reference labelling); Silhouette on
          the same distance matrix; wall-clock time of the clustering call.
STATS     Friedman test, Nemenyi critical difference, and mNN-Divisive vs every reference with
          Wilcoxon signed-rank tests (Holm-corrected) - overall and per dataset group.

RUN       python benchmark.py --source all --gb_root clustering-data-v1
          python benchmark.py --source synthetic --R 3          # 1-minute smoke test
          Spyder: edit the CONFIG block at the bottom and press F5.
OUTPUTS   (--out, default final_out)
          run_info.txt, datasets_table.csv, main_raw.csv, main_summary.csv,
          overall_<group>.csv, stats_<group>.txt, proposed_vs_refs_<group>.csv, Fig_avg_ranks.png
"""
import argparse, glob, gzip, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# ======================================================= reference splitters / helpers
def split_kmeans(Xs, D, rng):
    return KMeans(2, n_init=10, random_state=rng.randint(1 << 30)).fit_predict(Xs)

def split_gmm(Xs, D, rng):
    return GaussianMixture(2, n_init=3, reg_covar=1e-4,
                           random_state=rng.randint(1 << 30)).fit_predict(Xs)

def split_ward(Xs, D, rng):
    return AgglomerativeClustering(2, linkage="ward").fit_predict(Xs)

def split_pddp(Xs, D, rng):
    """Principal Direction Divisive Partitioning (Boley, 1998)."""
    Z = Xs - Xs.mean(0)
    v = np.linalg.svd(Z, full_matrices=False)[2][0]
    return (Z @ v > 0).astype(int)

def split_diana(Xs, D, rng):
    """Textbook DIANA splinter step (Kaufman & Rousseeuw 1990, ch. 6), O(m^2)."""
    m = len(D)
    inS = np.zeros(m, bool)
    s = int(np.argmax(D.sum(1)))
    inS[s] = True
    sumS = D[:, s].copy()
    sumC = D.sum(1) - D[:, s]
    while True:
        nS, nC = inS.sum(), m - inS.sum()
        if nC <= 1:
            break
        avgC = sumC / (nC - 1)              # excludes self (D[i,i]=0)
        avgS = sumS / nS
        diff = np.where(inS, -np.inf, avgC - avgS)
        j = int(np.argmax(diff))
        if diff[j] <= 0:
            break
        inS[j] = True
        sumS += D[:, j]
        sumC -= D[:, j]
    return inS.astype(int)

def leaf_score(idx, D, X, rule):
    if rule == "size":
        return len(idx)
    sub = D[np.ix_(idx, idx)]
    if rule == "diameter":
        return sub.max()
    if rule == "sse":                       # sum_{i<j} d_ij^2 / m  (= SSE for Euclidean)
        return (sub ** 2).sum() / (2 * len(idx))
    raise ValueError(rule)

def divisive(X, D, k, splitter, split_rule="size", seed=0):
    rng = np.random.RandomState(seed)
    leaves = [np.arange(len(X))]
    while len(leaves) < k:
        cand = [i for i, L in enumerate(leaves) if len(L) >= 2]
        if not cand:
            break
        i = max(cand, key=lambda t: leaf_score(leaves[t], D, X, split_rule))
        L = leaves.pop(i)
        lab = np.asarray(splitter(X[L], D[np.ix_(L, L)], rng))
        if lab.min() == lab.max():           # degenerate split -> fall back to halving
            lab = np.zeros(len(L), int); lab[len(L) // 2:] = 1
        leaves += [L[lab == 0], L[lab == 1]]
    out = np.zeros(len(X), int)
    for c, L in enumerate(leaves):
        out[L] = c
    return out

def kmedoids(D, k, rng, n_init=5, max_iter=100):
    """Alternating k-medoids with k-medoids++ initialisation; distance matrix only."""
    m = len(D); best, best_cost = None, np.inf
    for _ in range(n_init):
        med = [rng.randint(m)]
        for _ in range(1, k):
            dmin = D[:, med].min(1) ** 2
            p = dmin / dmin.sum() if dmin.sum() > 0 else None
            med.append(rng.choice(m, p=p))
        med = np.array(med)
        for _ in range(max_iter):
            lab = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for c in range(k):
                idx = np.where(lab == c)[0]
                if len(idx):
                    new[c] = idx[np.argmin(D[np.ix_(idx, idx)].sum(1))]
            if np.array_equal(new, med):
                break
            med = new
        lab = np.argmin(D[:, med], axis=1); cost = D[np.arange(m), med[lab]].sum()
        if cost < best_cost:
            best, best_cost = lab, cost
    return best

def split_kmedoids(Xs, D, rng):
    return kmedoids(D, 2, rng)


# ================================================================ dataset loaders
def _ds(name, kind, X, y, noise=False, metric="euclidean", D=None):
    return dict(name=name, kind=kind, X=np.asarray(X, float),
                labels=[np.asarray(pd.factorize(y)[0] if not noise else y, int)],
                noise=noise, metric=metric, D=D)

def csv_datasets(folder):
    out = []
    for fn in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(fn).dropna()
        y = df.iloc[:, -1]
        F = df.iloc[:, :-1]
        cat = F.select_dtypes(exclude="number")
        num = F.select_dtypes(include="number").to_numpy(float)
        if cat.shape[1]:
            oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(cat)
            X = np.hstack([num, oh]) if num.size else oh
        else:
            X = num
        out.append(_ds(os.path.basename(fn)[:-4], "real", X, y))
    return out

UCI_CLUSTERABLE = {
    "iris": (53, "iris"), "wine": (109, "wine"), "wdbc": (17, "breast cancer wisconsin"),
    "seeds": (236, "seeds"), "ecoli": (39, "ecoli"), "glass": (42, "glass"),
    "dermatology": (33, "dermatology"), "zoo": (111, "zoo"),
    "optdigits": (80, "optical"), "pendigits": (81, "pen"),
    "segmentation": (50, "segmentation"), "landsat": (146, "landsat"),
    "vertebral": (212, "vertebral"), "banknote": (267, "banknote"),
    "ionosphere": (52, "ionosphere"), "sonar": (151, "sonar"),
    "yeast": (110, "yeast"), "parkinsons": (174, "parkinsons"),
}

def uci_datasets(names=None, cache_dir="uci_cache"):
    """Download (once) with ucimlrepo, cache as <cache_dir>/<name>.csv (label = last
    column), then load through csv_datasets(). Afterwards it works offline."""
    os.makedirs(cache_dir, exist_ok=True)
    names = names or list(UCI_CLUSTERABLE)
    for nm in names:
        path = os.path.join(cache_dir, f"{nm}.csv")
        if os.path.exists(path):
            continue
        if nm not in UCI_CLUSTERABLE:
            print(f"  [skip] {nm}: not in UCI_CLUSTERABLE (add its id)"); continue
        uid, key = UCI_CLUSTERABLE[nm]
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            print(f"  [skip] {nm}: not cached and ucimlrepo not installed (pip install ucimlrepo)")
            continue
        try:
            r = fetch_ucirepo(id=uid)
        except Exception as e:
            print(f"  [skip] {nm} (id {uid}): download failed: {e}"); continue
        title = str(r.metadata.name)
        if key not in title.lower():
            print(f"  [WARNING] id {uid} returned '{title}', expected '{key}'. Skipped - fix the id.")
            continue
        F, T = r.data.features, r.data.targets
        if T is None or T.shape[1] == 0:
            print(f"  [skip] {nm}: no target column"); continue
        df = pd.concat([F, T.iloc[:, [0]].rename(columns=lambda c: "__label__")], axis=1)
        df.to_csv(path, index=False)
        print(f"  cached {nm}: '{title}', n={len(df)}, d={F.shape[1]}, classes={T.iloc[:, 0].nunique()}")
    sel = set(names)
    out = [d for d in csv_datasets(cache_dir) if d["name"] in sel]
    for d in out:
        d["kind"] = "uci"
    print(f"UCI: {len(out)} datasets loaded")
    return out

GB_DEFAULT_BATTERIES = ["uci", "fcps", "wut", "graves", "sipu", "other"]

GB_SKIP = {"mnist", "h2mg", "g2mg"}          # very large / generated families

def _read_txt_matrix(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return np.loadtxt(fh, ndmin=2)

def gagolewski_datasets(root, batteries=None, names=None, max_n_full=None):
    """Load Gagolewski's clustering-data-v1 batteries. Every dataset is returned with all
    its reference labellings (labels0, labels1, ...); label 0 marks noise points."""
    batteries = batteries or GB_DEFAULT_BATTERIES
    out = []
    try:
        import clustbench
    except ImportError:
        clustbench = None
    for bat in batteries:
        if bat in GB_SKIP:
            continue
        files = sorted(glob.glob(os.path.join(root, bat, "*.data.gz")) +
                       glob.glob(os.path.join(root, bat, "*.data")))
        for f in files:
            nm = os.path.basename(f).split(".data")[0]
            if names and nm not in names:
                continue
            if clustbench is not None:
                b = clustbench.load_dataset(bat, nm, path=root)
                X, labs = np.asarray(b.data, float), [np.asarray(l, int) for l in b.labels]
            else:
                X = _read_txt_matrix(f)
                labs = []
                for lf in sorted(glob.glob(os.path.join(root, bat, nm + ".labels*"))):
                    labs.append(_read_txt_matrix(lf).ravel().astype(int))
            if not labs:
                print(f"  [skip] {bat}/{nm}: no reference labels"); continue
            if max_n_full and len(X) > max_n_full:
                print(f"  [skip] {bat}/{nm}: n={len(X)} > {max_n_full}"); continue
            out.append(dict(name=f"{bat}/{nm}", kind=bat, X=X, labels=labs,
                            noise=True, metric="euclidean", D=None))
    print(f"Gagolewski: {len(out)} datasets loaded from {root}")
    return out


# ======================================================= protocol and statistics
def _score(pred, labels, noise):
    """Max ARI over reference labellings, on non-noise points (if noise=True)."""
    best = -np.inf
    for lab in labels:
        m = lab > 0 if noise else np.ones(len(lab), bool)
        best = max(best, adjusted_rand_score(lab[m], pred[m]))
    return best

def _k_of(lab, noise):
    return len(np.unique(lab[lab > 0])) if noise else len(np.unique(lab))

def run(datasets, algos, R=10, frac=0.8, max_n=2000):
    rows = []
    for ds in datasets:
        X0, labs0, noise = ds["X"], ds["labels"], ds["noise"]
        strat = labs0[0]
        # each reference labelling with its own k; the clustering is run with k of labels0
        k = _k_of(labs0[0], noise)
        n_use = min(int(frac * len(strat)), max_n)
        try:
            splits = list(StratifiedShuffleSplit(R, train_size=n_use, random_state=0).split(X0, strat))
        except ValueError:                                  # classes too small to stratify
            rs = np.random.RandomState(0)
            splits = [(rs.choice(len(strat), n_use, replace=False), None) for _ in range(R)]
        for r, (idx, _) in enumerate(splits):
            labs = [l[idx] for l in labs0]
            X = X0[idx]
            X = X[:, X.std(0) > 0]
            Xf = StandardScaler().fit_transform(X) if ds["metric"] == "euclidean" else X
            De = pairwise_distances(Xf)
            Dm = De if ds["D"] is None else ds["D"][np.ix_(idx, idx)]
            for an, (f, needs_feat) in algos.items():
                D_use = De if needs_feat else Dm
                inp = "euclidean" if needs_feat else ds["metric"]
                t = time.perf_counter()
                try:
                    pred = np.asarray(f(Xf, D_use, k, r))
                except Exception as e:                      # never silently substitute
                    print(f"  [error] {ds['name']} / {an}: {e}")
                    rows.append(dict(dataset=ds["name"], kind=ds["kind"], rep=r, method=an,
                                     input=inp, ARI=np.nan, Silhouette=np.nan, time=np.nan,
                                     n=len(idx), d=X.shape[1], k=k, largest_frac=np.nan,
                                     error=str(e)))
                    continue
                el = time.perf_counter() - t
                sil = (silhouette_score(Dm, pred, metric="precomputed")
                       if 1 < len(np.unique(pred)) < len(pred) else np.nan)
                rows.append(dict(dataset=ds["name"], kind=ds["kind"], rep=r, method=an, input=inp,
                                 ARI=_score(pred, labs, noise), Silhouette=sil, time=el,
                                 n=len(idx), d=X.shape[1], k=k,
                                 largest_frac=np.bincount(pred).max() / len(pred), error=""))
        print(f"  done {ds['name']}  (n={len(strat)}, k={k}, metric={ds['metric']})")
    return pd.DataFrame(rows)

def dataset_table(datasets):
    """Table for the manuscript: one row per dataset (answers R1.2 / R3)."""
    rows = []
    for ds in datasets:
        lab = ds["labels"][0]; core = lab[lab > 0] if ds["noise"] else lab
        cnt = np.bincount(pd.factorize(core)[0])
        rows.append(dict(dataset=ds["name"], group=ds["kind"], n=len(lab), d=ds["X"].shape[1],
                         k=len(cnt), majority_frac=round(cnt.max() / cnt.sum(), 3),
                         imbalance_ratio=round(cnt.max() / cnt.min(), 2),
                         noise_frac=round(float((lab == 0).mean()), 3) if ds["noise"] else 0.0,
                         n_references=len(ds["labels"]), metric=ds["metric"]))
    return pd.DataFrame(rows)

Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
       10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
       16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}

def friedman_nemenyi(df, metric="ARI"):
    M = df.groupby(["dataset", "method"])[metric].mean().unstack()
    M = M.dropna()
    ranks = M.rank(axis=1, ascending=False)           # per dataset, 1 = best
    avg = ranks.mean().sort_values()
    N, k = M.shape
    chi, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    try:
        from scipy.stats import studentized_range
        qa = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    except Exception:
        qa = Q05[min(k, max(Q05))]
    cd = qa * np.sqrt(k * (k + 1) / (6 * N))
    best = avg.index[0]
    pw = []
    for m in avg.index[1:]:
        try:
            pv = stats.wilcoxon(M[best], M[m], zero_method="zsplit").pvalue
        except ValueError:
            pv = 1.0
        pw.append((m, pv))
    # Holm
    order = np.argsort([p for _, p in pw]); adj = np.empty(len(pw)); run_max = 0
    for rank_i, j in enumerate(order):
        run_max = max(run_max, min(1, (len(pw) - rank_i) * pw[j][1])); adj[j] = run_max
    wil = pd.DataFrame({"vs_best": [m for m, _ in pw], "p_raw": [p for _, p in pw], "p_holm": adj})
    return avg, chi, p, cd, N, k, best, wil


# =================================================================== frozen method
from mnn_divisive import MNNDivisive, _HAS_NUMBA

PROPOSED = "mNN-Divisive (m=10)"

def references():
    agg = lambda link: (lambda X, D, k, s:
        AgglomerativeClustering(k, metric="precomputed", linkage=link).fit_predict(D))
    A = {
        "DIANA":           (lambda X, D, k, s: divisive(X, D, k, split_diana, split_rule="diameter", seed=s), False),
        "Div-KMeans":      (lambda X, D, k, s: divisive(X, D, k, split_kmeans, seed=s), True),
        "Div-KMedoids":    (lambda X, D, k, s: divisive(X, D, k, split_kmedoids, seed=s), False),
        "Div-Ward":        (lambda X, D, k, s: divisive(X, D, k, split_ward, seed=s), True),
        "Div-GMM":         (lambda X, D, k, s: divisive(X, D, k, split_gmm, seed=s), True),
        "PDDP":            (lambda X, D, k, s: divisive(X, D, k, split_pddp, seed=s), True),
        "Agglo-Single":    (agg("single"), False),
        "Agglo-Complete":  (agg("complete"), False),
        "Agglo-Average":   (agg("average"), False),
        "Agglo-Ward":      (lambda X, D, k, s: AgglomerativeClustering(k, linkage="ward").fit_predict(X), True),
        "KMeans":          (lambda X, D, k, s: KMeans(k, n_init=10, random_state=s).fit_predict(X), True),
        "KMedoids":        (lambda X, D, k, s: kmedoids(D, k, np.random.RandomState(s)), False),
        "GMM":             (lambda X, D, k, s: GaussianMixture(k, n_init=3, reg_covar=1e-4, random_state=s).fit_predict(X), True),
        "Spectral (kNN graph)": (lambda X, D, k, s: SpectralClustering(
            k, affinity="precomputed_nearest_neighbors", n_neighbors=min(10, len(D) - 1),
            assign_labels="cluster_qr", random_state=s).fit_predict(D), False),
    }
    try:
        import genieclust
        A["Genie (g=0.3)"] = (lambda X, D, k, s: genieclust.Genie(n_clusters=k, gini_threshold=0.3).fit_predict(X), True)
    except ImportError:
        print("  [info] genieclust not installed -> Genie skipped (pip install genieclust)")
    return A

def algorithms(m_sensitivity=False, backend="auto"):
    A = {PROPOSED: (lambda X, D, k, s: MNNDivisive(k, m=10, metric="precomputed", backend=backend).fit_predict(D), False)}
    if m_sensitivity:
        for m in (3, 5, 20):
            A[f"mNN-Divisive (m={m})"] = ((lambda m_: lambda X, D, k, s:
                MNNDivisive(k, m=m_, metric="precomputed", backend=backend).fit_predict(D))(m), False)
    A.update(references())
    return A

# =========================================================================== datasets
def load(a):
    ds = []
    if a.source in ("synthetic", "all"):
        ds += synthetic_datasets()
    if a.source in ("uci", "all"):
        ds += uci_datasets(a.uci_names, a.uci_cache)
    if a.source in ("gagolewski", "all"):
        ds += gagolewski_datasets(a.gb_root, a.batteries, None, a.gb_max_n_full)
    if not ds:
        raise SystemExit("No datasets loaded - check paths/options.")
    return ds

def synthetic_datasets():
    out = []
    X, y = make_blobs([300, 300, 300], centers=[[0, 0], [6, 0], [3, 5]], random_state=0)
    out.append(_ds("blobs_balanced", "synthetic", X, y))
    X, y = make_blobs([700, 150, 50], centers=[[0, 0], [6, 0], [3, 5]], random_state=0)
    out.append(_ds("blobs_imbalanced", "synthetic", X, y))
    X, y = make_blobs([400, 400], centers=[[0, 0], [5, 0]], cluster_std=[0.5, 2.0], random_state=0)
    out.append(_ds("varied_density", "synthetic", X, y))
    X, y = make_blobs([300, 300, 300], centers=[[0, 0], [6, 0], [3, 5]], random_state=1)
    out.append(_ds("anisotropic", "synthetic", X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y))
    return out

# ========================================================================== reporting
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return adj

def report(df, out):
    M = df.groupby(["dataset", "method"]).ARI.mean().unstack()
    gaps = M.columns[M.isna().any()]
    if len(gaps):
        print(f"  [warning] dropping methods with missing results: {list(gaps)}")
        df = df[~df.method.isin(gaps)]; M = M.drop(columns=gaps)
    for g in ["ALL"] + sorted(df.kind.unique()):
        sub = df if g == "ALL" else df[df.kind == g]
        if sub.dataset.nunique() < 2:
            continue
        avg, chi, p, cd, N, q, best, _ = friedman_nemenyi(sub)
        Mg = M.loc[M.index.isin(sub.dataset.unique())]
        # proposed vs every reference: wins/losses, Wilcoxon, Holm
        rows = []
        for c in Mg.columns:
            if c == PROPOSED:
                continue
            d = Mg[PROPOSED] - Mg[c]
            try:
                pv = stats.wilcoxon(Mg[PROPOSED], Mg[c], zero_method="zsplit").pvalue
            except ValueError:
                pv = 1.0
            rows.append(dict(reference=c, wins=int((d > 0.005).sum()), losses=int((d < -0.005).sum()),
                             ties=int((d.abs() <= 0.005).sum()), mean_dARI=d.mean(), p_raw=pv))
        H = pd.DataFrame(rows); H["p_holm"] = holm(H.p_raw)
        H.sort_values("mean_dARI").round(4).to_csv(os.path.join(out, f"proposed_vs_refs_{g}.csv"), index=False)
        with open(os.path.join(out, f"stats_{g}.txt"), "w") as fh:
            fh.write(f"Friedman chi2={chi:.2f}, p={p:.4g}, N={N} datasets, q={q} algorithms, "
                     f"Nemenyi CD(0.05)={cd:.2f}\n\nAverage ranks (1=best):\n{avg.round(2).to_string()}\n\n"
                     f"{PROPOSED} vs each reference (Wilcoxon signed-rank, Holm):\n"
                     f"{H.sort_values('mean_dARI').round(4).to_string(index=False)}\n")
        ov = sub.groupby("method").agg(ARI=("ARI", "mean"), ARI_median=("ARI", "median"),
                                       ARI_sd_reps=("ARI", "std"), Silhouette=("Silhouette", "mean"),
                                       time_ms=("time", lambda x: 1000 * x.mean()))
        ov["avg_rank"] = avg; ov["n_best"] = Mg.eq(Mg.max(axis=1), axis=0).sum()
        ov.sort_values("avg_rank").round(3).to_csv(os.path.join(out, f"overall_{g}.csv"))
        if g == "ALL":
            plot_ranks(avg, cd, os.path.join(out, "Fig_avg_ranks.png"))
    print(pd.read_csv(os.path.join(out, "overall_ALL.csv")).to_string(index=False))

def plot_ranks(avg, cd, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    avg = avg.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(avg) + 1.2))
    cols = ["#eb6834" if i.startswith("mNN-Divisive") else "#2a78d6" for i in avg.index]
    ax.barh(range(len(avg)), avg.values, color=cols, height=0.7)
    ax.set_yticks(range(len(avg))); ax.set_yticklabels(avg.index, fontsize=9)
    ax.axvline(avg.min() + cd, color="#262624", ls="--", lw=1)
    ax.text(avg.min() + cd, len(avg) - 0.4, f"  best + CD ({cd:.2f})", fontsize=8, va="top")
    for i, v in enumerate(avg.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Average rank by ARI (1 = best)"); ax.grid(axis="x", alpha=.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def preflight(algos):
    """Run every algorithm once on a tiny dataset; stop with a clear message if any fails,
    so a broken package does not ruin a long run."""
    X, y = make_blobs([30, 30, 30], centers=[[0, 0], [5, 0], [2, 4]], random_state=0)
    X = StandardScaler().fit_transform(X); D = pairwise_distances(X)
    bad = []
    for nm, (f, _) in algos.items():
        try:
            lab = np.asarray(f(X, D, 3, 0))
            assert len(lab) == len(X)
        except Exception as e:
            bad.append(f"  {nm}: {type(e).__name__}: {e}")
    if bad:
        raise SystemExit("Pre-flight check failed - fix these before the full run:\n" + "\n".join(bad) +
                         "\nTypical fix: %pip install -U numpy scikit-learn genieclust numba, then restart the kernel.")
    print(f"Pre-flight check passed for all {len(algos)} algorithms.")
    return algos

def run_info(path, a, n_ds, n_alg):
    import platform, sklearn
    try:
        import psutil; ram = f"{psutil.virtual_memory().total / 2**30:.1f} GB"
    except ImportError:
        ram = "n/a (pip install psutil)"
    with open(path, "w") as fh:
        fh.write(f"date: {time.strftime('%Y-%m-%d %H:%M')}\nmachine: {platform.platform()}\n"
                 f"processor: {platform.processor() or platform.machine()}\ncpu cores: {os.cpu_count()}\nRAM: {ram}\n"
                 f"python {platform.python_version()}, numpy {np.__version__}, scikit-learn {sklearn.__version__}\n"
                 f"mNN backend: {'numba' if (a.backend != 'numpy' and _HAS_NUMBA) else 'numpy'}\n"
                 f"datasets: {n_ds}, algorithms: {n_alg}, replicates: {a.R}, subsample frac: {a.frac}, max_n: {a.max_n}\n")

# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["all", "uci", "gagolewski", "synthetic"], default="all")
    ap.add_argument("--gb_root", default="clustering-data-v1")
    ap.add_argument("--batteries", nargs="*", default=["fcps", "wut", "graves", "sipu", "other"])
    ap.add_argument("--gb_max_n_full", type=int, default=None)
    ap.add_argument("--uci_names", nargs="*", default=None)
    ap.add_argument("--uci_cache", default="uci_cache")
    ap.add_argument("--m_sensitivity", action="store_true", help="also run m = 3, 5, 20")
    ap.add_argument("--backend", choices=["auto", "numba", "numpy"], default="auto")
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--max_n", type=int, default=2000)
    ap.add_argument("--out", default="final_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = load(a)
    dataset_table(ds).to_csv(os.path.join(a.out, "datasets_table.csv"), index=False)
    algos = preflight(algorithms(a.m_sensitivity, a.backend))
    if a.backend != "numpy" and _HAS_NUMBA:                    # compile once before timing
        MNNDivisive(2, metric="precomputed", backend="numba").fit(pairwise_distances(np.random.rand(20, 2)))
    run_info(os.path.join(a.out, "run_info.txt"), a, len(ds), len(algos))
    print(f"{len(ds)} datasets x {a.R} replicates x {len(algos)} algorithms")
    t = time.time()
    df = run(ds, algos, R=a.R, frac=a.frac, max_n=a.max_n)
    print(f"finished in {(time.time() - t) / 60:.1f} min")
    df.to_csv(os.path.join(a.out, "main_raw.csv"), index=False)
    df.groupby(["dataset", "method"]).agg(kind=("kind", "first"), ARI_mean=("ARI", "mean"),
        ARI_sd=("ARI", "std"), Sil_mean=("Silhouette", "mean"), time_ms=("time", lambda x: 1000 * x.mean())
        ).round(4).to_csv(os.path.join(a.out, "main_summary.csv"))
    report(df, a.out)
    print("Outputs in", os.path.abspath(a.out))

if __name__ == "__main__":
    if len(sys.argv) == 1:                       # ---- CONFIG for Spyder (F5) ----
        sys.argv += ["--source", "all",
                     "--gb_root", "clustering-data-v1",            # <- path to the Gagolewski suite
                     "--uci_cache", "uci_cache",
                     "--R", "10",
                     "--out", "final_out"]
    main()
