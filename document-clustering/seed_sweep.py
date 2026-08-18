import io
import sys

import numpy as np

from cluster import N_INIT, X, df, k, kmeans, purity_score, run_kmeans_multi_restart

N_SEEDS = 100


def sweep(run_once, n_seeds=N_SEEDS):
    purities, categories_covered = [], []
    for seed in range(n_seeds):
        np.random.seed(seed)
        purity, labels = run_once()
        purities.append(purity)
        categories_covered.append(len(set(labels.values())))
    return np.array(purities), categories_covered


def run_naive():
    clusters, _ = kmeans(X, k, init="random", verbose=False)
    return purity_score(clusters, df["category"], k)


def run_current():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    clusters, _, _ = run_kmeans_multi_restart(X, k, n_init=N_INIT)
    sys.stdout = old_stdout
    return purity_score(clusters, df["category"], k)


def summarise(name, purities, categories_covered, n_seeds=N_SEEDS):
    missing = sum(1 for n in categories_covered if n < 3)
    print(f"\n--- {name} ---")
    print(f"Mean purity:                     {purities.mean():.4f}")
    print(f"Std. dev. of purity:             {purities.std():.4f}")
    print(f"Min / max purity:                {purities.min():.4f} / {purities.max():.4f}")
    print(f"Seeds missing a category:        {missing} / {n_seeds}")
    print(f"Seeds covering all 3 categories: {n_seeds - missing} / {n_seeds}")


purities_naive, covered_naive = sweep(run_naive)
summarise("Naive: single random-init k-means (old approach)", purities_naive, covered_naive)

purities_current, covered_current = sweep(run_current)
summarise(
    f"Current: k-means++ + {N_INIT}-restart k-means (used by cluster.py)",
    purities_current,
    covered_current,
)
