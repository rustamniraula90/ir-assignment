import numpy as np

from cluster import X, df, k, kmeans, purity_score

SEED = 0  # arbitrary but fixed, for a reproducible side-by-side

print("=" * 60)
print("Euclidean distance")
print("=" * 60)
np.random.seed(SEED)
clusters_euclidean, _ = kmeans(X, k, metric="euclidean", init="random")
purity_e, labels_e = purity_score(clusters_euclidean, df["category"], k)

print("=" * 60)
print("Cosine similarity")
print("=" * 60)
np.random.seed(SEED)
clusters_cosine, _ = kmeans(X, k, metric="cosine", init="random")
purity_c, labels_c = purity_score(clusters_cosine, df["category"], k)


def summarise(name, clusters, labels, purity):
    sizes = [int((clusters == c).sum()) for c in range(k)]
    print(f"\n{name}:")
    print(f"  Cluster sizes:  {sizes}")
    print(f"  Cluster labels: {labels}")
    print(f"  Purity:         {purity:.4f}")
    print(f"  Categories represented: {len(set(labels.values()))} / 3")


print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
summarise("Euclidean distance", clusters_euclidean, labels_e, purity_e)
summarise("Cosine similarity", clusters_cosine, labels_c, purity_c)
