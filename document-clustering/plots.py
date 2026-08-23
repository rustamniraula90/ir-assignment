import os

import numpy as np
import matplotlib.pyplot as plt
import umap

from cluster import (
    N_INIT, X, df, k, centroids, cluster_labels, confusion,
    kmeans, purity_score, run_kmeans_multi_restart, vocabulary,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

CATEGORIES = ["Economics", "Entertainment", "Politics"]
CATEGORY_COLOR = {
    "Economics": "#1b9e77",
    "Entertainment": "#d95f02",
    "Politics": "#7570b3",
}

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)

def umap_2d(X, random_state=0):
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=random_state)
    return reducer.fit_transform(X)

XY = umap_2d(X)

def scatter_by_label(ax, xy, point_labels, title):
    for category in CATEGORIES:
        mask = point_labels == category
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=16, alpha=0.75, linewidths=0.3, edgecolors="white",
            color=CATEGORY_COLOR[category], label=category,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

SEED = 0

np.random.seed(SEED)
clusters_euclidean, _ = kmeans(X, k, metric="euclidean", init="random", verbose=False)
purity_e, labels_e = purity_score(clusters_euclidean, df["category"], k)

np.random.seed(SEED)
clusters_cosine, _ = kmeans(X, k, metric="cosine", init="random", verbose=False)
purity_c, labels_c = purity_score(clusters_cosine, df["category"], k)

pred_cat_euclidean = np.array([labels_e[c] for c in clusters_euclidean])
pred_cat_cosine = np.array([labels_c[c] for c in clusters_cosine])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
scatter_by_label(axes[0], XY, pred_cat_euclidean, f"Euclidean distance (purity = {purity_e:.3f})")
scatter_by_label(axes[1], XY, pred_cat_cosine, f"Cosine similarity (purity = {purity_c:.3f})")
axes[1].legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle(
    "K-means cluster assignment by distance metric\n"
    "(same TF-IDF vectors, same seed-0 starting centroids)"
)
fig.tight_layout()
savefig(fig, "dc_distance_metric_scatter.png")

true_category = df["category"].values
predicted_category = df["predicted_category"].values

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
scatter_by_label(axes[0], XY, true_category, "True category")
scatter_by_label(axes[1], XY, predicted_category, "Predicted cluster (majority label)")
axes[1].legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle("Current model (k-means++, 10 restarts, seed 0): true category vs predicted cluster")
fig.tight_layout()
savefig(fig, "dc_cluster_scatter.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for cluster_id in range(k):
    ax = axes[cluster_id]
    centroid = centroids[cluster_id]
    top_idx = np.argsort(centroid)[::-1][:10]
    top_terms = [vocabulary[i] for i in top_idx][::-1]
    top_weights = centroid[top_idx][::-1]
    label = cluster_labels[cluster_id]
    ax.barh(top_terms, top_weights, color=CATEGORY_COLOR[label])
    ax.set_title(f"Cluster {cluster_id} ({label})", fontsize=10)
    ax.set_xlabel("Mean TF-IDF weight")
fig.suptitle("Top 10 TF-IDF terms per cluster centroid")
fig.tight_layout()
savefig(fig, "dc_top_terms_bars.png")

conf_matrix = confusion.reindex(index=CATEGORIES, columns=CATEGORIES).values

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(conf_matrix, cmap="Blues")
ax.set_xticks(range(3))
ax.set_xticklabels(CATEGORIES, rotation=20)
ax.set_yticks(range(3))
ax.set_yticklabels(CATEGORIES)
ax.set_xlabel("Predicted category")
ax.set_ylabel("True category")
ax.set_title("Clustering confusion matrix\n(k-means++, 10 restarts, seed 0)")
for i in range(3):
    for j in range(3):
        value = conf_matrix[i, j]
        color = "white" if value > conf_matrix.max() * 0.5 else "black"
        ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Documents")
fig.tight_layout()
savefig(fig, "dc_confusion_heatmap.png")

def per_class_prf(category):
    tp = ((df["predicted_category"] == category) & (df["category"] == category)).sum()
    predicted_positive = (df["predicted_category"] == category).sum()
    actual_positive = (df["category"] == category).sum()
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

per_class = {category: per_class_prf(category) for category in CATEGORIES}
macro = tuple(np.mean([per_class[c][i] for c in CATEGORIES]) for i in range(3))

groups = CATEGORIES + ["Macro average"]
precision_vals = [per_class[c][0] for c in CATEGORIES] + [macro[0]]
recall_vals = [per_class[c][1] for c in CATEGORIES] + [macro[1]]
f1_vals = [per_class[c][2] for c in CATEGORIES] + [macro[2]]

x = np.arange(len(groups))
width = 0.25

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(x - width, precision_vals, width, label="Precision", color="#377eb8")
ax.bar(x, recall_vals, width, label="Recall", color="#ff7f00")
ax.bar(x + width, f1_vals, width, label="$F_1$", color="#4daf4a")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Per class clustering effectiveness")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
fig.tight_layout()
savefig(fig, "dc_effectiveness_bars.png")

def sweep(run_once, n_seeds=100):
    purities = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        purity, _ = run_once()
        purities.append(purity)
    return np.array(purities)

def run_naive():
    seed_clusters, _ = kmeans(X, k, init="random", verbose=False)
    return purity_score(seed_clusters, df["category"], k)

def run_current():
    seed_clusters, _, _ = run_kmeans_multi_restart(X, k, n_init=N_INIT)
    return purity_score(seed_clusters, df["category"], k)

print("\nRunning the 100-seed stability sweep for the plot (~1 minute)...")
purities_naive = sweep(run_naive)
purities_current = sweep(run_current)

fig, ax = plt.subplots(figsize=(6.5, 4.8))
box = ax.boxplot(
    [purities_naive, purities_current],
    labels=["Plain random init\n(original)", "k-means++, 10 restarts\n(current)"],
    widths=0.5, patch_artist=True, showmeans=True,
)
for patch, color in zip(box["boxes"], ["#e78ac3", "#66c2a5"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

rng = np.random.default_rng(0)
for i, purities in enumerate([purities_naive, purities_current], start=1):
    jitter = rng.uniform(-0.08, 0.08, size=len(purities))
    ax.scatter(np.full(len(purities), i) + jitter, purities, s=10, alpha=0.5, color="black")

ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Random 3-way baseline (0.333)")
ax.set_ylabel("Purity")
ax.set_ylim(bottom=0.30)
ax.set_title("Purity across 100 random seeds")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
fig.tight_layout()
savefig(fig, "dc_stability_boxplot.png")

print("\nDone.")
