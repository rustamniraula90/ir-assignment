import re
import math
import time

from collections import Counter

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data.csv")

def clean_and_tokenize(text):
    text = text.lower()
    words = re.split(r"[^a-z]+", text)
    return [STEMMER.stem(w) for w in words if w and w not in STOP_WORDS]

df["tokens"] = df["text"].apply(clean_and_tokenize)

N = len(df)

print("Number of documents:", N)


# ============================================================
# 2. BUILD VOCABULARY
# ============================================================

vocabulary = set()

for tokens in df["tokens"]:
    vocabulary.update(tokens)

vocabulary = sorted(vocabulary)

word_to_index = {
    word: i
    for i, word in enumerate(vocabulary)
}

print("Vocabulary size:", len(vocabulary))


# ============================================================
# 3. CALCULATE DOCUMENT FREQUENCY (DF)
# ============================================================

document_frequency = {}

for tokens in df["tokens"]:

    # Count a word only once per document
    unique_words = set(tokens)

    for word in unique_words:
        document_frequency[word] = (
            document_frequency.get(word, 0) + 1
        )


# ============================================================
# 4. CALCULATE IDF
# ============================================================

idf = {}

for word in vocabulary:

    df_word = document_frequency[word]

    idf[word] = math.log(N / df_word)


# ============================================================
# 5. CALCULATE TF
# ============================================================

def calculate_tf(tokens):

    tf = {}

    total_words = len(tokens)

    if total_words == 0:
        return tf

    for word in tokens:
        tf[word] = tf.get(word, 0) + 1

    for word in tf:
        tf[word] /= total_words

    return tf


# ============================================================
# 6. CREATE TF-IDF MATRIX
# ============================================================

X = np.zeros(
    (N, len(vocabulary))
)

for i, tokens in enumerate(df["tokens"]):

    tf = calculate_tf(tokens)

    for word, tf_value in tf.items():

        j = word_to_index[word]

        X[i, j] = tf_value * idf[word]


print("TF-IDF matrix shape:", X.shape)


# ============================================================
# 7. DISTANCE / SIMILARITY FUNCTIONS
# ============================================================

def cosine_similarity(x, centroid):

    numerator = np.dot(x, centroid)

    denominator = (
        np.linalg.norm(x) * np.linalg.norm(centroid)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def euclidean_distance(x, centroid):
    return np.linalg.norm(x - centroid)


# ============================================================
# 8. ASSIGN DOCUMENTS TO CLUSTERS
# ============================================================

def assign_clusters(X, centroids, metric="cosine"):

    clusters = []

    for x in X:

        if metric == "cosine":
            # Higher similarity = closer, so pick the largest value.
            scores = [cosine_similarity(x, c) for c in centroids]
            closest_cluster = np.argmax(scores)
        elif metric == "euclidean":
            # Smaller distance = closer, so pick the smallest value.
            scores = [euclidean_distance(x, c) for c in centroids]
            closest_cluster = np.argmin(scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        clusters.append(closest_cluster)

    return np.array(clusters)


# ============================================================
# 9. UPDATE CENTROIDS
# ============================================================

def update_centroids(X, clusters, k):

    new_centroids = np.zeros(
        (k, X.shape[1])
    )

    for cluster in range(k):

        cluster_points = X[
            clusters == cluster
        ]

        if len(cluster_points) > 0:
            new_centroids[cluster] = np.mean(
                cluster_points,
                axis=0
            )
        else:
            # Empty cluster: re-seed it with a random document instead of
            # leaving a zero centroid, which would never attract anything.
            random_index = np.random.randint(
                0,
                X.shape[0]
            )

            new_centroids[cluster] = X[random_index]

    return new_centroids


# ============================================================
# 9b. K-MEANS++ INITIALISATION
# ============================================================

def kmeans_plusplus_init(X, k, metric="cosine"):

    n_samples = X.shape[0]

    first_index = np.random.randint(n_samples)
    centroids = [X[first_index]]

    for _ in range(1, k):

        if metric == "cosine":
            nearest_dist = np.array([
                min(1 - cosine_similarity(x, c) for c in centroids)
                for x in X
            ])
        else:
            nearest_dist = np.array([
                min(euclidean_distance(x, c) for c in centroids)
                for x in X
            ])

        weights = nearest_dist ** 2
        total_weight = weights.sum()

        if total_weight > 0:
            probabilities = weights / total_weight
        else:
            # avoid divide by zero: fall back to a uniform draw
            probabilities = np.full(n_samples, 1 / n_samples)

        next_index = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_index])

    return np.array(centroids)


# ============================================================
# 10. K-MEANS
# ============================================================

def kmeans(X, k, metric="cosine", max_iterations=100, init="kmeans++", verbose=True):

    if init == "kmeans++":
        centroids = kmeans_plusplus_init(X, k, metric)
    elif init == "random":
        random_indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[random_indices].copy()
    else:
        raise ValueError(f"Unknown init: {init}")

    for iteration in range(max_iterations):

        clusters = assign_clusters(
            X,
            centroids,
            metric
        )

        new_centroids = update_centroids(
            X,
            clusters,
            k
        )

        centroid_movement = np.linalg.norm(
            new_centroids - centroids
        )

        if verbose:
            print(
                f"Iteration {iteration + 1}: "
                f"centroid movement = {centroid_movement:.6f}"
            )

        if np.allclose(
            centroids,
            new_centroids
        ):

            if verbose:
                print(
                    "\nConverged at iteration:",
                    iteration + 1
                )

            centroids = new_centroids

            break

        centroids = new_centroids

    return clusters, centroids


# ============================================================
# 10b. MULTI-RESTART: RUN K-MEANS SEVERAL TIMES, KEEP LOWEST INERTIA
# ============================================================

def compute_inertia(X, clusters, centroids, metric="cosine"):

    if metric == "cosine":
        return sum(
            1 - cosine_similarity(x, centroids[c])
            for x, c in zip(X, clusters)
        )

    return sum(
        euclidean_distance(x, centroids[c]) ** 2
        for x, c in zip(X, clusters)
    )


def run_kmeans_multi_restart(X, k, metric="cosine", n_init=10, max_iterations=100):

    best_clusters, best_centroids, best_inertia = None, None, None

    for restart in range(n_init):

        clusters, centroids = kmeans(
            X, k, metric=metric, max_iterations=max_iterations, verbose=False
        )
        inertia = compute_inertia(X, clusters, centroids, metric)

        print(f"  Restart {restart + 1}/{n_init}: inertia = {inertia:.4f}")

        if best_inertia is None or inertia < best_inertia:
            best_clusters, best_centroids, best_inertia = clusters, centroids, inertia

    return best_clusters, best_centroids, best_inertia


# ============================================================
# 11. RUN K-MEANS WITH K = 3
# ============================================================

np.random.seed(0)
N_INIT = 10

k = 3

_start_time = time.time()

print(f"\nRunning k-means++ with {N_INIT} restarts, keeping the lowest-inertia one:")
clusters, centroids, _best_inertia = run_kmeans_multi_restart(X, k, n_init=N_INIT)

_elapsed = time.time() - _start_time
print(f"\nK-means clustering time ({N_INIT} restarts): {_elapsed * 1000:.1f} ms")


# ============================================================
# 12. ADD CLUSTER ASSIGNMENTS TO DATAFRAME
# ============================================================

df["cluster"] = clusters


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\nCluster sizes:")

print(
    df["cluster"].value_counts().sort_index()
)


print("\nSample results:")

print(
    df[
        ["text", "category", "cluster"]
    ].head(20)
)


# ============================================================
# 14. SHOW MOST IMPORTANT WORDS IN EACH CENTROID
# ============================================================

print("\nTop words in each centroid:")

for cluster in range(k):

    centroid = centroids[cluster]

    top_indices = np.argsort(centroid)[::-1][:10]

    top_words = [
        vocabulary[i]
        for i in top_indices
    ]

    print(
        f"Cluster {cluster}:",
        top_words
    )


# ============================================================
# 15. LABEL EACH CLUSTER WITH ITS MAJORITY CATEGORY
# ============================================================

def label_clusters(clusters, categories, k):
    labels = {}

    for cluster in range(k):
        subset = categories[clusters == cluster]
        labels[cluster] = subset.mode()[0] if len(subset) > 0 else "unknown"

    return labels


cluster_labels = label_clusters(clusters, df["category"], k)

print("\nCluster -> majority category:")

for cluster in range(k):
    print(f"  Cluster {cluster}: {cluster_labels[cluster]}")


# ============================================================
# 15b. EVALUATE CLUSTERING AGAINST THE KNOWN GROUND TRUTH
# ============================================================

def purity_score(clusters, categories, k):
    labels = label_clusters(clusters, categories, k)
    predicted = pd.Series(clusters).map(labels)
    purity = (predicted.values == categories.values).mean()
    return purity, labels


df["predicted_category"] = df["cluster"].map(cluster_labels)

confusion = pd.crosstab(
    df["category"],
    df["predicted_category"],
    rownames=["True"],
    colnames=["Predicted"],
)

print("\nConfusion matrix (true category vs. predicted category):")
print(confusion)

purity, _ = purity_score(clusters, df["category"], k)
print(f"\nOverall accuracy / purity: {purity:.4f}")

print("\nPer-category precision / recall / F1:")

categories = sorted(df["category"].unique())

for category in categories:

    true_positive = (
        (df["predicted_category"] == category) & (df["category"] == category)
    ).sum()

    predicted_positive = (df["predicted_category"] == category).sum()
    actual_positive = (df["category"] == category).sum()

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(
        f"  {category:15s} precision={precision:.4f}  "
        f"recall={recall:.4f}  f1={f1:.4f}"
    )


# ============================================================
# 16. CLASSIFY A NEW DOCUMENT ENTERED BY THE USER
# ============================================================

def vectorize_document(text):
    tokens = clean_and_tokenize(text)
    tf = calculate_tf(tokens)
    vector = np.zeros(len(vocabulary))

    for word, tf_value in tf.items():
        if word in word_to_index:
            j = word_to_index[word]
            vector[j] = tf_value * idf[word]

    return vector


def predict_cluster(text):
    vector = vectorize_document(text)

    similarities = [
        cosine_similarity(vector, centroid)
        for centroid in centroids
    ]

    predicted_cluster = int(np.argmax(similarities))

    return predicted_cluster, similarities


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Enter a document to classify it into one of the clusters.")
    print("Leave the line empty (or type 'quit') to stop.")
    print("=" * 60)

    while True:

        try:
            user_input = input("\nNew document: ").strip()
        except EOFError:
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        predicted_cluster, similarities = predict_cluster(user_input)

        print(
            f"-> Suggested cluster: {predicted_cluster} "
            f"(likely category: {cluster_labels[predicted_cluster]})"
        )

        print(
            "   Similarity to each cluster: "
            + ", ".join(
                f"cluster {c}={s:.4f}" for c, s in enumerate(similarities)
            )
        )
