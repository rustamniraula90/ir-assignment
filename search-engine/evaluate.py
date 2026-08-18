import math
import statistics
import time

from indexer import build_search_index, clean_and_tokenize, db_publications, search_publications

QUERIES = [
    "community nursing intervention",
    "physical activity older adults",
    "health inequalities",
    "mental health young people",
    "bariatric surgery",
]

QUERY_TERM_MATCH_FRACTION = 0.6

def average_precision(judgments):
    """Average of precision@k at each rank where a relevant result appears."""
    hits = 0
    precisions = []
    for k, relevant in enumerate(judgments, start=1):
        if relevant:
            hits += 1
            precisions.append(hits / k)
    return sum(precisions) / hits if hits else 0.0


def relevant_urls(query):
    query_terms = set(clean_and_tokenize(query))
    if not query_terms:
        return set()
    threshold = max(1, math.ceil(len(query_terms) * QUERY_TERM_MATCH_FRACTION))

    relevant = set()
    for pub in db_publications.find({}, {"url": 1, "title": 1, "abstract": 1, "keywords": 1}):
        text = " ".join([
            pub.get("title", ""),
            pub.get("abstract", ""),
            " ".join(pub.get("keywords", [])),
        ])
        doc_terms = set(clean_and_tokenize(text))
        if len(query_terms & doc_terms) >= threshold:
            relevant.add(pub["url"])
    return relevant


def confusion_matrix(retrieved, relevant, corpus_size):
    retrieved = set(retrieved)
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    tn = corpus_size - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def print_confusion_matrix(cm):
    print(f"{'':16s}{'Retrieved':>12s}{'Not retrieved':>16s}")
    print(f"{'Relevant':16s}{'TP=' + str(cm['tp']):>12s}{'FN=' + str(cm['fn']):>16s}")
    print(f"{'Not relevant':16s}{'FP=' + str(cm['fp']):>12s}{'TN=' + str(cm['tn']):>16s}")


def evaluate_effectiveness(index_data):
    corpus_size = len(index_data["docs"])
    print(f"{'Query':32s} {'Prec.':>6s} {'Recall':>7s} {'F1':>6s} {'P@5':>6s} {'AP':>6s}")

    aps = []
    matrices = []
    for query in QUERIES:
        relevant = relevant_urls(query)
        results = search_publications(index_data, query, top_n=10)
        retrieved_urls = [result["url"] for result in results]
        judgments = [url in relevant for url in retrieved_urls]

        relevant_retrieved = sum(judgments)
        precision = relevant_retrieved / len(judgments) if judgments else 0.0
        p_at_5 = sum(judgments[:5]) / 5

        recall = relevant_retrieved / len(relevant) if relevant else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        ap = average_precision(judgments)
        aps.append(ap)
        matrices.append((query, confusion_matrix(retrieved_urls, relevant, corpus_size)))

        print(f"{query:32s} {precision:6.2f} {recall:7.3f} {f1:6.3f} {p_at_5:6.2f} {ap:6.3f}")

    print(f"\nMAP = {sum(aps) / len(aps):.3f}")

    print("\nConfusion matrices (relevant/retrieved over the full corpus):")
    for query, cm in matrices:
        print(f"\n{query!r}")
        print_confusion_matrix(cm)


def evaluate_efficiency(index_data, n_repeats=5):
    queries = QUERIES * n_repeats
    timings_ms = []

    for query in queries:
        start = time.perf_counter()
        search_publications(index_data, query)
        timings_ms.append((time.perf_counter() - start) * 1000)

    timings_ms.sort()
    p95 = timings_ms[int(0.95 * len(timings_ms)) - 1]
    print(f"\nQuery response time over {len(timings_ms)} calls:")
    print(f"  mean   = {statistics.mean(timings_ms):.3f} ms")
    print(f"  median = {statistics.median(timings_ms):.3f} ms")
    print(f"  p95    = {p95:.3f} ms")


if __name__ == "__main__":
    build_start = time.perf_counter()
    index_data = build_search_index()
    build_ms = (time.perf_counter() - build_start) * 1000
    print(f"Indexed {len(index_data['docs'])} publications in {build_ms:.0f} ms.\n")

    evaluate_effectiveness(index_data)
    evaluate_efficiency(index_data)
