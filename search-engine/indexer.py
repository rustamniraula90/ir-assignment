import math
import re

from collections import Counter, defaultdict
from pymongo import MongoClient

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["pureportal_search_engine"]
db_publications = db["publications"]

FIELD_WEIGHTS = {"title": 0.5, "keywords": 0.3, "abstract": 0.2}

def clean_and_tokenize(text):
    text = text.lower()
    words = re.split(r"[^a-z]+", text)
    return [STEMMER.stem(w) for w in words if w and w not in STOP_WORDS]

def term_frequency(tokens):
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}

def build_field_index(tokenized, n_docs):
    doc_freq = Counter()
    for tokens in tokenized.values():
        for term in set(tokens):
            doc_freq[term] += 1
    idf = {term: math.log(n_docs / df) for term, df in doc_freq.items()} if n_docs else {}

    inverted_index = defaultdict(dict)
    magnitude = {}
    for doc_id, tokens in tokenized.items():
        tf = term_frequency(tokens)
        vector = {term: weight * idf[term] for term, weight in tf.items()}
        magnitude[doc_id] = math.sqrt(sum(w * w for w in vector.values()))
        for term, weight in vector.items():
            inverted_index[term][doc_id] = weight

    return {"index": inverted_index, "magnitude": magnitude, "idf": idf}

def build_search_index():
    tokenized = {field: {} for field in FIELD_WEIGHTS}
    docs = {}

    for pub in db_publications.find({}):
        doc_id = pub["_id"]

        tokenized["title"][doc_id] = clean_and_tokenize(pub.get("title", ""))
        tokenized["abstract"][doc_id] = clean_and_tokenize(pub.get("abstract", ""))
        tokenized["keywords"][doc_id] = clean_and_tokenize(" ".join(pub.get("keywords", [])))

        docs[doc_id] = {
            "title": pub.get("title", ""),
            "url": pub.get("url", ""),
            "authors": pub.get("authors", []),
            "department_authors": pub.get("department_authors", []),
            "publication_year": pub.get("publication_year"),
            "keywords": pub.get("keywords", []),
        }

    n_docs = len(docs)
    fields = {field: build_field_index(tokenized[field], n_docs) for field in FIELD_WEIGHTS}

    return {"fields": fields, "docs": docs}

def field_scores(field_index, tokens):
    idf = field_index["idf"]
    inverted_index = field_index["index"]
    magnitude = field_index["magnitude"]

    qtf = term_frequency(tokens)
    query_vector = {term: weight * idf[term] for term, weight in qtf.items() if term in idf}

    magnitude_query = math.sqrt(sum(w * w for w in query_vector.values()))
    if magnitude_query == 0:
        return {}

    dot_products = defaultdict(float)
    for term, q_weight in query_vector.items():
        for doc_id, d_weight in inverted_index.get(term, {}).items():
            dot_products[doc_id] += q_weight * d_weight

    return {
        doc_id: dot / (magnitude_query * magnitude[doc_id])
        for doc_id, dot in dot_products.items()
        if magnitude[doc_id] > 0
    }

def search_publications(index_data, query, top_n=10):
    tokens = clean_and_tokenize(query)
    if not tokens:
        return []

    per_field = {field: field_scores(index_data["fields"][field], tokens) for field in FIELD_WEIGHTS}

    combined = defaultdict(float)
    for field, weight in FIELD_WEIGHTS.items():
        for doc_id, score in per_field[field].items():
            combined[doc_id] += weight * score

    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top_n]

    results = []
    for doc_id, score in ranked:
        result = dict(index_data["docs"][doc_id])
        result["score"] = score
        result["field_scores"] = {
            field: per_field[field].get(doc_id, 0.0) for field in FIELD_WEIGHTS
        }
        results.append(result)
    return results
