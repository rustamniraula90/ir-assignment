# IR Assignment

Two independent projects for an Information Retrieval module.

## `search-engine/`

Vertical search engine over Coventry University's Centre for Healthcare and Community
Transformation PurePortal publications.

- `crawler.py` — crawls PurePortal into MongoDB (Cloudflare-protected, needs a real/virtual display)
- `indexer.py` — per-field weighted TF-IDF (title/keywords/abstract), cosine similarity ranking
- `api.py` — FastAPI search UI, reindexes from MongoDB hourly
- `evaluate.py` — precision/recall/F1/P@5/MAP + confusion matrix against the crawled corpus

```bash
pip install -r search-engine/requirements.txt
python search-engine/crawler.py          # populate MongoDB (mongodb://localhost:27017)
uvicorn api:app --reload --app-dir search-engine   # http://127.0.0.1:8000
python search-engine/evaluate.py         # evaluation numbers
```

## `document-clustering/`

K-means clustering of ~450 BBC/Guardian news excerpts into Economics, Entertainment and Politics.

- `crawl.py` — collects excerpts into `data.csv` (sources/citations in `SOURCES.md`)
- `cluster.py` — TF-IDF + K-means from scratch, assigns a new document to a cluster
- `seed_sweep.py`, `compare_distance_metrics.py` — stability/metric evaluation used in the report

```bash
pip install numpy pandas nltk beautifulsoup4 requests
python document-clustering/crawl.py      # optional: refresh data.csv with today's news
python document-clustering/cluster.py    # cluster + classify
```
