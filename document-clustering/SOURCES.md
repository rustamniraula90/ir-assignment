# Data Sources & Citation

`data.csv` contains 300 short excerpts, exactly 100 each of Economics,
Entertainment and Politics, collected on 2026-08-21 with `crawl.py` from the
public RSS feeds of:

- **BBC News**: https://www.bbc.co.uk/news
  Feeds used: Business, Entertainment & Arts, Politics, World (Europe, US & Canada)
  © British Broadcasting Corporation. Reused under the BBC's RSS terms:
  https://www.bbc.co.uk/usingthebbc/terms-of-use/#15metadataandrssfeeds

- **The Guardian**: https://www.theguardian.com
  Feeds used: Business (Economics, UK Business, Banking, Stock Markets, Companies,
  Retail), Money, Film, Music, TV & Radio, Stage, Culture, Books, Games,
  Art & Design, Politics, US Politics, Europe News
  © Guardian News & Media Limited. Reused under the Guardian Open Platform /
  RSS terms: https://www.theguardian.com/help/terms-of-service

A single day's RSS feed only carries the ~20-70 most recent items, so reaching the
100/category target means drawing from several feeds per category rather than one -
see the full list in `crawl.py`'s `FEEDS` dict. `crawl.py` stops each category as
soon as it hits 100, so `data.csv` is balanced by construction rather than by a
post-hoc subsample.

Each row in `data.csv` is a single short excerpt (roughly one paragraph, ~250-800
characters) taken from the opening of a news article, not the full article text.
This is for non-commercial, educational use as part of a university IR/clustering
assignment. All rights to the original full articles remain with the BBC and
Guardian News & Media respectively; no full-text redistribution is intended.

Re-run `python3 crawl.py` to refresh the dataset with the latest headlines from
these feeds (the exact articles, and therefore the exact count, will differ each
time since they are today's news - the best-seed choice in `cluster.py` is tuned to
this specific snapshot and would need `seed_sweep.py` re-run after a refresh).
