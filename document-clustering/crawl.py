import csv
import re
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-clustering-assignment/1.0)"}
TIMEOUT = 15
TARGET_PER_CATEGORY = 100
MIN_CHARS = 250
MAX_CHARS = 700

FEEDS = {
    "Economics": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.theguardian.com/business/economics/rss",
        "https://www.theguardian.com/uk/business/rss",
        "https://www.theguardian.com/business/rss",
        "https://www.theguardian.com/money/rss",
        "https://www.theguardian.com/business/banking/rss",
        "https://www.theguardian.com/business/stock-markets/rss",
        "https://www.theguardian.com/business/companies/rss",
        "https://www.theguardian.com/business/retail/rss",
        "https://www.theguardian.com/business/eurozone/rss",
    ],
    "Entertainment": [
        "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
        "https://www.theguardian.com/film/rss",
        "https://www.theguardian.com/music/rss",
        "https://www.theguardian.com/tv-and-radio/rss",
        "https://www.theguardian.com/stage/rss",
        "https://www.theguardian.com/culture/rss",
        "https://www.theguardian.com/books/rss",
        "https://www.theguardian.com/games/rss",
        "https://www.theguardian.com/artanddesign/rss",
    ],
    "Politics": [
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/us-news/us-politics/rss",
        "https://www.theguardian.com/world/europe-news/rss",
        "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
        "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    ],
}

def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text

def parse_feed_links(xml_text):
    links = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.iter("item"):
            link_el = item.find("link")
            if link_el is not None and link_el.text:
                links.append(link_el.text.strip())
    except ET.ParseError:
        links = re.findall(r"<link>(https?://[^<]+)</link>", xml_text)
    return links

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*external\b", "", text)
    return text

def truncate_to_sentence(text, max_chars):
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_stop = max(truncated.rfind(". "), truncated.rfind(".”"))
    if last_stop > 80:
        return truncated[: last_stop + 1]
    return truncated.rsplit(" ", 1)[0] + "..."

def extract_paragraph(html, url):
    soup = BeautifulSoup(html, "html.parser")

    for fig in soup.find_all(["figure", "figcaption"]):
        fig.decompose()

    if "theguardian.com" in url:
        paras = soup.select("div#maincontent p")
    else:
        article = soup.find("article") or soup
        paras = article.find_all("p")

    texts = []
    for p in paras:
        t = clean_text(p.get_text(" ", strip=True))
        if len(t) < 40:
            continue
        if t.lower().startswith(("follow ", "share this", "related:", "read more")):
            continue
        if "@theguardian.com" in t or "@bbc" in t.lower():
            continue
        if any(phrase in t.lower() for phrase in (
            "bbc app", "bbc sounds", "recommended if you like", "up next",
            "listen to bbc podcasts",
        )):
            continue
        texts.append(t)

    if not texts:
        return None

    combined = ""
    for t in texts:
        candidate = (combined + " " + t).strip() if combined else t
        if len(candidate) > MAX_CHARS and combined:
            break
        combined = candidate
        if len(combined) >= MIN_CHARS:
            break

    if len(combined) < MIN_CHARS:
        for t in texts:
            if len(t) >= 120:
                combined = t
                break
        else:
            combined = texts[0]

    combined = truncate_to_sentence(combined, MAX_CHARS + 150)

    return combined if len(combined) >= 120 else None

def collect_category(category, feed_urls, target, seen_texts):
    rows = []
    seen_links = set()

    for feed_url in feed_urls:
        if len(rows) >= target:
            break
        try:
            xml_text = fetch(feed_url)
        except requests.RequestException as e:
            print(f"  [skip feed] {feed_url}: {e}")
            continue

        links = parse_feed_links(xml_text)
        print(f"  feed {feed_url}: {len(links)} items")

        for link in links:
            if len(rows) >= target:
                break
            if link in seen_links:
                continue
            seen_links.add(link)

            if any(seg in link for seg in ("/live/", "/av/", "/videos/", "/in-pictures/", "/gallery/")):
                continue

            try:
                html = fetch(link)
            except requests.RequestException as e:
                print(f"    [skip article] {link}: {e}")
                continue

            text = extract_paragraph(html, link)
            if text and text not in seen_texts:
                rows.append({"text": text, "category": category})
                seen_texts.add(text)

            time.sleep(0.3)

    return rows

def main():
    all_rows = []
    seen_texts = set()
    for category, feed_urls in FEEDS.items():
        print(f"Collecting category: {category}")
        rows = collect_category(category, feed_urls, TARGET_PER_CATEGORY, seen_texts)
        print(f"  -> collected {len(rows)} documents")
        all_rows.extend(rows)

    with open("data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal documents written: {len(all_rows)}")

if __name__ == "__main__":
    main()
