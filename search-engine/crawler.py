import time
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup

from seleniumbase import SB
from selenium.common.exceptions import WebDriverException
from urllib3.exceptions import MaxRetryError, NewConnectionError

import requests
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from pymongo import MongoClient


class BrowserDiedError(Exception):
    pass


BROWSER_FATAL_EXCEPTIONS = (WebDriverException, MaxRetryError, NewConnectionError, ConnectionRefusedError, OSError)

MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["pureportal_search_engine"]

db_urls = db["urls"]
db_publications = db["publications"]

db_urls.create_index("url", unique=True)
db_publications.create_index("url", unique=True)

print("Connected:", db.name, "| collections:", db.list_collection_names())

BASE_URL = "https://pureportal.coventry.ac.uk"
SEED_URL = "https://pureportal.coventry.ac.uk/en/organisations/centre-for-healthcare-and-community-transformation/persons/"

DEFAULT_CRAWL_DELAY_SECONDS = 5
MAX_RETRIES = 5 
RECRAWL_INTERVAL_DAYS = 7

PRIORITY = {
    "profiles": 0,
    "profile_publications": 1,
    "publication": 2,
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def refresh_cf_clearance(sb):
    try:
        sb.uc_open_with_reconnect(BASE_URL + "/en/", reconnect_time=4)
        sb.uc_gui_click_captcha()
        sb.sleep(2)
        for c in sb.driver.get_cookies():
            session.cookies.set(c["name"], c["value"], domain=c["domain"])
        session.headers["User-Agent"] = sb.driver.execute_script("return navigator.userAgent")
    except BROWSER_FATAL_EXCEPTIONS as e:
        raise BrowserDiedError(f"Browser died during cf_clearance refresh: {e}") from e


def is_challenge_page(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    title = soup.title.get_text(strip=True).lower() if soup.title else ""
    if title.startswith("just a moment"):
        return True

    if soup.select_one("#challenge-running, #challenge-form, #cf-challenge-running"):
        return True

    return False


def fetch_page(sb, url, max_challenge_retries=2):
    response = session.get(url, timeout=30)

    if response.status_code == 200 and not is_challenge_page(response.text):
        return response.text

    if response.status_code != 200:
        print(f"requests got status {response.status_code} for {url}, falling back to browser")
    else:
        print(f"Cloudflare challenge detected for {url}, falling back to browser")

    for attempt in range(1, max_challenge_retries + 1):
        try:
            sb.uc_open_with_reconnect(url, reconnect_time=4)
            sb.uc_gui_click_captcha()
            sb.sleep(2)
            html_content = sb.get_page_source()
        except BROWSER_FATAL_EXCEPTIONS as e:
            raise BrowserDiedError(f"Browser died while fetching {url}: {e}") from e

        if not is_challenge_page(html_content):
            for c in sb.driver.get_cookies():
                session.cookies.set(c["name"], c["value"], domain=c["domain"])
            return html_content

        print(f"Still a challenge page on browser attempt {attempt} for {url}")
        sb.sleep(3)

    raise RuntimeError(f"Could not get past Cloudflare challenge for {url}")


def get_robot_parser(base_url):
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        resp = session.get(robots_url, timeout=15)
        resp.raise_for_status()
        rp.parse(resp.text.splitlines())
    except Exception as e:
        print(f"Could not fetch robots.txt ({e}); falling back to known rules")
        rp = None
    return rp


def can_fetch(rp, url):
    if rp is None:
        return True

    return rp.can_fetch("*", url)


def get_crawl_delay(rp):
    if rp is not None:
        delay = rp.crawl_delay("*")
        if delay:
            return float(delay)
    return DEFAULT_CRAWL_DELAY_SECONDS


def enqueue(url_type, url, source_member=None):
    now = datetime.now(timezone.utc)
    update = {
        "$setOnInsert": {
            "url": url,
            "next_crawl_date": now,
            "url_type": url_type,
            "priority": PRIORITY[url_type],
            "retries": 0,
        }
    }
    if source_member:
        update["$addToSet"] = {"source_members": source_member}
    db_urls.update_one({"url": url}, update, upsert=True)


def get_next_job():
    return db_urls.find_one_and_update(
        {"next_crawl_date": {"$lt": datetime.now(timezone.utc)}},
        {"$set": {"next_crawl_date": datetime.now(timezone.utc) + timedelta(minutes=10)}},
        sort=[("priority", 1), ("next_crawl_date", 1)],
    )


def mark_success(url):
    db_urls.update_one(
        {"url": url},
        {"$set": {"next_crawl_date": datetime.now(timezone.utc) + timedelta(days=RECRAWL_INTERVAL_DAYS), "retries": 0}},
    )


def schedule_retry(url):
    doc = db_urls.find_one({"url": url})
    retries = (doc.get("retries", 0) if doc else 0) + 1
    if retries > MAX_RETRIES:
        print(f"Giving up on {url} for now after {retries} failed attempts (retrying next week)")
        db_urls.update_one({"url": url}, {"$set": {
            "next_crawl_date": datetime.now(timezone.utc) + timedelta(days=RECRAWL_INTERVAL_DAYS),
            "retries": 0,
        }})
    else:
        db_urls.update_one({"url": url}, {"$set": {
            "next_crawl_date": datetime.now(timezone.utc) + timedelta(hours=1),
            "retries": retries,
        }})


def parse_authors(soup):
    authors = []
    for li in soup.select("ul.relations.persons li, p.relations.persons"):
        a_tag = li.select_one("a.link.person")
        if a_tag and a_tag.has_attr("href"):
            name = a_tag.get_text(strip=True)
            profile_url = urljoin(BASE_URL, a_tag["href"])
            authors.append({"name": name, "profile_url": profile_url})
        else:
            text = li.get_text(strip=True).lstrip(",").strip()
            if text:
                authors.append({"name": text, "profile_url": None})
    return authors


def extract_content(html, url_type, current_url=""):
    soup = BeautifulSoup(html, "html.parser")

    match url_type:
        case "profiles":
            members = []
            urls = []
            for a in soup.select("ul.grid-results li.grid-result-item h3.title a.link.person"):
                if not a.has_attr("href"):
                    continue
                profile_url = urljoin(BASE_URL, a["href"])
                name = a.get_text(strip=True)
                member = {"member_url": profile_url, "member_name": name}

                members.append({"url": profile_url, "name": name})

                pub_url = profile_url.rstrip("/") + "/publications/"
                urls.append((pub_url, "profile_publications", member))

            next_page = soup.select_one("nav.pages ul li.next a.nextLink")
            if next_page and next_page.has_attr("href"):
                urls.append((urljoin(BASE_URL, next_page["href"]), "profiles", None))

            return {"members": members, "next_urls": urls}

        case "profile_publications":
            urls = []
            for a in soup.select("ul.list-results li.list-result-item h3.title a[href]"):
                pub_url = urljoin(BASE_URL, a["href"])
                urls.append((pub_url, "publication", None))

            next_page = soup.select_one("nav.pages ul li.next a.nextLink")
            if next_page and next_page.has_attr("href"):
                urls.append((urljoin(BASE_URL, next_page["href"]), "profile_publications", None))

            return {"next_urls": urls}

        case "publication":
            def get_text_safe(selector):
                el = soup.select_one(selector)
                return el.get_text(strip=True) if el else ""

            title = get_text_safe(".introduction h1 span")
            authors = parse_authors(soup)

            date_text = get_text_safe("tr.status .date")
            pub_year = None
            for token in date_text.replace(",", " ").split():
                if token.isdigit() and len(token) == 4:
                    pub_year = int(token)
                    break

            keywords = []
            for span in soup.select(".keyword-group li.userdefined-keyword span, .content-concept-list li span.concept"):
                kw = span.get_text(strip=True)
                if kw:
                    keywords.append(kw)

            return {
                "url": current_url,
                "title": title,
                "abstract": get_text_safe(".publication-content .rendering_researchoutput_abstractportal .textblock"),
                "publication_date_text": date_text,
                "publication_year": pub_year,
                "authors": authors,
                "keywords": keywords,
                "language": get_text_safe("tr.language td"),
                "last_crawled": datetime.now(timezone.utc),
            }

    return None



def process_job(sb, rp, crawl_delay, document):
    url = document["url"]
    url_type = document["url_type"]
    source_members = document.get("source_members", [])

    if not can_fetch(rp, url):
        print(f"Blocked by robots.txt: {url}")
        mark_success(url)
        return

    try:
        print("crawling", url)
        html_content = fetch_page(sb, url)
        content = extract_content(html_content, url_type, url)

        if url_type == "profiles":
            if content and (content["members"] or content["next_urls"]):
                for next_url, next_type, member in content["next_urls"]:
                    enqueue(next_type, next_url, member)
                mark_success(url)
            else:
                schedule_retry(url)

        elif url_type == "profile_publications":
            if content and content["next_urls"]:
                for next_url, next_type, _ in content["next_urls"]:
                    for member in (source_members or [None]):
                        enqueue(next_type, next_url, member)
                mark_success(url)
            else:
                if content is not None:
                    mark_success(url)  # valid empty last page, not a failure
                else:
                    schedule_retry(url)

        elif url_type == "publication":
            if content and content.get("title"):
                update = {"$set": content}
                if source_members:
                    update["$addToSet"] = {"department_authors": {"$each": source_members}}
                db_publications.update_one({"url": content["url"]}, update, upsert=True)
                mark_success(url)
            else:
                schedule_retry(url)

    except BrowserDiedError:
        raise
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        schedule_retry(url)

    time.sleep(crawl_delay)


def crawl(seed_url=SEED_URL):
    if not db_urls.find_one({"url": seed_url}):
        db_urls.insert_one({
            "url": seed_url,
            "next_crawl_date": datetime.now(timezone.utc),
            "url_type": "profiles",
            "priority": PRIORITY["profiles"],
            "retries": 0,
        })
    while True:
        try:
            with SB(uc=True, test=True, xvfb=True) as sb:
                refresh_cf_clearance(sb)
                rp = get_robot_parser(BASE_URL)
                crawl_delay = get_crawl_delay(rp)

                while True:
                    document = get_next_job()
                    if not document:
                        time.sleep(60)
                        continue

                    process_job(sb, rp, crawl_delay, document)

        except BrowserDiedError as e:
            print(f"Browser session died ({e}). Creating new session in 10 seconds.")
            time.sleep(10)
            continue


if __name__ == "__main__":
    crawl()
