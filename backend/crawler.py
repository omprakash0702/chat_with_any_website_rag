# backend/crawler.py

import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from collections import deque
import logging
from backend.utils import obey_robots, normalize_url

logger = logging.getLogger(__name__)


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(base_url, href)
        full = normalize_url(full)
        out.add(full)
    return out


def crawl(
    start_url,
    max_pages=50,
    allowed_domains=None,
    delay=1.0,
    url_prefix=None,
):
    """
    BFS crawl, but **restricted** to URLs that start with `url_prefix`
    (or, if not provided, restricted to same domain).

    This prevents wandering off into portals/categories/etc.
    """
    start_url = normalize_url(start_url)
    url_prefix = url_prefix or start_url

    if allowed_domains is None:
        allowed_domains = {urlparse(start_url).netloc}

    if not obey_robots(start_url):
        logger.warning("Crawling disallowed by robots.txt for %s", start_url)
        return {}

    queue = deque([start_url])
    seen = set()
    pages = {}

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        parsed = urlparse(url)
        if parsed.netloc not in allowed_domains:
            continue

        # HARD LIMIT: stay under prefix
        if not url.startswith(url_prefix):
            continue

        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "ChatWithSiteBot/1.0"})
            if resp.status_code != 200:
                continue
            html = resp.text
            pages[url] = html

            # Only push more URLs if we still have budget
            if len(pages) < max_pages:
                for link in extract_links(html, url):
                    if link not in seen and link.startswith(url_prefix):
                        queue.append(link)

            time.sleep(delay)
        except Exception as e:
            logger.exception("Error fetching %s: %s", url, e)
            continue

    return pages
