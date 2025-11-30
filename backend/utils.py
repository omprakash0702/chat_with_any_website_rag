import requests
from urllib.parse import urlparse, urljoin
import re
import hashlib

###########################################
# Normalize URL
###########################################
def normalize_url(url: str) -> str:
    """
    Normalize a URL by:
    - Removing fragments (#)
    - Removing trailing slash
    - Lowercasing scheme + domain
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")  # remove trailing slash
    query = parsed.query

    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized += f"?{query}"

    return normalized


###########################################
# Robots.txt Checker
###########################################
def obey_robots(url: str) -> bool:
    """
    Checks robots.txt for the given site.
    Returns True if allowed, False if disallowed.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        r = requests.get(robots_url, timeout=5)
        if r.status_code != 200:
            return True  # assume allowed if robots.txt missing

        lines = r.text.split("\n")
        rules = []
        user_agent = None
        allowed = True

        for line in lines:
            line = line.strip().lower()

            if line.startswith("user-agent:"):
                user_agent = line.replace("user-agent:", "").strip()

            # apply rules only for *
            if user_agent == "*" and line.startswith("disallow:"):
                rule = line.replace("disallow:", "").strip()
                rules.append(rule)

        path = parsed.path

        # if any rule is a prefix of path → disallow
        for rule in rules:
            if rule != "" and path.startswith(rule):
                return False

        return True

    except:
        return True  # if error → assume allowed


###########################################
# Hashing for incremental indexing
###########################################
def hash_text(text: str) -> str:
    """
    Returns SHA256 hash of given text.
    Used for detecting page changes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


###########################################
# Clean print for debugging
###########################################
def shorten(text: str, limit: int = 200) -> str:
    """
    Returns first N characters for console preview.
    """
    text = text.strip().replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


###########################################
# URL domain extractor
###########################################
def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()
