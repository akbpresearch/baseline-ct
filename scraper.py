import logging
import time
from urllib.parse import urlparse

import requests
import trafilatura
from ddgs import DDGS

from models import ContentItem

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

REDDIT_SUBREDDITS_BY_CATEGORY = {
    "kitchen appliance": ["Coffee", "BuyItForLife", "cookingforbeginners", "Cooking"],
    "tech": ["technology", "gadgets", "BuyItForLife"],
    "audio": ["headphones", "audiophile", "BuyItForLife"],
    "fitness": ["fitness", "homegym", "BuyItForLife"],
    "gaming": ["gaming", "pcgaming", "NintendoSwitch"],
    "beauty": ["SkincareAddiction", "MakeupAddiction", "beauty"],
    "home": ["homeimprovement", "BuyItForLife", "InteriorDesign"],
}
DEFAULT_SUBREDDITS = ["BuyItForLife", "reviews", "consumer"]


def _generate_search_queries(product_name: str) -> list[str]:
    return [
        f"{product_name} review",
        f"{product_name} worth it reddit",
        f"{product_name} vs",
        f"{product_name} pros cons",
        f'"{product_name}" opinion',
    ]


def _get_subreddits(category: str | None) -> list[str]:
    if not category:
        return DEFAULT_SUBREDDITS
    category_lower = category.lower()
    for key, subs in REDDIT_SUBREDDITS_BY_CATEGORY.items():
        if key in category_lower or category_lower in key:
            return subs
    return DEFAULT_SUBREDDITS


def _truncate_text(text: str, max_words: int = 3000) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def scrape_url(url: str, timeout: int = 15) -> ContentItem | None:
    """Scrape a single URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            logger.warning(f"Failed to fetch: {url}")
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text:
            logger.warning(f"No text extracted: {url}")
            return None
        metadata = trafilatura.extract(downloaded, output_format="xmltei", include_comments=False)
        title = urlparse(url).path.strip("/").split("/")[-1].replace("-", " ").title() if not metadata else ""
        # Try to get title from trafilatura metadata
        meta = trafilatura.metadata.extract_metadata(downloaded)
        if meta and meta.title:
            title = meta.title
        text = _truncate_text(text)
        item = ContentItem(url=url, source_type="article", title=title, text=text)
        if item.word_count < 50:
            logger.info(f"Skipping short content ({item.word_count} words): {url}")
            return None
        return item
    except Exception as e:
        logger.warning(f"Error scraping {url}: {e}")
        return None


def search_duckduckgo(query: str, max_results: int = 10, timeout: int = 15) -> list[str]:
    """Search DuckDuckGo using the duckduckgo-search library."""
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                href = r.get("href", "")
                if href and href.startswith("http"):
                    urls.append(href)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
    return urls[:max_results]


def search_reddit(product_name: str, subreddits: list[str], max_per_sub: int = 5,
                   timeout: int = 15) -> list[ContentItem]:
    """Search Reddit via public JSON endpoint."""
    items = []
    for sub in subreddits:
        try:
            url = f"https://old.reddit.com/r/{sub}/search.json"
            params = {
                "q": product_name,
                "restrict_sr": "on",
                "sort": "relevance",
                "t": "all",
                "limit": max_per_sub,
            }
            resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
            if resp.status_code == 429:
                logger.warning(f"Reddit rate-limited on r/{sub}, skipping")
                continue
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for post in posts:
                pd = post.get("data", {})
                selftext = pd.get("selftext", "")
                title = pd.get("title", "")
                permalink = pd.get("permalink", "")
                full_url = f"https://reddit.com{permalink}" if permalink else ""
                if not selftext or len(selftext.split()) < 50:
                    continue
                text = _truncate_text(f"{title}\n\n{selftext}")
                item = ContentItem(
                    url=full_url,
                    source_type="reddit",
                    title=title,
                    text=text,
                )
                items.append(item)
            time.sleep(1.0)  # Polite delay between subreddits
        except Exception as e:
            logger.warning(f"Reddit search failed for r/{sub}: {e}")
    return items


def scrape_all(product_name: str, category: str | None = None,
               max_articles: int = 50, timeout: int = 15, delay: float = 1.5) -> list[ContentItem]:
    """Run all scraping strategies and return deduplicated ContentItems."""
    seen_urls: set[str] = set()
    items: list[ContentItem] = []

    def _add(item: ContentItem | None):
        if item and item.url not in seen_urls:
            seen_urls.add(item.url)
            items.append(item)

    # 1. DuckDuckGo search discovery
    queries = _generate_search_queries(product_name)
    discovered_urls: list[str] = []
    for query in queries:
        logger.info(f"Searching: {query}")
        urls = search_duckduckgo(query, max_results=10, timeout=timeout)
        discovered_urls.extend(urls)
        time.sleep(delay)

    # Deduplicate discovered URLs
    unique_urls = list(dict.fromkeys(discovered_urls))
    logger.info(f"Discovered {len(unique_urls)} unique URLs from search")

    # 2. Scrape discovered URLs via trafilatura
    for url in unique_urls:
        if len(items) >= max_articles:
            break
        logger.info(f"Scraping: {url}")
        item = scrape_url(url, timeout=timeout)
        _add(item)
        time.sleep(delay)

    # 3. Reddit search
    if len(items) < max_articles:
        subreddits = _get_subreddits(category)
        logger.info(f"Searching Reddit: r/{', r/'.join(subreddits)}")
        reddit_items = search_reddit(product_name, subreddits, timeout=timeout)
        for item in reddit_items:
            if len(items) >= max_articles:
                break
            _add(item)

    logger.info(f"Total items scraped: {len(items)}")
    return items
