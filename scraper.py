import logging
import time
from urllib.parse import parse_qs, urlparse

import requests
import trafilatura
from ddgs import DDGS
from youtube_transcript_api import YouTubeTranscriptApi

from models import ContentItem

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}


def _generate_search_queries(product_name: str) -> list[str]:
    return [
        f"{product_name} review",
        f"{product_name} worth it reddit",
        f"{product_name} vs",
        f"{product_name} pros cons",
        f'"{product_name}" opinion',
        f"{product_name} review youtube",
    ]


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


def search_reddit_via_ddgs(product_name: str, max_results: int = 10) -> list[str]:
    """Find Reddit threads via DuckDuckGo site:reddit.com search."""
    queries = [
        f"site:reddit.com {product_name} review",
        f"site:reddit.com {product_name} worth it",
    ]
    urls = []
    for q in queries:
        urls.extend(search_duckduckgo(q, max_results=max_results // 2))
    # Filter to actual reddit comment threads, deduplicated
    return [u for u in dict.fromkeys(urls) if "reddit.com/r/" in u and "/comments/" in u]


def _fetch_reddit_comments(thread_url: str, max_comments: int = 15,
                           timeout: int = 15) -> ContentItem | None:
    """Fetch post self-text + top comments from a Reddit thread via JSON."""
    try:
        json_url = thread_url.rstrip("/") + ".json"
        resp = requests.get(json_url, headers=DEFAULT_HEADERS, timeout=timeout)
        if resp.status_code == 429:
            logger.warning(f"Reddit rate-limited for {thread_url}")
            return None
        resp.raise_for_status()
        data = resp.json()

        # data is [post_listing, comments_listing]
        if not isinstance(data, list) or len(data) < 2:
            return None

        # Extract post info
        post_data = data[0]["data"]["children"][0]["data"]
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")

        # Extract top-level comments
        comments_listing = data[1]["data"]["children"]
        comment_texts = []
        for c in comments_listing[:max_comments]:
            if c.get("kind") != "t1":
                continue
            body = c.get("data", {}).get("body", "")
            if body and body != "[deleted]" and body != "[removed]":
                comment_texts.append(body)

        if not comment_texts and not selftext:
            return None

        # Combine post + comments into a single text block
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if selftext:
            parts.append(f"Post:\n{selftext}")
        if comment_texts:
            parts.append("Comments:\n" + "\n---\n".join(comment_texts))

        text = _truncate_text("\n\n".join(parts))
        item = ContentItem(url=thread_url, source_type="reddit", title=title, text=text)
        if item.word_count < 30:
            return None
        return item
    except Exception as e:
        logger.warning(f"Error fetching Reddit thread {thread_url}: {e}")
        return None


def search_youtube(product_name: str, max_results: int = 5) -> list[str]:
    """Find YouTube video URLs via DuckDuckGo site:youtube.com search."""
    query = f"site:youtube.com {product_name} review"
    urls = search_duckduckgo(query, max_results=max_results)
    return [u for u in urls if "youtube.com/watch" in u or "youtu.be/" in u]


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from a URL."""
    parsed = urlparse(url)
    if "youtube.com" in parsed.hostname:
        return parse_qs(parsed.query).get("v", [None])[0]
    if "youtu.be" in parsed.hostname:
        return parsed.path.lstrip("/")
    return None


def _fetch_youtube_transcript(video_url: str, timeout: int = 15) -> ContentItem | None:
    """Fetch transcript for a YouTube video."""
    try:
        video_id = _extract_video_id(video_url)
        if not video_id:
            logger.warning(f"Could not extract video ID from {video_url}")
            return None

        # Get transcript
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        snippets = list(transcript)
        if not snippets:
            return None
        transcript_text = " ".join(s.text for s in snippets)

        # Try to get video title via trafilatura
        title = ""
        try:
            downloaded = trafilatura.fetch_url(video_url)
            if downloaded:
                meta = trafilatura.metadata.extract_metadata(downloaded)
                if meta and meta.title:
                    title = meta.title
        except Exception:
            pass

        if not title:
            title = f"YouTube video {video_id}"

        text = _truncate_text(f"{title}\n\n{transcript_text}")
        item = ContentItem(url=video_url, source_type="youtube", title=title, text=text)
        if item.word_count < 50:
            logger.info(f"Skipping short transcript ({item.word_count} words): {video_url}")
            return None
        return item
    except Exception as e:
        logger.warning(f"Error fetching YouTube transcript for {video_url}: {e}")
        return None


def scrape_all(product_name: str, category: str | None = None,
               max_articles: int = 50, max_results_per_query: int = 20,
               timeout: int = 15, delay: float = 1.5) -> list[ContentItem]:
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
        urls = search_duckduckgo(query, max_results=max_results_per_query, timeout=timeout)
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

    # 3. Reddit threads via DuckDuckGo → fetch comments
    if len(items) < max_articles:
        logger.info("Searching Reddit via DuckDuckGo...")
        reddit_urls = search_reddit_via_ddgs(product_name)
        logger.info(f"Found {len(reddit_urls)} Reddit threads")
        for rurl in reddit_urls:
            if len(items) >= max_articles:
                break
            if rurl in seen_urls:
                continue
            logger.info(f"Fetching Reddit thread: {rurl}")
            item = _fetch_reddit_comments(rurl, timeout=timeout)
            _add(item)
            time.sleep(delay)

    # 4. YouTube videos via DuckDuckGo → fetch transcripts
    if len(items) < max_articles:
        logger.info("Searching YouTube via DuckDuckGo...")
        yt_urls = search_youtube(product_name)
        logger.info(f"Found {len(yt_urls)} YouTube videos")
        for yurl in yt_urls:
            if len(items) >= max_articles:
                break
            if yurl in seen_urls:
                continue
            logger.info(f"Fetching YouTube transcript: {yurl}")
            item = _fetch_youtube_transcript(yurl, timeout=timeout)
            _add(item)
            time.sleep(delay)

    logger.info(f"Total items scraped: {len(items)}")
    return items
