"""
expert_loader.py — fetch news items for monitored experts.

Priority logic for each expert:
  1. If rss_url is set  → fetch directly from that RSS feed
  2. Otherwise           → query Google News RSS using search_names
"""

import requests
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dateutil import parser as _dateutil_parser


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
_TIMEOUT = 12   # seconds per request


def load_experts():
    import json
    try:
        with open("config/experts.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _parse_date(raw):
    """Try to parse a date string; return datetime or None."""
    if not raw:
        return None
    try:
        return _dateutil_parser.parse(raw)
    except Exception:
        return None


def _entry_to_item(entry, expert_name: str, source_label: str) -> dict:
    """Convert a feedparser entry to a standard item dict."""
    published = None
    for attr in ("published", "updated"):
        if hasattr(entry, attr):
            published = _parse_date(getattr(entry, attr))
            if published:
                break

    url = getattr(entry, "link", "") or ""
    title = getattr(entry, "title", "（無標題）") or "（無標題）"
    summary = getattr(entry, "summary", "") or ""
    source_name = (
        (entry.get("source") or {}).get("title", "") or source_label
    )

    return {
        "title": title,
        "url": url,
        "original_url": url,
        "source": source_name or expert_name,
        "published": published,
        "summary": summary,
        "content": "",
        "expert": expert_name,
        "type": "expert",
        "source_type": "expert",
        "source_category": ["自訂專家"],
        "source_region": "",
    }


def _fetch_from_rss_url(rss_url: str, expert_name: str) -> list:
    """Fetch items directly from a given RSS URL."""
    try:
        resp = requests.get(rss_url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            print(f"[ExpertLoader] RSS fetch failed for {expert_name}: HTTP {resp.status_code}")
            return []
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[ExpertLoader] RSS fetch error for {expert_name}: {e}")
        return []

    items = []
    for entry in feed.entries:
        items.append(_entry_to_item(entry, expert_name, expert_name))

    print(f"[ExpertLoader] {expert_name} (RSS) → {len(items)} 篇")
    return items


def _fetch_from_google_news(search_names: list, expert_name: str) -> list:
    """Query Google News RSS for each search name and deduplicate."""
    import urllib.parse

    seen_urls: set = set()
    items = []

    for name in search_names:
        if not name:
            continue
        query = urllib.parse.quote(f'"{name}"')
        # Try both zh-TW and en-US to maximise coverage
        for lang_params in [
            "hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
            "hl=en-US&gl=US&ceid=US:en",
        ]:
            url = f"https://news.google.com/rss/search?q={query}&{lang_params}"
            try:
                # Use requests.get with an explicit timeout so we never hang.
                # feedparser.parse(url) calls urllib with no timeout and can
                # block indefinitely; passing bytes avoids that issue.
                resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
            except Exception as e:
                print(f"[ExpertLoader] Google News error for '{name}': {e}")
                continue

            for entry in feed.entries:
                item = _entry_to_item(entry, expert_name, "Google News")
                link = item["url"].lower().strip()
                if link and link not in seen_urls:
                    seen_urls.add(link)
                    items.append(item)

    print(f"[ExpertLoader] {expert_name} (Google News) → {len(items)} 篇")
    return items


def search_expert_news(expert: dict) -> list:
    """
    Fetch news for a single expert.

    Priority:
      1. expert['rss_url'] is set  → fetch directly from that RSS feed
      2. otherwise                  → search Google News using search_names
    """
    expert_name = expert.get("name") or expert.get("name_zh") or expert.get("name_en") or "Unknown"
    rss_url = (expert.get("rss_url") or "").strip()

    if rss_url:
        return _fetch_from_rss_url(rss_url, expert_name)

    # Fall back to Google News keyword search
    # Use search_names (populated by normalize_expert / build_expert_search_names)
    # Prefer search_names over the old aliases field
    search_names = expert.get("search_names") or []
    if not search_names:
        # Fallback: build minimal list from name fields
        for field in ("name_zh", "name_en", "name"):
            v = (expert.get(field) or "").strip()
            if v and v not in search_names:
                search_names.append(v)

    if not search_names:
        return []

    return _fetch_from_google_news(search_names, expert_name)


def fetch_expert_items(selected_experts=None) -> list:
    """
    Fetch news for selected experts.

    selected_experts can be:
      - None or []        → fetch for ALL enabled experts
      - list of dicts     → expert objects (from UI selection)
      - list of strings   → expert display names
    """
    experts = load_experts()

    if selected_experts:
        # Support both list-of-dicts and list-of-name-strings
        if isinstance(selected_experts[0], dict):
            # Extract the canonical names from the selected dicts
            selected_names = {
                (e.get("name") or e.get("name_zh") or "").strip()
                for e in selected_experts
                if isinstance(e, dict)
            }
        else:
            selected_names = {str(s).strip() for s in selected_experts}

        experts = [
            e for e in experts
            if (e.get("name") or "").strip() in selected_names
            or (e.get("name_zh") or "").strip() in selected_names
        ]

    active_experts = [e for e in experts if e.get("enabled", True)]

    all_items = []

    def _fetch_one(expert):
        try:
            return search_expert_news(expert)
        except Exception as e:
            print(f"[ExpertLoader] fetch failed for {expert.get('name')}: {e}")
            return []

    # Fetch all experts in parallel (up to 8 concurrent) so the stage does not
    # block sequentially on each expert's network requests.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_one, e): e for e in active_experts}
        for future in as_completed(futures):
            all_items.extend(future.result())

    return all_items
