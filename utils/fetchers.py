import requests
import feedparser
from dateutil import parser


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def fetch_rss_items(source: dict):

    print(f"[DEBUG] source={source['name']}")

    rss_url = source.get("rss") or source.get("url")

    if not rss_url:
        print(f"[DEBUG] {source['name']} 沒有 RSS URL")
        return []

    response = requests.get(rss_url, headers=HEADERS, timeout=10)

    if response.status_code != 200:
        print(f"[DEBUG] HTTP status={response.status_code}")
        return []

    feed = feedparser.parse(response.content)

    print(f"[DEBUG] feed entries={len(feed.entries)}")

    items = []

    for entry in feed.entries:

        published = None

        if hasattr(entry, "published"):
            try:
                published = parser.parse(entry.published)
            except Exception:
                pass

        elif hasattr(entry, "updated"):
            try:
                published = parser.parse(entry.updated)
            except Exception:
                pass

        item = {
            "source_id": source.get("id"),
            "source_name": source.get("name"),
            "title": getattr(entry, "title", "（無標題）"),
            "link": getattr(entry, "link", ""),
            "summary": getattr(entry, "summary", ""),
            "published": published,
        }

        items.append(item)

    print(f"[DEBUG] {source['name']} 抓到 {len(items)} 篇")

    return items


def fetch_items_from_sources(selected_sources: list):

    all_items = []

    for source in selected_sources:

        try:
            items = fetch_rss_items(source)
            all_items.extend(items)

        except Exception as e:
            print(f"[DEBUG] {source.get('name')} 抓取失敗: {e}")

    print(f"[DEBUG] 全部來源總共抓到 {len(all_items)} 篇")

    return all_items