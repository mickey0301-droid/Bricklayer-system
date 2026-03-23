import re


def normalize_title(title: str) -> str:
    """
    將標題標準化，用於去重
    """
    if not title:
        return ""

    title = title.lower()

    # 移除標點符號
    title = re.sub(r"[^\w\s]", "", title)

    # 壓縮多餘空白
    title = re.sub(r"\s+", " ", title).strip()

    return title


def deduplicate_by_url(items):
    """
    用 URL 去重
    """
    seen_links = set()
    unique_items = []

    for item in items:
        link = item.get("link")

        if not link:
            unique_items.append(item)
            continue

        if link in seen_links:
            continue

        seen_links.add(link)
        unique_items.append(item)

    return unique_items


def deduplicate_by_title(items):
    """
    用標題標準化後的結果去重
    """
    seen_titles = set()
    unique_items = []

    for item in items:
        title = item.get("title", "")
        normalized = normalize_title(title)

        if not normalized:
            unique_items.append(item)
            continue

        if normalized in seen_titles:
            continue

        seen_titles.add(normalized)
        unique_items.append(item)

    return unique_items