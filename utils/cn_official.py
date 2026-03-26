from __future__ import annotations

from datetime import datetime, timedelta
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

KEYWORDS = [
    "台湾", "台独", "台岛", "两岸", "台海", "涉台",
    "一中原则", "一个中国原则", "一个中国政策",
    "国民党", "民进党", "一中政策",
    "台企", "台胞", "宝岛",
    "陈水扁", "马英九", "蔡英文", "赖清德"
]

EXCLUDE_WORDS = ["黄河两岸", "长江两岸", "大江两岸", "江河两岸", "平台企业", "平台企划"]


def contains_real_keyword(text: str, title: str = "") -> bool:
    for exclude in EXCLUDE_WORDS:
        if exclude in text or exclude in title:
            return False

    for kw in KEYWORDS:
        if kw in text or kw in title:
            return True

    return False


def _make_item(source_name: str, title: str, link: str, published: datetime, summary: str = "", content: str = "", category=None):
    return {
        "source_name": source_name,
        "title": title.strip(),
        "link": link.strip(),
        "published": published,
        "summary": summary.strip(),
        "content": content.strip(),
        "category": category or ["China Official Media"],
        "type": "cn_official",
        "region": "中國"
    }


def fetch_people_daily(target_date: datetime) -> list[dict]:
    items = []
    seen_titles = set()

    yymm = target_date.strftime("%Y%m")   # e.g. "202603" — no dash, required by the site
    dd = target_date.strftime("%d")

    for p in range(1, 21):
        url = f"https://paper.people.com.cn/rmrb/pc/layout/{yymm}/{dd}/node_{p:02d}.html"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            r.encoding = "utf-8"

            matches = re.findall(r'content/\d{6}/\d{2}/content_\d+\.html', r.text)
            for m in set(matches):
                art_url = f"https://paper.people.com.cn/rmrb/pc/{m}"
                try:
                    art_r = requests.get(art_url, headers=HEADERS, timeout=10)
                    art_r.raise_for_status()
                    art_r.encoding = "utf-8"

                    title_match = re.search(r"<title>(.*?)</title>", art_r.text)
                    if not title_match:
                        continue

                    title = title_match.group(1).replace("-人民网", "").strip()
                    if title in seen_titles:
                        continue

                    seen_titles.add(title)

                    content_match = re.search(r'id="ozoom">([\s\S]*?)</div>', art_r.text)
                    txt = re.sub(r"<[^>]+>", "", content_match.group(1)).strip() if content_match else ""
                    txt_preview = txt[:1000] if len(txt) > 1000 else txt

                    items.append(_make_item(
                        source_name="人民日報",
                        title=title,
                        link=art_url,
                        published=target_date,
                        summary=txt_preview[:280],
                        content=txt_preview,
                    ))
                except Exception:
                    continue

        except Exception:
            continue

    return items


def fetch_xinwen_lianbo(target_date: datetime) -> list[dict]:
    items = []
    yyyymmdd = target_date.strftime("%Y%m%d")
    xl_url = f"https://cn.govopendata.com/xinwenlianbo/{yyyymmdd}/"

    try:
        r = requests.get(xl_url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return items

        soup = BeautifulSoup(r.text, "html.parser")
        links = [a for a in soup.find_all("a", href=True) if f"/xinwenlianbo/{yyyymmdd}/" in a["href"]]

        for a in links:
            item_url = "https://cn.govopendata.com" + a["href"]
            title = a.get_text(strip=True)
            try:
                item_r = requests.get(item_url, headers=HEADERS, timeout=10)
                text = BeautifulSoup(item_r.text, "html.parser").get_text(" ", strip=True)
                items.append(_make_item(
                    source_name="新聞聯播",
                    title=title,
                    link=item_url,
                    published=target_date,
                    summary=text[:280],
                    content=text[:1200],
                ))
            except Exception:
                continue
    except Exception:
        pass

    return items


def fetch_pla_daily(target_date: datetime) -> list[dict]:
    items = []
    try:
        jfj_url = f"http://www.81.cn/jfjbmap/content/{target_date.strftime('%Y-%m/%d')}/node_01.htm"
        r = requests.get(jfj_url, headers=HEADERS, timeout=10)
        r.encoding = "utf-8"

        if r.status_code == 200:
            text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
            items.append(_make_item(
                source_name="解放軍報",
                title=f"解放軍報 {target_date.strftime('%Y-%m-%d')}",
                link=jfj_url,
                published=target_date,
                summary=text[:280],
                content=text[:1200],
            ))
    except Exception:
        pass

    return items


def fetch_xinhua(target_date: datetime) -> list[dict]:
    items = []
    try:
        url = "http://www.news.cn/fortune/index.htm"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if contains_real_keyword(r.text):
            text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
            items.append(_make_item(
                source_name="新華社",
                title=f"新華社頁面涉台訊號 {target_date.strftime('%Y-%m-%d')}",
                link=url,
                published=target_date,
                summary=text[:280],
                content=text[:1200],
            ))
    except Exception:
        pass

    return items


def _fetch_list_page_articles(list_url: str, source_name: str, target_date: datetime, base_url: str | None = None) -> list[dict]:
    items = []
    try:
        r = requests.get(list_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"

        soup = BeautifulSoup(r.text, "html.parser")
        anchors = soup.find_all("a", href=True)

        seen = set()
        for a in anchors:
            href = a["href"].strip()
            title = a.get_text(" ", strip=True).strip()
            if not href or not title:
                continue

            full_url = href
            if base_url and href.startswith("/"):
                full_url = base_url.rstrip("/") + href

            if full_url in seen:
                continue
            seen.add(full_url)

            try:
                ar = requests.get(full_url, headers=HEADERS, timeout=10)
                ar.encoding = "utf-8"
                if not contains_real_keyword(ar.text, title):
                    continue

                text = BeautifulSoup(ar.text, "html.parser").get_text(" ", strip=True)
                items.append(_make_item(
                    source_name=source_name,
                    title=title,
                    link=full_url,
                    published=target_date,
                    summary=text[:280],
                    content=text[:1200],
                ))
            except Exception:
                continue

    except Exception:
        pass

    return items


def fetch_fmprc(target_date: datetime) -> list[dict]:
    return _fetch_list_page_articles(
        list_url="https://www.mfa.gov.cn/fyrbt_673021/jzhsl_673025/",
        source_name="中國外交部記者會",
        target_date=target_date,
        base_url="https://www.mfa.gov.cn"
    )


def fetch_mod(target_date: datetime) -> list[dict]:
    return _fetch_list_page_articles(
        list_url="http://www.mod.gov.cn/gfbw/xwfyr/fyrth/",
        source_name="中國國防部記者會",
        target_date=target_date,
        base_url="http://www.mod.gov.cn"
    )


def fetch_gwytb(target_date: datetime) -> list[dict]:
    return _fetch_list_page_articles(
        list_url="http://www.gwytb.gov.cn/xwdt/xwfb/",
        source_name="國台辦",
        target_date=target_date,
        base_url="http://www.gwytb.gov.cn"
    )


def fetch_official_media_for_day(target_date: datetime, requested_subsources: list[str] | None = None) -> dict[str, list[dict]]:
    # None → 預設抓全部；空 list [] → 明確表示什麼都不抓，直接回傳空結果
    if requested_subsources is not None and len(requested_subsources) == 0:
        return {"rmrb": [], "xwlb": [], "jfjb": [], "xhs": [], "fmprc": [], "mod": [], "gwytb": []}
    requested = set(requested_subsources) if requested_subsources is not None else {"rmrb", "xwlb", "jfjb", "xhs", "fmprc", "mod", "gwytb"}

    result = {
        "rmrb": [],
        "xwlb": [],
        "jfjb": [],
        "xhs": [],
        "fmprc": [],
        "mod": [],
        "gwytb": [],
    }

    if "rmrb" in requested:
        result["rmrb"] = fetch_people_daily(target_date)
    if "xwlb" in requested:
        result["xwlb"] = fetch_xinwen_lianbo(target_date)
    if "jfjb" in requested:
        result["jfjb"] = fetch_pla_daily(target_date)
    if "xhs" in requested:
        result["xhs"] = fetch_xinhua(target_date)
    if "fmprc" in requested:
        result["fmprc"] = fetch_fmprc(target_date)
    if "mod" in requested:
        result["mod"] = fetch_mod(target_date)
    if "gwytb" in requested:
        result["gwytb"] = fetch_gwytb(target_date)

    return result


def fetch_official_media_for_range(start_time: datetime, end_time: datetime, requested_subsources: list[str] | None = None) -> dict[str, list[dict]]:
    result = {
        "rmrb": [],
        "xwlb": [],
        "jfjb": [],
        "xhs": [],
        "fmprc": [],
        "mod": [],
        "gwytb": [],
    }

    current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end_time.replace(hour=0, minute=0, second=0, microsecond=0)

    while current <= last:
        daily = fetch_official_media_for_day(current, requested_subsources=requested_subsources)
        for key, value in daily.items():
            result[key].extend(value)
        current += timedelta(days=1)

    return result