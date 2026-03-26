from __future__ import annotations

from datetime import datetime, timedelta
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

_CN_DISPLAY = {
    "rmrb":  "人民日報",
    "xwlb":  "新聞聯播",
    "jfjb":  "解放軍報",
    "xhs":   "新華社",
    "fmprc": "中國外交部",
    "mod":   "中國國防部",
    "gwytb": "國台辦",
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
    """
    cn.govopendata.com/xinwenlianbo/{yyyymmdd}/ 是單一頁面。
    每則新聞以 h2 或 h3 標題開頭（有左側裝飾邊框），後接若干 <p> 段落。
    按標題為單位切割，每則獨立成一筆 item。
    """
    items = []
    yyyymmdd = target_date.strftime("%Y%m%d")
    xl_url = f"https://cn.govopendata.com/xinwenlianbo/{yyyymmdd}/"

    try:
        r = requests.get(xl_url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return items
        r.encoding = "utf-8"

        soup = BeautifulSoup(r.text, "html.parser")

        # 每則標題是 h2 或 h3；過濾掉導覽列等極短文字
        headings = [h for h in soup.find_all(["h2", "h3"])
                    if len(h.get_text(strip=True)) > 4]

        if headings:
            for idx, h in enumerate(headings, 1):
                title_text = h.get_text(strip=True)

                # 收集這個標題之後、下一個標題之前的所有文字段落
                content_parts = []
                for sibling in h.find_next_siblings():
                    if sibling.name in ("h2", "h3"):
                        break
                    text = sibling.get_text(" ", strip=True)
                    if text:
                        content_parts.append(text)

                content = "\n".join(content_parts)
                items.append(_make_item(
                    source_name="新聞聯播",
                    title=f"新聞聯播 第{idx}則：{title_text}",
                    link=xl_url,
                    published=target_date,
                    summary=content[:280],
                    content=content[:1200],
                ))
        else:
            # fallback：頁面結構特殊，改抓所有 <p> 段落，每段編一則
            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")
                     if len(p.get_text(strip=True)) > 15]
            for idx, para in enumerate(paras, 1):
                hint = para[:50].rstrip("，。：: ") + ("…" if len(para) > 50 else "")
                items.append(_make_item(
                    source_name="新聞聯播",
                    title=f"新聞聯播 第{idx}則：{hint}",
                    link=xl_url,
                    published=target_date,
                    summary=para[:280],
                    content=para[:1200],
                ))

    except Exception:
        pass

    return items


def fetch_pla_daily(target_date: datetime) -> list[dict]:
    """
    解放軍報數字報 — 使用 JSON API 一次取得當天全部文章。
    API：http://www.81.cn/_szb/jfjb/{YYYY}/{MM}/{DD}/index.json
    回傳 paperInfo 陣列（12版），每版含 xyList（文章列表）。
    每篇文章有 id、title、content（HTML）、type（文字/圖片）、paperNumber。
    文章頁 URL：szbxq/index.html?paperName=jfjb&paperDate=YYYY-MM-DD&paperNumber=XX&articleid=ID
    """
    items = []
    date_str = target_date.strftime("%Y-%m-%d")
    y = target_date.strftime("%Y")
    m = target_date.strftime("%m")
    d = target_date.strftime("%d")

    json_url = f"http://www.81.cn/_szb/jfjb/{y}/{m}/{d}/index.json"
    try:
        r = requests.get(json_url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return items
        data = r.json()
    except Exception:
        return items

    seen_ids: set[str] = set()
    art_base = "http://www.81.cn/szb_223187/szbxq/index.html"

    for paper in data.get("paperInfo", []):
        paper_num = paper.get("paperNumber", "01")
        for xy in paper.get("xyList", []):
            # 只取文字稿，跳過圖片
            if xy.get("type") == "圖片" or xy.get("type") == "图片":
                continue
            art_id = str(xy.get("id", ""))
            if not art_id or art_id in seen_ids:
                continue
            seen_ids.add(art_id)

            title = xy.get("title", "").strip()
            if not title:
                continue

            # 從 HTML content 提取純文字
            raw_html = xy.get("content", "")
            if raw_html:
                try:
                    content_soup = BeautifulSoup(raw_html, "html.parser")
                    content = content_soup.get_text(" ", strip=True)
                except Exception:
                    content = re.sub(r'<[^>]+>', ' ', raw_html)
                    content = re.sub(r'\s+', ' ', content).strip()
            else:
                content = ""

            art_url = (
                f"{art_base}?paperName=jfjb&paperDate={date_str}"
                f"&paperNumber={paper_num}&articleid={art_id}"
            )

            items.append(_make_item(
                source_name="解放軍報",
                title=title,
                link=art_url,
                published=target_date,
                summary=content[:280],
                content=content[:1200],
            ))

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


_FETCH_MAP = {
    "rmrb":  fetch_people_daily,
    "xwlb":  fetch_xinwen_lianbo,
    "jfjb":  fetch_pla_daily,
    "xhs":   fetch_xinhua,
    "fmprc": fetch_fmprc,
    "mod":   fetch_mod,
    "gwytb": fetch_gwytb,
}

_ALL_KEYS = ["rmrb", "xwlb", "jfjb", "xhs", "fmprc", "mod", "gwytb"]


def fetch_official_media_for_day(
    target_date: datetime,
    requested_subsources: list[str] | None = None,
    callback=None,
) -> dict[str, list[dict]]:
    """callback(event, detail, completed, total, total_items) — 與 RSS status_callback 格式相同。"""
    empty = {k: [] for k in _ALL_KEYS}
    if requested_subsources is not None and len(requested_subsources) == 0:
        return empty

    requested_set = set(requested_subsources) if requested_subsources is not None else set(_ALL_KEYS)
    ordered = [k for k in _ALL_KEYS if k in requested_set]
    total = len(ordered)
    result = {k: [] for k in _ALL_KEYS}
    total_items = 0

    for idx, key in enumerate(ordered, 1):
        name = _CN_DISPLAY.get(key, key)
        if callback:
            try:
                callback("stage", f"⏳ {name}：抓取中…")
            except Exception:
                pass
        items = _FETCH_MAP[key](target_date)
        result[key] = items
        total_items += len(items)
        if callback:
            try:
                callback("rss", f"{name}：{len(items)} 篇", idx, total, total_items)
            except Exception:
                pass

    return result


def fetch_official_media_for_range(
    start_time: datetime,
    end_time: datetime,
    requested_subsources: list[str] | None = None,
    callback=None,
) -> dict[str, list[dict]]:
    result = {k: [] for k in _ALL_KEYS}

    current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end_time.replace(hour=0, minute=0, second=0, microsecond=0)

    while current <= last:
        daily = fetch_official_media_for_day(
            current,
            requested_subsources=requested_subsources,
            callback=callback,
        )
        for key, value in daily.items():
            result[key].extend(value)
        current += timedelta(days=1)

    return result