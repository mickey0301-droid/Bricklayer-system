import os
import re
import json
from datetime import datetime

from openai import OpenAI

import streamlit as st

def _ensure_gemini_configured():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass

    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in env or secrets")

    genai.configure(api_key=api_key)

from utils.loaders import load_sources, load_experts, load_formats, load_insights, load_category_keywords
from utils.expert_loader import fetch_expert_items

try:
    from utils.cn_official import fetch_official_media_for_range
except Exception:
    def fetch_official_media_for_range(start_time=None, end_time=None, requested_subsources=None):
        return {}

# Mapping from loaders.py subsource keys → cn_official.py subsource keys
_CN_SUBSOURCE_MAP = {
    "people_daily": "rmrb",
    "xinwen_lianbo": "xwlb",
    "pla_daily": "jfjb",
    "xinhua": "xhs",
    "mfa_press": "fmprc",
    "mod_press": "mod",
    "taiwan_affairs_office": "gwytb",
}

# ── 台灣／中國相關關鍵字（中英文） ──────────────────────────────────────
_KEYWORDS_TW_CN = [
    # 繁體中文
    "台灣", "台海", "兩岸", "台獨", "台美", "台中", "台日",
    "解放軍", "共軍", "共產黨", "中共", "習近平", "賴清德",
    "民進黨", "國民黨", "蔡英文", "馬英九", "陳水扁",
    "台積電", "半導體", "印太", "東海", "南海", "第一島鏈",
    "統一", "九二共識", "一個中國", "金門", "馬祖",
    # 簡體中文
    "台湾", "两岸", "台独", "解放军", "习近平", "赖清德",
    "民进党", "国民党", "台积电", "半导体", "印太",
    # 英文
    "taiwan", "cross-strait", "pla ", "chinese military",
    "tsmc", "semiconductor", "indo-pacific", "strait",
    "xi jinping", "ccp", "beijing", "one china",
]

# 全球媒體類別名稱（用來判斷是否走「熱度排名」邏輯）
_GLOBAL_MEDIA_CATEGORIES = {"全球媒體"}

# 各來源類別對應的 Google News RSS 查詢關鍵字
# 實際值由 load_category_keywords() 動態讀取（使用者可在 Sources 頁面自訂並儲存）
# 此處僅保留做為 import 路徑別名，實際預設值定義在 utils/loaders.py
from utils.loaders import DEFAULT_CATEGORY_KEYWORDS as _CATEGORY_KEYWORDS


def _matches_tw_cn(item: dict) -> bool:
    """判斷文章是否與台灣／中國議題相關（看標題＋摘要）。"""
    text = (
        (item.get("title") or "") + " " + (item.get("summary") or "")
    ).lower()
    return any(kw.lower() in text for kw in _KEYWORDS_TW_CN)


def _rank_by_coverage(items: list, top_n: int = 30) -> list:
    """
    國際要聞熱度排名：找出被最多來源同時報導的新聞。
    做法：把標題切成詞（≥3 字元），找出詞頻最高的詞組合，
    相同詞組代表同一則新聞，依來源數量排序後取 top_n。
    """
    import re
    from collections import defaultdict

    # 英文停用詞（不計入聚合鍵）
    _STOP = {
        "the","a","an","in","on","at","to","for","of","and","or",
        "is","are","was","were","be","been","has","have","had",
        "it","its","this","that","with","as","by","from","about",
        "after","before","over","says","say","said","will","would",
        "could","his","her","their","our","we","he","she","they",
        "not","no","new","more","one","two","three","also","than",
    }

    def _key(title: str) -> frozenset:
        words = re.findall(r"[a-zA-Z\u4e00-\u9fff]{3,}", title.lower())
        return frozenset(w for w in words if w not in _STOP)

    groups: dict[frozenset, list] = defaultdict(list)
    for item in items:
        k = _key(item.get("title", ""))
        if k:
            groups[k].append(item)

    # 按各組的文章數（即有多少來源報導）降序排列
    ranked_groups = sorted(groups.values(), key=len, reverse=True)

    result = []
    seen_urls: set = set()
    for group in ranked_groups:
        if len(result) >= top_n:
            break
        # 每組取一篇代表（選來源最多那組的第一篇）
        for art in group:
            url = art.get("url") or art.get("original_url") or ""
            if url not in seen_urls:
                seen_urls.add(url)
                result.append(art)
                break

    return result


def _select_diverse_topics(
    items: list,
    n_topics: int = 4,
    articles_per_topic: int = 2,
) -> list:
    """
    在一個章節的文章池中，找出被最多來源報導的前 n_topics 個不同議題，
    每個議題取 articles_per_topic 篇代表文章。

    目的：避免某章節全部文章都在講同一件事，確保每章涵蓋 3-4 個獨立議題。

    做法與 _rank_by_coverage 相似——用標題關鍵字（≥3 字元、排除停用詞）建立
    議題群組，按群組大小（即有多少文章報導該議題）降序排列，依序取代表文章，
    直到達到 n_topics 個議題或已無更多文章為止。
    """
    from collections import defaultdict

    _STOP = {
        "the","a","an","in","on","at","to","for","of","and","or",
        "is","are","was","were","be","been","has","have","had",
        "it","its","this","that","with","as","by","from","about",
        "after","before","over","says","say","said","will","would",
        "could","his","her","their","our","we","he","she","they",
        "not","no","new","more","one","two","three","also","than",
    }

    def _key(title: str) -> frozenset:
        words = re.findall(r"[a-zA-Z\u4e00-\u9fff]{3,}", title.lower())
        return frozenset(w for w in words if w not in _STOP)

    # ── 1. 用標題關鍵字把文章分群 ──────────────────────────────────────────
    clusters: dict = defaultdict(list)
    singletons: list = []  # 標題無法解析的文章，最後補位用

    for item in items:
        k = _key(item.get("title", ""))
        if k:
            clusters[k].append(item)
        else:
            singletons.append(item)

    # ── 2. 先嘗試合併「高度重疊」的小群（共用 ≥ 2/3 的關鍵字） ────────────
    merged: list[list] = []
    used: set = set()
    sorted_keys = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)

    for ki in sorted_keys:
        if id(ki) in used:
            continue
        group = list(clusters[ki])
        used.add(id(ki))
        for kj in sorted_keys:
            if id(kj) in used:
                continue
            # 重疊度：共同詞 / 較小集合大小
            if ki and kj:
                overlap = len(ki & kj) / max(len(ki), len(kj), 1)
                if overlap >= 0.5:
                    group.extend(clusters[kj])
                    used.add(id(kj))
        merged.append(group)

    # 按群組大小排序（大群 = 被多來源報導）
    merged.sort(key=len, reverse=True)

    # ── 3. 從前 n_topics 個議題各取 articles_per_topic 篇 ─────────────────
    result: list = []
    seen_urls: set = set()

    for cluster in merged[:n_topics]:
        count = 0
        for item in cluster:
            if count >= articles_per_topic:
                break
            url = (item.get("original_url") or item.get("url") or "").lower().strip()
            title = (item.get("title") or "").lower().strip()
            key = url or title
            if key and key not in seen_urls:
                seen_urls.add(key)
                result.append(item)
                count += 1

    # ── 4. 若結果不足 n_topics，用 singletons 補位 ─────────────────────────
    for item in singletons:
        if len(result) >= n_topics * articles_per_topic:
            break
        url = (item.get("original_url") or item.get("url") or "").lower().strip()
        title = (item.get("title") or "").lower().strip()
        key = url or title
        if key and key not in seen_urls:
            seen_urls.add(key)
            result.append(item)

    return result


import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime
from bs4 import BeautifulSoup


from utils.loaders import load_sources

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
_REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
}

def _normalize_selected_sources(selected_sources, all_sources=None):
    """
    Accept both:
    1) ["Reuters", "AP"]
    2) [{"name": "Reuters", ...}, {"name": "AP", ...}]
    Return a normalized list of source dicts.
    """
    if not selected_sources:
        return []

    if isinstance(selected_sources[0], dict):
        return selected_sources

    if not all_sources:
        return []

    source_map = {s.get("name"): s for s in all_sources if isinstance(s, dict) and s.get("name")}
    normalized = []
    for name in selected_sources:
        if name in source_map:
            normalized.append(source_map[name])
    return normalized


def _resolve_google_news_url(url):
    """
    Best-effort: convert Google News redirect URLs back to original article URLs.
    Tries multiple methods; falls back to original URL if all fail.
    """
    if not url or "news.google.com" not in url:
        return url

    # 方法一：嘗試 base64 解碼 Google News article ID
    try:
        import base64
        path = url.split("?")[0]
        article_id = path.rstrip("/").split("/")[-1]
        if article_id.startswith("CBMi"):
            padded = article_id + "=" * (-len(article_id) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode("latin-1")
            http_start = decoded.find("http")
            if http_start != -1:
                candidate = ""
                for ch in decoded[http_start:]:
                    if ord(ch) < 32 or ord(ch) > 126:
                        break
                    candidate += ch
                if candidate.startswith("http") and "google.com" not in candidate:
                    return candidate
    except Exception:
        pass

    # 方法二：HTTP redirect 跟隨
    try:
        r = requests.get(
            url,
            headers=_REQUEST_HEADERS,
            timeout=10,
            allow_redirects=True,
        )
        final_url = r.url or ""
        if final_url and "news.google.com" not in final_url:
            return final_url
        # 方法三：從 HTML 內容找 canonical / og:url
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in [
            soup.find("link", rel="canonical"),
            soup.find("meta", property="og:url"),
        ]:
            if tag:
                href = tag.get("href") or tag.get("content") or ""
                if href.startswith("http") and "news.google.com" not in href:
                    return href
    except Exception:
        pass

    return url


def _parse_rss(xml_text):
    items = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # 支援 Atom feed 的 <entry> 以及 RSS 的 <item>
    entries = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for item in entries:
        def _text(*tags):
            for t in tags:
                v = item.findtext(t) or item.findtext(f"{{http://www.w3.org/2005/Atom}}{t}")
                if v:
                    return v.strip()
            return ""

        title = _text("title")
        # Atom <link href="..."> 和 RSS <link>
        link = _text("link")
        if not link:
            le = item.find("link") or item.find("{http://www.w3.org/2005/Atom}link")
            if le is not None:
                link = le.get("href", "")

        # 支援多種日期欄位：pubDate / published / updated / dc:date
        pub_date = (
            _text("pubDate")
            or _text("published")
            or _text("updated")
            or _text("{http://purl.org/dc/elements/1.1/}date")
        )
        description = _text("description", "summary", "content")

        # Google News RSS 的 <source url="https://cna.com.tw">中央社</source>
        src_elem = item.find("source")
        source_publisher_url = src_elem.get("url", "") if src_elem is not None else ""

        if title or link:
            items.append({
                "title": title,
                "url": link,
                "published": pub_date,
                "summary": description,
                "source_publisher_url": source_publisher_url,
            })
    return items


def _fetch_article_content(url, max_chars=8000):
    """
    Fetch full article text from a single article URL.
    Best-effort generic extractor using common containers + paragraph fallback.
    """
    if not url:
        return ""

    try:
        r = requests.get(
            url,
            headers=_REQUEST_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[Briefings] Article fetch failed for {url}: {e}")
        return ""

    try:
        soup = BeautifulSoup(r.text, "html.parser")

        # 去掉常見雜訊
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
            tag.decompose()

        candidates = []

        # 優先找比較像正文的區塊
        selectors = [
            "article",
            "[role='article']",
            ".article-body",
            ".article-content",
            ".post-content",
            ".entry-content",
            ".story-body",
            ".news-content",
            ".article__content",
            ".main-content",
        ]

        for selector in selectors:
            try:
                nodes = soup.select(selector)
                for node in nodes:
                    paragraphs = [p.get_text(" ", strip=True) for p in node.find_all("p")]
                    text = "\n".join([p for p in paragraphs if p])
                    if len(text) > 300:
                        candidates.append(text)
            except Exception:
                pass

        # 如果沒抓到合格正文，就退回全頁 paragraph
        if not candidates:
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = "\n".join([p for p in paragraphs if p])
            candidates.append(text)

        # 取最長的那個
        content = max(candidates, key=len) if candidates else ""
        content = content.strip()

        if len(content) > max_chars:
            content = content[:max_chars]

        return content

    except Exception as e:
        print(f"[Briefings] Article parse failed for {url}: {e}")
        return ""


def _get_gnews_session() -> requests.Session:
    """
    建立帶有 Google News Cookie 的 Session。
    先訪問 news.google.com 首頁取得 Cookie，再用同一個 Session 抓 RSS，
    避免被 Google 的 bot 偵測擋掉（回傳 HTML Sorry... 頁面）。
    Session 物件快取在 module 層級，同一執行期只建一次。
    """
    if not hasattr(_get_gnews_session, "_session"):
        sess = requests.Session()
        try:
            sess.get(
                "https://news.google.com/",
                headers=_REQUEST_HEADERS,
                timeout=10,
            )
        except Exception:
            pass
        _get_gnews_session._session = sess
    return _get_gnews_session._session


def _fetch_rss_items(rss_url, source_name, limit=20):
    # Google News RSS：使用帶 Cookie 的 Session 避免 bot 偵測
    if "news.google.com" in rss_url:
        sess = _get_gnews_session()
        fetch = lambda u, h, t: sess.get(u, headers=h, timeout=t)
    else:
        fetch = lambda u, h, t: requests.get(u, headers=h, timeout=t)

    try:
        r = fetch(rss_url, _REQUEST_HEADERS, 10)
        r.raise_for_status()
        # r.content (bytes) 讓 ET 直接用 XML 宣告中的 encoding，
        # 避免 requests 把 text/xml 錯誤預設成 iso-8859-1
        parsed = _parse_rss(r.content)
        print(f"[RSS] {source_name}: HTTP {r.status_code}, "
              f"content-type={r.headers.get('content-type','?')!r}, "
              f"len={len(r.content)}, items={len(parsed)}, "
              f"url={rss_url[:120]}")
        if len(parsed) == 0 and len(r.content) > 0:
            print(f"[RSS] {source_name} response preview: {r.content[:300]!r}")
    except Exception as e:
        print(f"[Briefings] RSS fetch failed for {source_name}: {e}")
        return []

    # 只使用 RSS 的 title + summary，不額外抓全文
    # （與 daily_report.py 做法一致，速度快且不會因為抓全文而逾時）
    output = []
    for item in parsed[:limit]:
        raw_url = item.get("url", "")
        output.append({
            "title": item.get("title", "").strip(),
            "url": raw_url,
            "original_url": raw_url,
            "source": source_name,
            "published": item.get("published", ""),
            "summary": item.get("summary", ""),
            "content": "",
            "source_type": "rss",
        })
    return output


_FEED_SERVICE_DOMAINS = {
    "feedburner.com", "feedblitz.com", "feedpress.me",
    "feeds.feedburner.com",
}

_FEED_SUBDOMAIN_RE = re.compile(r'^(feeds?\.|rss\.|feed\.)', re.IGNORECASE)


def _extract_news_domain(url: str) -> str | None:
    """
    從 URL 萃取可供 Google News site: 搜尋的新聞網站 domain。

    處理常見情況：
    - https://feeds.udn.com/rss/...  → udn.com
    - https://rss.cna.com.tw/...     → cna.com.tw
    - https://udn.com/rss/...        → udn.com
    - https://apnews.com             → apnews.com
    Feedburner 等第三方 feed 服務無法反推原始 domain，直接回傳 None。
    """
    if not url:
        return None
    try:
        raw = url.replace("https://", "").replace("http://", "")
        domain = raw.split("/")[0].lower()
    except Exception:
        return None

    # 第三方 feed 服務無法反推原始來源
    for svc in _FEED_SERVICE_DOMAINS:
        if svc in domain:
            return None

    # 移除 feeds. / rss. / feed. 子網域前綴，還原新聞網站本體
    domain = _FEED_SUBDOMAIN_RE.sub("", domain)
    return domain or None


def _build_google_news_rss_for_domain(domain, start_time=None, end_time=None, keywords=None):
    """
    建立 Google News RSS 查詢 URL。

    重要：Google News RSS 的 site: operator 單獨使用時回傳 0 結果。
    必須在 site: 前面加上關鍵字才能正常運作，格式：
        (kw1 OR kw2) site:domain when:Xd
    時間精確過濾由 _filter_items_by_time_range 在 client 端補做。
    """
    # 計算 when: 參數
    when_str = "when:3d"   # 預設抓最近 3 天
    if start_time and end_time:
        hours = max(1, int((end_time - start_time).total_seconds() / 3600))
        if hours <= 6:
            when_str = "when:6h"
        elif hours <= 24:
            when_str = "when:1d"
        elif hours <= 72:
            when_str = "when:3d"
        elif hours <= 168:
            when_str = "when:7d"
        else:
            when_str = ""

    if keywords:
        query = f"({keywords}) site:{domain}"
    else:
        # 沒關鍵字時用最基本的台灣相關詞，確保 Google News RSS 能回傳結果
        query = f"台灣 OR Taiwan site:{domain}"
    if when_str:
        query += f" {when_str}"

    # 用 safe=':/' 確保 site: 的冒號不被編碼（編碼後 Google 不識別為 operator）
    query_encoded = quote(query, safe=':/')
    return f"https://news.google.com/rss/search?q={query_encoded}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"


def _fetch_domain_items(domain, source_name, limit=20, start_time=None, end_time=None, keywords=None):
    rss_url = _build_google_news_rss_for_domain(
        domain, start_time=start_time, end_time=end_time, keywords=keywords
    )
    items = _fetch_rss_items(rss_url, source_name, limit=limit)

    for item in items:
        item["source_type"] = "domain"
        if not item.get("source"):
            item["source"] = source_name

    return items


from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def _domains_match(a: str, b: str) -> bool:
    """比對兩個 domain 是否屬於同一個網站（忽略 www. 前綴）。"""
    a = (a or "").lower().replace("www.", "")
    b = (b or "").lower().replace("www.", "")
    return bool(a and b and (a == b or a.endswith("." + b) or b.endswith("." + a)))


def _kw_matches(item: dict, keywords_str: str) -> bool:
    """
    Client 端關鍵字過濾。
    keywords_str 格式：「kw1 OR kw2 OR kw3」（以 OR 分隔，不分大小寫）。
    空字串視為「全部符合」（不過濾）。
    """
    if not keywords_str:
        return True
    text = ((item.get("title") or "") + " " + (item.get("summary") or "")).lower()
    terms = [t.strip().lower() for t in keywords_str.split(" OR ") if t.strip()]
    return any(t in text for t in terms)


def fetch_items_from_sources(selected_sources, all_sources=None, limit_per_source=20,
                              start_time=None, end_time=None, status_callback=None):
    """
    抓取流程（與舊版 daily_report 一致）：
    - 直接 RSS URL：優先使用 url/rss/rss_url/feed/feed_url 欄位裡的 http 網址
    - domain 來源：用「(keywords) site:domain when:Xd」查 Google News RSS
      （site: 單獨使用時 Google News RSS 回傳 0，必須搭配關鍵字）
    - 關鍵字來自各類別的 category_keywords 設定
    - 自訂專家：用 "{name}" 關鍵字直接查 Google News RSS
    - cn_official：由 generate_report() 專屬爬蟲處理，此處跳過
    """

    normalized_sources = _normalize_selected_sources(selected_sources, all_sources=all_sources)
    category_keywords = load_category_keywords()

    src_list = [s for s in normalized_sources if s.get("type") != "cn_official"]
    all_items = []
    total = len(src_list)
    completed = 0

    def fetch_single(src):
        cats = src.get("category", []) or []
        if isinstance(cats, str):
            cats = [cats]
        cat = cats[0] if cats else ""
        src_name = src.get("name", "")

        # 自訂專家：用名字查詢，不限定 site:
        if cat == "自訂專家":
            expert_name = src_name.strip()
            if not expert_name:
                return []
            kw = f'"{expert_name}"'
            if start_time:
                kw += f" after:{start_time.strftime('%Y/%m/%d')}"
            if end_time:
                kw += f" before:{end_time.strftime('%Y/%m/%d')}"
            rss_url = f"https://news.google.com/rss/search?q={quote(kw)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            fetched = _fetch_rss_items(rss_url, expert_name, limit=limit_per_source)
            for item in fetched:
                item["source"] = expert_name
                item["source_category"] = cats
                item["source_region"] = src.get("region", "")
                item["source_type"] = "gnews"
            return fetched

        src_type  = src.get("type", "rss")
        url_field = src.get("url", "")

        # type=rss：優先使用 rss/rss_url/feed/feed_url，
        #           再看 url 是否為完整 http 網址（使用者手填的真實 RSS feed）
        # type=domain（或其他）：永遠走 Google News RSS site: 查詢
        rss_url = None
        if src_type == "rss":
            rss_url = (
                src.get("rss") or src.get("rss_url")
                or src.get("feed") or src.get("feed_url")
            )
            if not rss_url and url_field.startswith("http"):
                rss_url = url_field

        if not rss_url:
            # domain 或沒有直接 feed URL → Google News site: 查詢（帶關鍵字）
            domain = (src.get("domain") or src.get("site")
                      or _extract_news_domain(url_field) or url_field)
            if not domain:
                return []
            domain = domain.lower().replace("www.", "")
            cat_kw = category_keywords.get(cat, "")
            rss_url = _build_google_news_rss_for_domain(
                domain, start_time=start_time, end_time=end_time, keywords=cat_kw
            )

        fetched = _fetch_rss_items(rss_url, src_name, limit=limit_per_source)

        for item in fetched:
            item["source"] = src_name
            item["source_category"] = cats
            item["source_region"] = src.get("region", "")
            item["source_type"] = src.get("type", "rss")
        return fetched

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_single, src): src for src in src_list}
        for future in as_completed(futures):
            src = futures[future]
            try:
                items = future.result()
                all_items.extend(items)
                completed += 1
                if status_callback:
                    src_name = src.get("name", "?")
                    if items:
                        status_callback("rss", f"{src_name}：{len(items)} 篇",
                                        completed, total, len(all_items))
                    else:
                        status_callback("rss_progress", None, completed, total, len(all_items))
            except Exception as e:
                completed += 1
                print(f"[Briefings] fetch_single error for {src.get('name','?')}: {e}")

    return all_items


def debug_fetch_source(src: dict, start_time=None, end_time=None) -> dict:
    """
    對單一來源執行完整診斷：走與正式 pipeline 完全相同的路徑，
    回傳包含 HTTP status、content-type、rss_url、parsed items 等詳細資訊的 dict。
    供 Sources 頁面的「測試抓取」按鈕使用。
    """
    category_keywords = load_category_keywords()

    src_type  = src.get("type", "rss")
    url_field = src.get("url", "")
    cats      = src.get("category", []) or []
    if isinstance(cats, str):
        cats = [cats]
    cat = cats[0] if cats else ""

    # ── 決定 rss_url（與 fetch_single 完全相同邏輯） ──
    rss_url = None
    domain_used = None

    if src_type == "rss":
        rss_url = (
            src.get("rss") or src.get("rss_url")
            or src.get("feed") or src.get("feed_url")
        )
        if not rss_url and url_field.startswith("http"):
            rss_url = url_field

    if not rss_url:
        domain_used = (src.get("domain") or src.get("site")
                       or _extract_news_domain(url_field) or url_field)
        if domain_used:
            domain_used = domain_used.lower().replace("www.", "")
        cat_kw = category_keywords.get(cat, "")
        rss_url = _build_google_news_rss_for_domain(
            domain_used or "", start_time=start_time, end_time=end_time, keywords=cat_kw
        )

    # ── 發出 HTTP 請求，收集診斷資訊 ──
    result = {
        "name": src.get("name", "?"),
        "src_type": src_type,
        "original_url": url_field,
        "domain_extracted": domain_used,
        "rss_url": rss_url,
        "http_status": None,
        "content_type": None,
        "response_len": 0,
        "response_preview": "",
        "items_parsed": 0,
        "items": [],
        "error": None,
    }

    try:
        r = requests.get(rss_url, headers=_REQUEST_HEADERS, timeout=12)
        result["http_status"]    = r.status_code
        result["content_type"]   = r.headers.get("content-type", "?")
        result["response_len"]   = len(r.content)
        result["response_preview"] = r.content[:300].decode("utf-8", errors="replace").replace("\n", " ")

        parsed = _parse_rss(r.content)
        result["items_parsed"] = len(parsed)
        result["items"] = parsed[:5]
    except Exception as e:
        result["error"] = str(e)

    return result


from email.utils import parsedate_to_datetime

def _parse_published_datetime(value):
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    try:
        return parsedate_to_datetime(str(value))
    except Exception:
        pass

    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _filter_items_by_time_range(items, start_time, end_time):
    if not start_time or not end_time:
        return items

    filtered = []

    for item in items:
        published_dt = _parse_published_datetime(item.get("published"))

        if not published_dt:
            continue

        # 如果 published 有 timezone，但 start/end 沒有，可先去掉 timezone
        try:
            if published_dt.tzinfo is not None:
                published_dt = published_dt.replace(tzinfo=None)
        except Exception:
            pass

        if start_time <= published_dt <= end_time:
            filtered.append(item)

    return filtered

# =====================================================
# Load format settings
# =====================================================

def _load_format_options():

    try:
        with open("config/formats.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data[0]

        if isinstance(data, dict) and "formats" in data:
            return data["formats"][0]

        return data

    except:
        return {
            "fonts": {"zh": "Arial", "en": "Arial"},
            "title": {"font_size": 16, "bold": True},
            "section_heading": {"font_size": 14, "bold": True},
            "body": {"font_size": 12, "line_spacing": 1.15},
            "notes": {"style": "footnote"},
            "links": {"placement": "inline"},
        }


# =====================================================
# Topic monitoring engine
# =====================================================

def load_topics():

    try:
        with open("config/topics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def filter_items_by_topic(items, topic_name):

    if not topic_name:
        return items

    topics = load_topics()

    topic = None

    for t in topics:
        if t.get("name") == topic_name:
            topic = t
            break

    if not topic:
        return items

    keywords = topic.get("keywords", [])

    keywords = [k.lower() for k in keywords]

    filtered = []

    for item in items:

        text = (
            (item.get("title","") + " " +
             item.get("summary","") + " " +
             item.get("source",""))
        ).lower()

        if any(k in text for k in keywords):

            filtered.append(item)

    return filtered

# =====================================================
# Insight relevance engine
# =====================================================

def select_relevant_insights(news_items, insights):

    if not isinstance(news_items, list) or not isinstance(insights, list):
        return []

    text_blob = " ".join(
        f"{i.get('title','')} {i.get('summary','')} {i.get('source','')}"
        for i in news_items if isinstance(i, dict)
    ).lower()

    matched = []

    for ins in insights:

        if not isinstance(ins, dict):
            continue

        tags = ins.get("tags", [])

        if not isinstance(tags, list):
            continue

        tags = [str(t).lower().strip() for t in tags if t]

        if any(tag in text_blob for tag in tags):

            matched.append(ins)

    return matched


# =====================================================
# Citation engine
# =====================================================

def _build_citation_source_map(items, max_sources=12):

    source_map = {}
    ordered = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        url = (item.get("original_url") or item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        source = (item.get("source") or "").strip()

        # Normalise published date → "Month Day, Year" string
        raw_pub = item.get("published")
        published_str = ""
        if raw_pub:
            try:
                if isinstance(raw_pub, str):
                    from dateutil import parser as _dp
                    raw_pub = _dp.parse(raw_pub)
                published_str = raw_pub.strftime("%B %-d, %Y")
            except Exception:
                published_str = str(raw_pub)[:10]

        key = (url or title).lower().strip()
        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        ordered.append({
            "title": title,
            "source": source,
            "url": url,
            "published_at": published_str,
        })

        if len(ordered) >= max_sources:
            break

    for idx, item in enumerate(ordered, start=1):
        source_map[f"S{idx}"] = item

    return source_map


def _strip_ai_link_markers(report_text):

    text = report_text

    text = re.sub(r'【連結】\((https?://[^\s)]+)\)', '', text)
    text = re.sub(r'【連結】', '', text)
    text = re.sub(r'\(\s*https?://[^\s)]+\s*\)', '', text)

    return text


_SUPERSCRIPT_TABLE = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _to_superscript(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT_TABLE)


def _format_chicago_note(idx: int, src: dict) -> str:
    """
    Chicago Notes & Bibliography – Web page format:
    Author Last, First. "Title of Web Page." Website Name. Organization, Month Day, Year. URL.
    When no individual author is available, the source name acts as the publisher.
    """
    title = src.get("title", "").strip()
    source = src.get("source", "").strip()
    published_at = src.get("published_at", "").strip()
    url = src.get("url", "").strip()
    author = src.get("author", "").strip()  # optional field; usually absent for RSS

    sup = _to_superscript(idx)
    parts = []

    # Author (Last, First) – only if present
    if author:
        parts.append(author + ".")

    # Title of web page in double quotes
    if title:
        parts.append(f'"{title}."')

    # Website / publication name (italics not possible in plain text; omit italics)
    if source:
        parts.append(source + ".")

    # Publication date
    if published_at:
        parts.append(published_at + ".")

    # URL
    if url:
        parts.append(url + ".")

    return sup + " " + " ".join(parts) if parts else sup


def _render_citations(report_text, source_map, format_options):

    notes_style = format_options.get("notes", {}).get("style", "endnote")
    link_mode = format_options.get("links", {}).get("placement", "none")

    text = _strip_ai_link_markers(report_text)

    # 無論任何模式，都把 [Sx] 或 [ Sx ] 清掉或換成上標
    if notes_style == "none" and link_mode == "none":
        text = re.sub(r'\[\s*S\d+\s*\]', '', text)
        return text

    used_codes = []

    def replace_code(match):
        code = match.group(1)
        if code not in source_map:
            return ""
        if code not in used_codes:
            used_codes.append(code)
        idx = used_codes.index(code) + 1
        return _to_superscript(idx)

    text = re.sub(r'\[\s*(S\d+)\s*\]', replace_code, text)

    # 清理多餘空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if not used_codes:
        return text

    # Chicago 腳註（footnote 或 endnote 都用同一格式）
    section_title = "Notes" if notes_style == "footnote" else "Sources"
    lines = ["", "", section_title, ""]
    for idx, code in enumerate(used_codes, start=1):
        src = source_map[code]
        lines.append(_format_chicago_note(idx, src))

    return text + "\n" + "\n".join(lines)


# =====================================================
# AI report generation
# =====================================================

def _get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            pass

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    return OpenAI(api_key=api_key)


# =====================================================
# Report structure helpers
# =====================================================

REGION_ORDER = [
    "亞太地區",
    "亞西地區",
    "北美地區",
    "拉丁美洲及加勒比海",
    "歐洲地區",
    "非洲地區",
]


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _item_text(item):
    parts = [
        _safe_text(item.get("title")),
        _safe_text(item.get("summary")),
        _safe_text(item.get("source")),
        _safe_text(item.get("source_region")),
        " ".join(item.get("source_category", []) or []),
        _safe_text(item.get("source_description")),
    ]
    return " ".join(parts).lower()


def _contains_any(text, keywords):
    return any(k.lower() in text for k in keywords)


def _normalize_language_label(language):
    mapping = {
        "zh": "繁體中文",
        "繁體中文": "繁體中文",
        "簡體中文": "簡體中文",
        "英文": "English",
        "english": "English",
        "en": "English",
        "日文": "日本語",
        "ja": "日本語",
        "jp": "日本語",
    }
    return mapping.get(language, language or "繁體中文")


def _is_taiwan_china_related(item):
    text = _item_text(item)
    keywords = [
        "taiwan", "台灣", "台湾",
        "china", "中國", "中国", "中共", "北京", "beijing",
        "taipei", "臺北", "台北",
        "兩岸", "两岸", "cross-strait",
        "pla", "解放軍", "解放军",
        "國台辦", "国台办",
        "美中台", "台美中",
    ]
    return _contains_any(text, keywords)


def _is_taiwan_us_china(item):
    text = _item_text(item)
    tw_keywords = ["taiwan", "台灣", "台湾", "taipei", "臺北", "台北"]
    us_keywords = ["united states", "u.s.", "us ", "usa", "美國", "美国", "washington"]
    cn_keywords = ["china", "中國", "中国", "中共", "beijing", "北京"]

    has_tw = _contains_any(text, tw_keywords)
    has_us = _contains_any(text, us_keywords)
    has_cn = _contains_any(text, cn_keywords)

    return (has_tw and has_us) or (has_tw and has_cn) or (has_us and has_cn)


def _is_taiwan_security(item):
    text = _item_text(item)
    tw_keywords = ["taiwan", "台灣", "台湾", "taipei", "臺北", "台北"]
    sec_keywords = [
        "security", "national security", "defense", "defence", "military",
        "國安", "安全", "國防", "防務", "防卫", "防衛",
        "軍事", "軍演", "演習", "灰色地帶", "灰色地带",
        "blockade", "封鎖", "封锁", "missile", "飛彈", "飞弹",
        "cyber", "網攻", "网攻", "資安", "网络安全", "網路安全",
    ]
    return _contains_any(text, tw_keywords) and _contains_any(text, sec_keywords)


def _is_china_news(item):
    text = _item_text(item)
    keywords = [
        "china", "中國", "中国", "中共", "北京", "beijing",
        "pla", "解放軍", "解放军",
        "新華社", "新华社", "人民日報", "人民日报",
        "國台辦", "国台办", "中國外交部", "中国外交部",
    ]
    return _contains_any(text, keywords)


def _detect_region_for_item(item):
    text = _item_text(item)
    region = _safe_text(item.get("source_region")).lower()
    categories = [str(x).lower() for x in (item.get("source_category", []) or [])]

    def has_any(*values):
        values = [v.lower() for v in values]
        return (
            any(v in text for v in values)
            or any(v in region for v in values)
            or any(any(v in c for v in values) for c in categories)
        )

    if has_any(
        "亞太", "asia-pacific", "indo-pacific", "east asia", "southeast asia",
        "taiwan", "台灣", "台湾", "china", "中國", "中国", "japan", "日本",
        "korea", "韓國", "韩国", "philippines", "菲律賓", "菲律宾",
        "australia", "australian", "澳洲", "澳大利亞", "new zealand", "紐西蘭", "新西兰",
        "india", "印度"
    ):
        return "亞太地區"

    if has_any(
        "亞西", "west asia", "middle east", "中東", "中东",
        "turkey", "土耳其", "iran", "伊朗", "iraq", "伊拉克", "israel", "以色列",
        "gaza", "加薩", "加沙", "saudi", "沙烏地", "沙特",
        "syria", "敘利亞", "叙利亚", "yemen", "葉門", "也门",
        "uae", "阿聯", "阿联", "qatar", "卡達", "卡塔爾"
    ):
        return "亞西地區"

    if has_any(
        "北美", "north america",
        "united states", "u.s.", "usa", "美國", "美国",
        "canada", "加拿大", "mexico", "墨西哥"
    ):
        return "北美地區"

    if has_any(
        "latin america", "caribbean", "拉丁美洲", "加勒比海",
        "brazil", "巴西", "argentina", "阿根廷", "chile", "智利",
        "peru", "秘魯", "秘鲁", "colombia", "哥倫比亞", "哥伦比亚",
        "venezuela", "委內瑞拉", "委内瑞拉", "paraguay", "巴拉圭",
        "guatemala", "瓜地馬拉", "危地马拉", "haiti", "海地"
    ):
        return "拉丁美洲及加勒比海"

    if has_any(
        "europe", "歐洲", "欧洲",
        "eu", "european union", "歐盟", "欧洲联盟",
        "uk", "britain", "united kingdom", "英國", "英国",
        "france", "法國", "法国", "germany", "德國", "德国",
        "poland", "波蘭", "波兰", "ukraine", "烏克蘭", "乌克兰",
        "nato"
    ):
        return "歐洲地區"

    if has_any(
        "africa", "非洲",
        "south africa", "南非", "egypt", "埃及", "ethiopia", "衣索比亞", "埃塞俄比亚",
        "kenya", "肯亞", "肯尼亚", "nigeria", "奈及利亞", "尼日利亚",
        "sudan", "蘇丹", "sudanese"
    ):
        return "非洲地區"

    return "亞太地區"


def _dedupe_items(items, limit=None):
    output = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        key = (
            _safe_text(item.get("original_url") or item.get("url")).strip().lower()
            or _safe_text(item.get("title")).strip().lower()
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

        if limit and len(output) >= limit:
            break

    return output


def _group_items_for_report(items):
    groups = {
        "國際要聞": [],
        "台美中要聞": [],
        "台灣國安要聞": [],
        "中國要聞": [],
        "區域情勢": {
            region: {
                "區域要聞": [],
                "台灣與中國相關要聞": [],
            }
            for region in REGION_ORDER
        },
    }

    for item in items:
        if not isinstance(item, dict):
            continue

        text = _item_text(item)
        region = _detect_region_for_item(item)

        groups["國際要聞"].append(item)

        if _is_taiwan_us_china(item):
            groups["台美中要聞"].append(item)

        if _is_taiwan_security(item):
            groups["台灣國安要聞"].append(item)

        if _is_china_news(item):
            groups["中國要聞"].append(item)

        if _is_taiwan_china_related(item):
            groups["區域情勢"][region]["台灣與中國相關要聞"].append(item)
        else:
            groups["區域情勢"][region]["區域要聞"].append(item)

    # 各章節：選出被最多來源報導的前 4 個不同議題，每議題取 2 篇
    # → 每章最多 8 篇，但涵蓋至少 3-4 個獨立話題
    groups["國際要聞"]   = _select_diverse_topics(groups["國際要聞"],   n_topics=4, articles_per_topic=2)
    groups["台美中要聞"] = _select_diverse_topics(groups["台美中要聞"], n_topics=4, articles_per_topic=2)
    groups["台灣國安要聞"] = _select_diverse_topics(groups["台灣國安要聞"], n_topics=4, articles_per_topic=2)
    groups["中國要聞"]   = _select_diverse_topics(groups["中國要聞"],   n_topics=4, articles_per_topic=2)

    for region in REGION_ORDER:
        # 區域情勢：各子節選 3 個議題，每議題 1 篇（資料較少，保持精簡）
        groups["區域情勢"][region]["區域要聞"] = _select_diverse_topics(
            groups["區域情勢"][region]["區域要聞"], n_topics=3, articles_per_topic=1
        )
        groups["區域情勢"][region]["台灣與中國相關要聞"] = _select_diverse_topics(
            groups["區域情勢"][region]["台灣與中國相關要聞"], n_topics=3, articles_per_topic=1
        )

    return groups


def _format_item_block(label, items, item_to_sx=None):
    lines = [f"【{label}】"]
    if not items:
        lines.append("- 無明顯代表性新聞")
        return "\n".join(lines)

    for idx, item in enumerate(items, start=1):
        title = _safe_text(item.get("title"))
        source = _safe_text(item.get("source"))
        summary = _safe_text(item.get("summary"))
        content = _safe_text(item.get("content"))
        url = _safe_text(item.get("original_url") or item.get("url"))
        region = _safe_text(item.get("source_region"))

        # 找對應的 Sx 代碼，讓 AI 知道引用哪個
        sx_code = ""
        if item_to_sx:
            key = (url or title).lower().strip()
            sx_code = item_to_sx.get(key, "")
        sx_label = f" [{sx_code}]" if sx_code else ""

        lines.append(f"{idx}.{sx_label} 標題: {title}")
        if source:
            lines.append(f"   來源: {source}")
        if region:
            lines.append(f"   區域: {region}")
        if summary:
            lines.append(f"   摘要: {summary}")

        if content:
            preview = content[:2000]
            lines.append(f"   內文: {preview}")

        if url:
            lines.append(f"   連結: {url}")

    return "\n".join(lines)


def _build_news_data_block(groups, source_map=None):
    # 建立 url/title → Sx 的反查表，讓每篇新聞旁邊標 [S1] 等代碼
    item_to_sx = {}
    if source_map:
        for sx, info in source_map.items():
            key = (info.get("url") or info.get("title") or "").lower().strip()
            if key:
                item_to_sx[key] = sx

    blocks = []
    blocks.append(_format_item_block("國際要聞", groups["國際要聞"], item_to_sx))
    blocks.append(_format_item_block("台美中要聞", groups["台美中要聞"], item_to_sx))
    blocks.append(_format_item_block("台灣國安要聞", groups["台灣國安要聞"], item_to_sx))
    blocks.append(_format_item_block("中國要聞", groups["中國要聞"], item_to_sx))

    for region in REGION_ORDER:
        region_block = groups["區域情勢"][region]
        blocks.append(_format_item_block(f"{region}｜區域要聞", region_block["區域要聞"], item_to_sx))
        blocks.append(_format_item_block(f"{region}｜台灣與中國相關要聞", region_block["台灣與中國相關要聞"], item_to_sx))

    return "\n\n".join(blocks)


def _build_sources_block(items, limit=30):
    lines = ["", "## Sources", ""]
    deduped = _dedupe_items(items, limit=limit)

    for idx, item in enumerate(deduped, start=1):
        title = _safe_text(item.get("title"))
        source = _safe_text(item.get("source"))
        url = _safe_text(item.get("original_url") or item.get("url"))
        if url:
            lines.append(f"{idx}. {title} | {source} | {url}")

    return "\n".join(lines)


# =====================================================
# Multi-phase report constants & helpers
# =====================================================

# Maps source group keys → Chinese display names
_MULTIPHASE_GROUP_ZH = {
    "自訂台灣媒體":                "台灣媒體",
    "自訂國際媒體":                "國際媒體",
    "Asia-Pacific":               "亞太地區媒體",
    "Europe":                     "歐洲媒體",
    "West Asia":                  "亞西暨中東媒體",
    "Africa":                     "非洲媒體",
    "North America":              "北美媒體",
    "Latin America and Caribbean":"拉丁美洲媒體",
    "Latin America":              "拉丁美洲媒體",
    "中共官媒":                   "中共官媒",
}

# Ordered list for UI display
MULTIPHASE_GROUP_OPTIONS = [
    "自訂台灣媒體",
    "自訂國際媒體",
    "Asia-Pacific",
    "Europe",
    "West Asia",
    "Africa",
    "North America",
    "Latin America",
    "中共官媒",
]


def _get_item_source_group(item: dict) -> str:
    """Return the source-group key for a single item."""
    if item.get("source_type") == "cn_official":
        return "中共官媒"
    cats = item.get("source_category") or []
    if "中共官媒" in cats:
        return "中共官媒"
    if "自訂台灣媒體" in cats:
        return "自訂台灣媒體"
    if "自訂國際媒體" in cats:
        return "自訂國際媒體"
    if "全球媒體" in cats:
        continent = next((c for c in cats if c != "全球媒體"), None)
        if continent:
            # Normalise variants
            if "Latin" in continent:
                return "Latin America"
            if continent in ("Middle East", "West Asia"):
                return "West Asia"
            return continent
    return "其他"


def _generate_multiphase_synthesis(
    items,
    expert_items,
    insights_block,
    language,
    format_options,
    multiphase_groups,
    status_callback=None,
):
    """
    Multi-phase report generation:
      1. Group items by source group
      2. Generate a focused sub-report (~600-900 words) for each group
      3. Synthesise all sub-reports into one final strategic briefing
    """
    from utils.ai_briefing import generate_sub_briefing

    def _cb(event, detail=None, *args):
        if status_callback:
            try:
                status_callback(event, detail, *args)
            except Exception:
                pass

    language_label = _normalize_language_label(language)
    format_options = format_options or _load_format_options()

    # ── 1. Group items ──────────────────────────────────────────────────
    news_items = [i for i in items if i.get("source_type") != "expert"]

    # ── 0. Build global source_map FIRST so [Sx] codes persist end-to-end ──
    source_map = _build_citation_source_map(news_items, max_sources=30)
    item_to_sx: dict[str, str] = {}
    for sx, info in source_map.items():
        key = (info.get("url") or info.get("title") or "").lower().strip()
        if key:
            item_to_sx[key] = sx

    all_groups: dict[str, list] = {}
    for item in news_items:
        key = _get_item_source_group(item)
        all_groups.setdefault(key, []).append(item)

    # 使用者明確選取的群組全部列入，沒有文章的群組補空 list，
    # 確保 N 個選定群組 → N 份子報告（不因無資料而被靜默跳過）
    if multiphase_groups:
        selected_groups = {k: all_groups.get(k, []) for k in multiphase_groups}
    else:
        selected_groups = all_groups

    # ── 2. Generate sub-reports ─────────────────────────────────────────
    sub_reports: list[tuple[str, str]] = []
    eligible = list(selected_groups.items())   # 全部群組，含空群組
    total_g = len(eligible)

    _EMPTY_SUBREPORT_TEMPLATE = (
        "一、摘要\n本期【{name}】無相關新聞。\n\n"
        "二、國際要聞\n本期無相關新聞。\n\n"
        "三、台美中要聞\n本期無相關新聞。\n\n"
        "四、台灣國安要聞\n本期無相關新聞。\n\n"
        "五、中國要聞\n本期無相關新聞。\n\n"
        "六、區域情勢\n（一）亞太地區\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n"
        "（二）亞西地區\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n"
        "（三）北美地區\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n"
        "（四）拉丁美洲及加勒比海\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n"
        "（五）歐洲地區\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n"
        "（六）非洲地區\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n\n"
        "七、專家研析\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。\n\n"
        "八、研析\n1. 國際要聞研析\n本期無相關新聞。\n2. 台美中要聞研析\n本期無相關新聞。"
    )

    for done, (group_key, group_items) in enumerate(eligible, 1):
        group_name_zh = _MULTIPHASE_GROUP_ZH.get(group_key, group_key)
        _cb("stage", f"📝 子報告 {done}/{total_g}：{group_name_zh}（{len(group_items)} 篇）…")

        # 若該群組本期無任何文章，直接插入固定空白子報告，不消耗 AI token
        if not group_items:
            sub_reports.append((group_name_zh, _EMPTY_SUBREPORT_TEMPLATE.format(name=group_name_zh)))
            _cb("stage", f"⬜ 子報告 {done}/{total_g}：{group_name_zh}（本期無資料）")
            continue

        # 在群組內選出最多來源報導的 4 個不同議題（每議題取 2 篇），
        # 確保子報告涵蓋多元話題而非集中於同一事件
        diverse_items = _select_diverse_topics(group_items, n_topics=4, articles_per_topic=2)
        # Build structured news block WITH [Sx] codes so citations survive synthesis
        news_block = _format_item_block(group_name_zh, diverse_items, item_to_sx)

        sub_prompt = (
            f"以下是來自【{group_name_zh}】的新聞條目（每條已標注引用代碼 [S1][S2]... 請在報告中保留這些代碼）：\n\n"
            + news_block
        )

        try:
            sub_text = generate_sub_briefing(
                sub_prompt, group_name=group_name_zh, language=language_label
            )
            sub_reports.append((group_name_zh, sub_text))
            _cb("stage", f"✅ 子報告完成 {done}/{total_g}：{group_name_zh}")
        except Exception as e:
            print(f"[Multiphase] Sub-report failed for {group_name_zh}: {e}")
            sub_reports.append((group_name_zh, _EMPTY_SUBREPORT_TEMPLATE.format(name=group_name_zh)))

    if not sub_reports:
        return "No news items found.", []

    # ── 3. Build expert blocks (same as generate_report) ────────────────
    expert_names = list({
        it["expert"] for it in expert_items
        if isinstance(it, dict) and it.get("expert")
    })
    has_expert_data = bool(expert_names)
    expert_data_lines = []
    if has_expert_data:
        expert_data_lines.append("Expert Analysis Data:")
        for en in expert_names:
            this_items = [it for it in expert_items if it.get("expert") == en]
            expert_data_lines.append(f"\n[{en}]")
            for i, ei in enumerate(this_items[:5], 1):
                t = (ei.get("title") or "").strip()
                s = (ei.get("summary") or "").strip()
                if t:
                    expert_data_lines.append(f"  {i}. {t}")
                if s:
                    expert_data_lines.append(f"     {s[:300]}")
    expert_data_block = "\n".join(expert_data_lines)

    if has_expert_data:
        expert_guidance = (
            f"- 「七、專家研析」必須根據 Expert Analysis Data 撰寫，引用具名專家"
            f"（{', '.join(expert_names)}）的實際觀點並標注姓名。"
            "若某小節無明確觀點可引用，請寫「本期無相關專家意見。」。勿憑空虛構。"
        )
        expert_guidance_note = "若本期有專家資料，依上述 expert_guidance 引用；若無，請寫「本期無相關專家意見。」。"
    else:
        expert_guidance = ""
        expert_guidance_note = "本期無專家資料，請在兩個小節各寫「本期無專家資料。」。"

    sub_reports_block = "\n\n".join(
        f"═══ {name} ═══\n{text}" for name, text in sub_reports
    )

    _cb("stage", f"🤖 AI 綜整 {len(sub_reports)} 份子報告，生成最終簡報…")

    # ── 4. Synthesis prompt ─────────────────────────────────────────────
    synthesis_prompt = f"""
You are a senior strategic intelligence analyst.

Write a polished strategic intelligence briefing in {language_label}.
The output must read like a real analytical report, not like bullet-point news notes.

You have received sub-reports from {len(sub_reports)} regional analyst teams.
Use ONLY the information in these sub-reports as your source material.

Requirements:
1. Write in formal report style with coherent paragraphs.
2. Do NOT place URLs anywhere in the body text.
3. Do NOT write source names in brackets such as [DW.com], [Reuters.com], [BBC].
4. Synthesize information across sub-reports into broader strategic analysis.
5. Identify cross-regional patterns, escalating trends, and strategic implications.
6. CRITICAL — Citation codes [S1][S2][S3]... are embedded in the sub-reports. You MUST preserve every [Sx] code exactly as it appears — do NOT renumber, merge, drop, or invent any [Sx] marker. When you incorporate a fact from a sub-report that has a citation code, carry that exact code into the synthesis text. These codes are the only link to the source bibliography and must NOT be lost.
7. MANDATORY — Media outlets: NEVER use vague collective terms such as "歐洲媒體", "西方媒體", "美國媒體", "外媒". Always write the specific outlet name. On first mention, provide both Chinese and English, e.g. 德國之聲（Deutsche Welle）、法新社（Agence France-Presse, AFP）、路透社（Reuters）、《紐約時報》（New York Times）. This rule has NO exceptions.
8. MANDATORY — People: Every person mentioned must be preceded by their full official title or role. Use the conventionally established Chinese name form: Western figures use surname only (e.g., 川普、拜登、馬克宏、梅洛尼、奧斯汀); East Asian figures use the full name in Chinese characters (e.g., 岸田文雄、尹錫悅、習近平、賴清德). On first mention, follow with the full English/romanised name in parentheses. ADDITIONAL RULE for Japanese, Korean, and Vietnamese names: after the Chinese characters, add the romanised form in square brackets, e.g. 岸田文雄[Kishida Fumio]、尹錫悅[Yoon Suk-yeol]、阮富仲[Nguyễn Phú Trọng]. Format: [Title][Chinese name][Romanised]（Full English Name）. Examples: 美國總統川普（Donald Trump）、日本首相岸田文雄[Kishida Fumio]（Fumio Kishida）、韓國總統尹錫悅[Yoon Suk-yeol]（Yoon Suk-yeol）、越南國家主席阮富仲[Nguyễn Phú Trọng]（Nguyễn Phú Trọng）、中華民國總統賴清德（Lai Ching-te）. This rule has NO exceptions.
9a. MANDATORY — Expert names: whenever an expert or analyst is cited in 七、專家研析, render their name in bold (**Name**) and include their full title and affiliation on first mention. E.g. **美國智庫戰略與國際研究中心（CSIS）資深研究員王大維（David Wang）**。
9. MANDATORY — Organizations and institutions: On first mention, always provide both Chinese and English names. Format: Chinese name（English Name）. Examples: 北大西洋公約組織（NATO）、美國國務院（U.S. Department of State）、歐盟委員會（European Commission）、美國在台協會（American Institute in Taiwan, AIT）. This rule has NO exceptions.

Output structure:
【戰略情報簡報】

一、摘要

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞

六、區域情勢
（一）亞太地區
1. 國際要聞研析（若本期無亞太地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無亞太地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（二）亞西地區
1. 國際要聞研析（若本期無亞西地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無亞西地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（三）北美地區
1. 國際要聞研析（若本期無北美地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無北美地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（四）拉丁美洲及加勒比海
1. 國際要聞研析（若本期無拉丁美洲相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無拉丁美洲涉台涉中新聞，請寫：「本期無相關新聞。」）

（五）歐洲地區
1. 國際要聞研析（若本期無歐洲地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無歐洲地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（六）非洲地區
1. 國際要聞研析（若本期無非洲地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無非洲地區涉台涉中新聞，請寫：「本期無相關新聞。」）

七、專家研析
1. 國際要聞研析
2. 台美中要聞研析

八、研析
1. 國際要聞研析
2. 台美中要聞研析

Writing guidance:
- 「摘要」請用一小段說明本期最重要判斷。
- 「國際要聞」聚焦全球戰略層次的重要發展。
- 「台美中要聞」聚焦台灣、美國、中國三角互動及其戰略意涵。
- 「台灣國安要聞」聚焦軍事、灰帶、資安、國防、國安治理等。
- 「中國要聞」聚焦中國政治、外交、軍事、經濟、對外作為。
- 「區域情勢」六大區域必須全部列出，不得省略任何一個。每個區域各有兩小節：「1. 國際要聞研析」與「2. 台美中要聞研析」。若某區域本期無相關新聞，請在該小節寫「本期無相關新聞。」，絕不可完全省略任何區域或小節。
{expert_guidance}
- 「七、專家研析」分兩小節：「1. 國際要聞研析」與「2. 台美中要聞研析」。{expert_guidance_note}
- 「八、研析」分兩小節：「1. 國際要聞研析」與「2. 台美中要聞研析」，請提出跨章節的整體判斷、風險、趨勢、可能後續觀察重點。

Strategic Context:
{insights_block or "None"}

Sub-reports from regional analyst teams:
{sub_reports_block}
{("Expert Analysis Data:\n" + expert_data_block) if has_expert_data else ""}
"""

    client = _get_openai_client()
    response = client.responses.create(model="gpt-4.1-mini", input=synthesis_prompt)
    report = response.output_text

    report = re.sub(r'\[\s*(?!S\d+\s*\])([A-Za-z][^\]]{0,40})\]', '', report)
    report = re.sub(r'[ \t]+', ' ', report)

    # Use the global source_map built at the start (preserves [Sx] codes end-to-end)
    report = _render_citations(report, source_map, format_options)

    return report, items


def generate_report(
    start_time,
    end_time,
    selected_sources=None,
    selected_experts=None,
    language="zh",
    insights_text="",
    format_options=None,
    topic=None,
    status_callback=None,
    multiphase_groups=None,
):
    """
    status_callback(event, detail, *args) 是可選的 UI 回呼，格式：
      ("rss",          "來源名：N篇", completed, total, total_items)
      ("rss_progress", None,          completed, total, total_items)
      ("stage",        "訊息文字")
    """

    def _cb(event, detail=None, *args):
        if status_callback:
            try:
                status_callback(event, detail, *args)
            except Exception:
                pass

    sources = load_sources()
    n_src = len(_normalize_selected_sources(selected_sources, all_sources=sources))
    _cb("stage", f"⏳ 開始從 {n_src} 個來源抓取 RSS…")

    items = fetch_items_from_sources(
        selected_sources=selected_sources,
        all_sources=sources,
        limit_per_source=20,
        start_time=start_time,
        end_time=end_time,
        status_callback=status_callback,
    )
    _cb("stage", f"✅ RSS 抓取完成，共取得 {len(items)} 篇原始文章")

    # -------------------------------------------------
    # Chinese Official Media (direct scraping)
    # -------------------------------------------------
    try:
        normalized_sel = _normalize_selected_sources(selected_sources, all_sources=sources)
        cn_subsources = [
            _CN_SUBSOURCE_MAP[s["subsource"]]
            for s in normalized_sel
            if s.get("type") == "cn_official" and s.get("subsource") in _CN_SUBSOURCE_MAP
        ]
        if cn_subsources:
            _cb("stage", f"🇨🇳 爬取中共官媒（{', '.join(cn_subsources)}）…")
            cn_results = fetch_official_media_for_range(
                start_time=start_time,
                end_time=end_time,
                requested_subsources=cn_subsources,
            )
            cn_count = 0
            for subsource_items in cn_results.values():
                for cn_item in subsource_items:
                    raw_published = cn_item.get("published")
                    if isinstance(raw_published, datetime):
                        clamped = max(raw_published.replace(hour=12, minute=0, second=0), start_time)
                        clamped = min(clamped, end_time)
                    else:
                        clamped = start_time
                    items.append({
                        "title": cn_item.get("title", ""),
                        "url": cn_item.get("link", ""),
                        "original_url": cn_item.get("link", ""),
                        "source": cn_item.get("source_name", ""),
                        "published": clamped,
                        "summary": cn_item.get("summary", ""),
                        "content": cn_item.get("content", ""),
                        "source_region": cn_item.get("region", "中國"),
                        "source_category": cn_item.get("category", ["中共官媒"]),
                        "source_type": "cn_official",
                    })
                    cn_count += 1
            _cb("stage", f"✅ 中共官媒取得 {cn_count} 篇")
    except Exception as e:
        print(f"[Briefings] CN official media fetch failed: {e}")

    items = _filter_items_by_time_range(items, start_time, end_time)
    _cb("stage", f"🕐 時間範圍過濾後剩 {len(items)} 篇")

    items = filter_items_by_topic(items, topic)

    # -------------------------------------------------
    # 全文補抓
    # Google News RSS 已在查詢層完成關鍵字＋時間篩選，
    # 此處直接對所有文章補抓全文。
    # cn_official / 已有全文的文章直接保留，其餘按時間排序取前 60 篇。
    # -------------------------------------------------

    cn_items  = [i for i in items if i.get("source_type") == "cn_official" or i.get("content")]
    web_items = [i for i in items if i.get("source_type") != "cn_official" and not i.get("content")]

    # 按發佈時間排序（新 → 舊），最多補抓 60 篇
    web_items.sort(key=lambda x: x.get("published") or "", reverse=True)
    to_enrich = web_items[:60]
    _cb("stage", f"📄 補抓 {len(to_enrich)} 篇全文（共 {len(web_items)} 篇候選，10 個並行連線）…")

    def _enrich_one(item):
        url = item.get("original_url") or item.get("url") or ""
        if not url or item.get("content"):
            return item
        resolved = _resolve_google_news_url(url)
        content = _fetch_article_content(resolved)
        enriched = dict(item)
        enriched["original_url"] = resolved
        enriched["content"] = content
        return enriched

    try:
        with ThreadPoolExecutor(max_workers=10) as enrich_ex:
            enriched = list(enrich_ex.map(_enrich_one, to_enrich, timeout=90))
        items = enriched + cn_items
    except Exception as e:
        print(f"[Briefings] Article enrichment failed (using summaries): {e}")
        items = to_enrich + cn_items

    with_content = sum(1 for i in items if i.get("content"))
    _cb("stage", f"✅ 全文補抓完成：{with_content} / {len(items)} 篇有全文")

    # -------------------------------------------------
    # Expert Monitoring
    # -------------------------------------------------
    expert_items = []
    try:
        if selected_experts:
            expert_items = fetch_expert_items(selected_experts)
    except Exception as e:
        print(f"[Briefings] Expert fetch failed: {e}")

    if expert_items:
        for item in expert_items:
            if isinstance(item, dict):
                item.setdefault("source_region", "")
                item.setdefault("source_category", ["專家"])
        items.extend(expert_items)

    # -------------------------------------------------
    # Insight Engine
    # -------------------------------------------------
    insights = load_insights()

    if isinstance(insights, str):
        insights_block = insights.strip()
    else:
        relevant_insights = select_relevant_insights(items, insights)
        insight_lines = []

        if isinstance(relevant_insights, list):
            for i in relevant_insights:
                if not isinstance(i, dict):
                    continue

                title = str(i.get("title", "")).strip()
                content = str(i.get("content", "")).strip()

                if title and content:
                    insight_lines.append(f"- {title}: {content}")
                elif content:
                    insight_lines.append(f"- {content}")

        insights_block = "\n".join(insight_lines).strip()

    if insights_text:
        insights_block = f"{insights_block}\n{insights_text}".strip()

    if not items:
        return "No news items found.", []

    format_options = format_options or _load_format_options()

    # ── 多段生成模式分支 ─────────────────────────────────────────────────
    if multiphase_groups is not None:
        return _generate_multiphase_synthesis(
            items=items,
            expert_items=expert_items,
            insights_block=insights_block,
            language=language,
            format_options=format_options,
            multiphase_groups=multiphase_groups,
            status_callback=status_callback,
        )

    # 先建 source_map，再傳給 news_data_block，讓每篇新聞旁邊標 [Sx]
    source_map = _build_citation_source_map(items, max_sources=12)

    groups = _group_items_for_report(items)
    news_data_block = _build_news_data_block(groups, source_map=source_map)
    language_label = _normalize_language_label(language)

    # 建立專家分析資料區塊（僅在有真實 expert_items 時）
    expert_names = list({
        item["expert"] for item in expert_items
        if isinstance(item, dict) and item.get("expert")
    })
    has_expert_data = bool(expert_names)

    expert_data_lines = []
    if has_expert_data:
        expert_data_lines.append("Expert Analysis Data:")
        for expert_name in expert_names:
            this_expert_items = [i for i in expert_items if i.get("expert") == expert_name]
            expert_data_lines.append(f"\n[{expert_name}]")
            for i, ei in enumerate(this_expert_items[:5], 1):
                t = (ei.get("title") or "").strip()
                s = (ei.get("summary") or "").strip()
                if t:
                    expert_data_lines.append(f"  {i}. {t}")
                if s:
                    expert_data_lines.append(f"     {s[:300]}")
    expert_data_block = "\n".join(expert_data_lines)

    if has_expert_data:
        expert_guidance = (
            f"- 「七、專家研析」必須根據 Expert Analysis Data 撰寫，引用具名專家"
            f"（{', '.join(expert_names)}）的實際觀點並標注姓名。"
            "若某小節無明確觀點可引用，請寫「本期無相關專家意見。」。勿憑空虛構。"
        )
        expert_guidance_note = "若本期有專家資料，依上述 expert_guidance 引用；若無，請寫「本期無相關專家意見。」。"
    else:
        expert_guidance = ""
        expert_guidance_note = "本期無專家資料，請在兩個小節各寫「本期無專家資料。」。"

    prompt = f"""
You are a senior strategic intelligence analyst.

Write a polished strategic intelligence briefing in {language}.
The output must read like a real analytical report, not like bullet-point news notes.

Requirements:
1. Write in formal report style with coherent paragraphs.
2. Do NOT place URLs anywhere in the body text.
3. Do NOT write the string 【連結】 anywhere.
4. Do NOT write source names in brackets such as [DW.com], [Reuters.com], [BBC]. Only use [S1], [S2] style markers.
5. Use ONLY the provided news items as source material.
6. Strategic Context is analyst guidance only. Do NOT cite it as evidence.
7. Synthesize multiple news items into broader analysis instead of summarizing one article at a time.
8. When making a factual claim based on a source, append source markers like [S1], [S2].
9. You may cite multiple sources together, for example [S1][S3].
10. Only use source markers that exist in the provided News data.
11. Keep citations light and readable. Do not attach a citation to every single sentence unless necessary.
12. MANDATORY — Media outlets: NEVER use vague collective terms such as "歐洲媒體", "西方媒體", "美國媒體", "外媒". Always write the specific outlet name. On first mention, provide both Chinese and English, e.g. 德國之聲（Deutsche Welle）、法新社（Agence France-Presse, AFP）、路透社（Reuters）、《紐約時報》（New York Times）. This rule has NO exceptions.
13. MANDATORY — People: Every person mentioned must be preceded by their full official title or role. Use the conventionally established Chinese name form: Western figures use surname only (e.g., 川普、拜登、馬克宏、梅洛尼、奧斯汀); East Asian figures use the full name in Chinese characters (e.g., 岸田文雄、尹錫悅、習近平、賴清德). On first mention, follow with the full English/romanised name in parentheses. ADDITIONAL RULE for Japanese, Korean, and Vietnamese names: after the Chinese characters, add the romanised form in square brackets, e.g. 岸田文雄[Kishida Fumio]、尹錫悅[Yoon Suk-yeol]、阮富仲[Nguyễn Phú Trọng]. Format: [Title][Chinese name][Romanised]（Full English Name）. Examples: 美國總統川普（Donald Trump）、日本首相岸田文雄[Kishida Fumio]（Fumio Kishida）、韓國總統尹錫悅[Yoon Suk-yeol]（Yoon Suk-yeol）、越南國家主席阮富仲[Nguyễn Phú Trọng]（Nguyễn Phú Trọng）、中華民國總統賴清德（Lai Ching-te）. This rule has NO exceptions.
13a. MANDATORY — Expert names: whenever an expert or analyst is cited in 七、專家研析, render their name in bold (**Name**) and include their full title and affiliation on first mention. E.g. **美國智庫戰略與國際研究中心（CSIS）資深研究員王大維（David Wang）**。
14. MANDATORY — Organizations and institutions: On first mention, always provide both Chinese and English names. Format: Chinese name（English Name）. Examples: 北大西洋公約組織（NATO）、美國國務院（U.S. Department of State）、歐盟委員會（European Commission）、美國在台協會（American Institute in Taiwan, AIT）、中華民國國防部（Ministry of National Defense, ROC）. This rule has NO exceptions.

Output structure:
【戰略情報簡報】

一、摘要

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞

六、區域情勢
（一）亞太地區
1. 國際要聞研析（若本期無亞太地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無亞太地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（二）亞西地區
1. 國際要聞研析（若本期無亞西地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無亞西地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（三）北美地區
1. 國際要聞研析（若本期無北美地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無北美地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（四）拉丁美洲及加勒比海
1. 國際要聞研析（若本期無拉丁美洲相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無拉丁美洲涉台涉中新聞，請寫：「本期無相關新聞。」）

（五）歐洲地區
1. 國際要聞研析（若本期無歐洲地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無歐洲地區涉台涉中新聞，請寫：「本期無相關新聞。」）

（六）非洲地區
1. 國際要聞研析（若本期無非洲地區相關新聞，請寫：「本期無相關新聞。」）
2. 台美中要聞研析（若本期無非洲地區涉台涉中新聞，請寫：「本期無相關新聞。」）

七、專家研析
1. 國際要聞研析
2. 台美中要聞研析

八、研析
1. 國際要聞研析
2. 台美中要聞研析

Writing guidance:
- 「摘要」請用一小段說明本期最重要判斷。
- 「國際要聞」聚焦全球戰略層次的重要發展。
- 「台美中要聞」聚焦台灣、美國、中國三角互動及其戰略意涵。
- 「台灣國安要聞」聚焦軍事、灰帶、資安、國防、國安治理等。
- 「中國要聞」聚焦中國政治、外交、軍事、經濟、對外作為。
- 「區域情勢」六大區域必須全部列出，不得省略任何一個。每個區域各有兩小節：「1. 國際要聞研析」與「2. 台美中要聞研析」。若某區域本期無相關新聞，請在該小節寫「本期無相關新聞。」，絕不可完全省略任何區域或小節。
{expert_guidance}
- 「七、專家研析」分兩小節：「1. 國際要聞研析」與「2. 台美中要聞研析」。{expert_guidance_note}
- 「八、研析」分兩小節：「1. 國際要聞研析」與「2. 台美中要聞研析」，請提出跨章節的整體判斷、風險、趨勢、可能後續觀察重點。

Strategic Context:
{insights_block or "None"}

News data:
{news_data_block}
{("Expert Analysis Data:" + chr(10) + expert_data_block) if has_expert_data else ""}
"""

    _cb("stage", f"🤖 AI 生成簡報中（共 {len(items)} 篇文章進入分析）…")

    client = _get_openai_client()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    report = response.output_text

    # 清除 AI 可能生成的 [DW.com] / [BBC] 等來源標籤（非 [Sx] 格式）
    report = re.sub(r'\[\s*(?!S\d+\s*\])([A-Za-z][^\]]{0,40})\]', '', report)
    report = re.sub(r'[ \t]+', ' ', report)

    # 建立 citation source map
    source_map = _build_citation_source_map(items, max_sources=12)

    # 插入 citation
    report = _render_citations(report, source_map, format_options)

    return report, items

# =====================================================
# DOCX export
# =====================================================

def export_docx(report_text, path):

    from docx import Document

    doc = Document()

    for line in report_text.split("\n"):

        doc.add_paragraph(line)

    doc.save(path)


# =====================================================
# PDF export
# =====================================================

def export_pdf(report_text, path):

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(path, pagesize=A4)

    y = 800

    for line in report_text.split("\n"):

        c.drawString(40, y, line)

        y -= 20

        if y < 40:
            c.showPage()
            y = 800

    c.save()