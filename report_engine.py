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

# ── 全球媒體兩階段抓取：各語言台灣／美國／中國關鍵字 ──────────────────────────
# Phase 1：以語言對應關鍵字搜尋 Google News，只收有關鍵字的文章
_LANG_TW_KEYWORDS = {
    "zh":    "台灣 OR 美國 OR 中國 OR 台海 OR 解放軍 OR 中共 OR 習近平",
    "zh-TW": "台灣 OR 美國 OR 中國 OR 台海 OR 解放軍 OR 中共 OR 習近平",
    "zh-CN": "台湾 OR 美国 OR 中国 OR 台海 OR 解放军 OR 中共 OR 习近平",
    "en":    "Taiwan OR China OR \"United States\" OR PLA OR CCP OR \"Xi Jinping\" OR \"Indo-Pacific\"",
    "ja":    "台湾 OR 中国 OR アメリカ OR 解放軍 OR 習近平 OR 米中 OR 台湾海峡",
    "ko":    "대만 OR 중국 OR 미국 OR 인민해방군 OR 시진핑 OR 대만해협",
    "es":    "Taiwán OR China OR \"Estados Unidos\" OR \"Xi Jinping\" OR \"estrecho de Taiwán\"",
    "de":    "Taiwan OR China OR USA OR \"Xi Jinping\" OR Volksbefreiungsarmee OR Taiwanstraße",
    "fr":    "Taïwan OR Chine OR \"États-Unis\" OR \"Xi Jinping\" OR \"détroit de Taïwan\"",
    "ru":    "Тайвань OR Китай OR США OR НОАК OR \"Си Цзиньпин\" OR \"Тайваньский пролив\"",
    "it":    "Taiwan OR Cina OR \"Stati Uniti\" OR \"Xi Jinping\" OR \"stretto di Taiwan\"",
    "pt":    "Taiwan OR China OR \"Estados Unidos\" OR \"Xi Jinping\" OR \"Estreito de Taiwan\"",
    "ar":    "تايوان OR الصين OR \"الولايات المتحدة\" OR \"شي جين بينغ\" OR \"مضيق تايوان\"",
    "he":    "טייוואן OR סין OR \"ארצות הברית\" OR \"שי ג'ינפינג\" OR \"מיצר טייוואן\"",
    "cs":    "Tchaj-wan OR Čína OR \"Spojené státy\" OR \"Si Ťin-pching\" OR \"Tchajwanský průliv\"",
    "nl":    "Taiwan OR China OR \"Verenigde Staten\" OR \"Xi Jinping\" OR \"Straat van Taiwan\"",
    "id":    "Taiwan OR China OR \"Amerika Serikat\" OR \"Xi Jinping\" OR \"Selat Taiwan\"",
    "ms":    "Taiwan OR China OR \"Amerika Syarikat\" OR \"Xi Jinping\" OR \"Selat Taiwan\"",
    "pl":    "Tajwan OR Chiny OR \"Stany Zjednoczone\" OR \"Xi Jinping\" OR \"Cieśnina Tajwańska\"",
    "ro":    "Taiwan OR China OR \"Statele Unite\" OR \"Xi Jinping\" OR \"Strâmtoarea Taiwan\"",
    "sv":    "Taiwan OR Kina OR USA OR \"Xi Jinping\" OR Taiwansundet",
    "uk":    "Тайвань OR Китай OR США OR НВАК OR \"Сі Цзіньпін\" OR \"Тайванська протока\"",
    "tl":    "Taiwan OR China OR \"Estados Unidos\" OR \"Xi Jinping\" OR \"Strait ng Taiwan\"",
    "tr":    "Tayvan OR Çin OR \"Amerika Birleşik Devletleri\" OR \"Xi Jinping\" OR \"Tayvan Boğazı\"",
    "vi":    "\"Đài Loan\" OR \"Trung Quốc\" OR \"Hoa Kỳ\" OR PLA OR \"Tập Cận Bình\" OR \"eo biển Đài Loan\"",
    "th":    "ไต้หวัน OR จีน OR สหรัฐอเมริกา OR PLA OR สีจิ้นผิง OR ช่องแคบไต้หวัน",
    "hi":    "ताइवान OR चीन OR \"संयुक्त राज्य\" OR PLA OR \"शी जिनपिंग\" OR \"ताइवान जलडमरूमध्य\"",
    "sw":    "Taiwan OR China OR Marekani OR PLA OR \"Xi Jinping\" OR \"Mlango-Bahari wa Taiwan\"",
}

# Google News RSS 語言參數（hl=UI 語言, gl=地區, ceid=地區:語言）
_LANG_NEWS_PARAMS = {
    "zh":    {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
    "zh-TW": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
    "zh-CN": {"hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"},
    "en":    {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    "ja":    {"hl": "ja",    "gl": "JP", "ceid": "JP:ja"},
    "ko":    {"hl": "ko",    "gl": "KR", "ceid": "KR:ko"},
    "es":    {"hl": "es",    "gl": "ES", "ceid": "ES:es"},
    "de":    {"hl": "de",    "gl": "DE", "ceid": "DE:de"},
    "fr":    {"hl": "fr",    "gl": "FR", "ceid": "FR:fr"},
    "ru":    {"hl": "ru",    "gl": "RU", "ceid": "RU:ru"},
    "it":    {"hl": "it",    "gl": "IT", "ceid": "IT:it"},
    "pt":    {"hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"},
    "ar":    {"hl": "ar",    "gl": "SA", "ceid": "SA:ar"},
    "he":    {"hl": "iw",    "gl": "IL", "ceid": "IL:iw"},
    "cs":    {"hl": "cs",    "gl": "CZ", "ceid": "CZ:cs"},
    "nl":    {"hl": "nl",    "gl": "NL", "ceid": "NL:nl"},
    "id":    {"hl": "id",    "gl": "ID", "ceid": "ID:id"},
    "ms":    {"hl": "ms",    "gl": "MY", "ceid": "MY:ms"},
    "pl":    {"hl": "pl",    "gl": "PL", "ceid": "PL:pl"},
    "ro":    {"hl": "ro",    "gl": "RO", "ceid": "RO:ro"},
    "sv":    {"hl": "sv",    "gl": "SE", "ceid": "SE:sv"},
    "uk":    {"hl": "uk",    "gl": "UA", "ceid": "UA:uk"},
    "tl":    {"hl": "tl",    "gl": "PH", "ceid": "PH:tl"},
    "tr":    {"hl": "tr",    "gl": "TR", "ceid": "TR:tr"},
    "vi":    {"hl": "vi",    "gl": "VN", "ceid": "VN:vi"},
    "th":    {"hl": "th",    "gl": "TH", "ceid": "TH:th"},
    "hi":    {"hl": "hi",    "gl": "IN", "ceid": "IN:hi"},
    "sw":    {"hl": "sw",    "gl": "KE", "ceid": "KE:sw"},
}

# 標題後顯示語言標籤（非中文、非英文才標）
_LANG_DISPLAY_LABEL = {
    "ja": "[日文]",
    "ko": "[韓文]",
    "es": "[西班牙文]",
    "de": "[德文]",
    "fr": "[法文]",
    "ru": "[俄文]",
    "it": "[義大利文]",
    "pt": "[葡萄牙文]",
    "ar": "[阿拉伯文]",
    "he": "[希伯來文]",
    "cs": "[捷克文]",
    "nl": "[荷蘭文]",
    "id": "[印尼文]",
    "ms": "[馬來文]",
    "pl": "[波蘭文]",
    "ro": "[羅馬尼亞文]",
    "sv": "[瑞典文]",
    "uk": "[烏克蘭文]",
    "tl": "[他加祿文]",
    "tr": "[土耳其文]",
    "vi": "[越南文]",
    "th": "[泰文]",
    "hi": "[印地文]",
    "sw": "[斯瓦希里文]",
}

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


def _build_google_news_rss_for_domain(domain, start_time=None, end_time=None, keywords=None,
                                       lang_params=None):
    """
    建立 Google News RSS 查詢 URL。

    重要：Google News RSS 的 site: operator 單獨使用時回傳 0 結果。
    必須在 site: 前面加上關鍵字才能正常運作，格式：
        (kw1 OR kw2) site:domain when:Xd
    時間精確過濾由 _filter_items_by_time_range 在 client 端補做。

    lang_params: dict with hl/gl/ceid keys，用於多語言來源。
                 預設使用繁體中文（TW）參數。
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

    # 語言/地區參數
    p = lang_params or {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
    hl   = p.get("hl",   "zh-TW")
    gl   = p.get("gl",   "TW")
    ceid = p.get("ceid", "TW:zh-Hant")

    # 用 safe=':/' 確保 site: 的冒號不被編碼（編碼後 Google 不識別為 operator）
    query_encoded = quote(query, safe=':/')
    return f"https://news.google.com/rss/search?q={query_encoded}&hl={hl}&gl={gl}&ceid={ceid}"


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

        # 自訂專家：若有 rss_url 直接抓 RSS；否則用 expert_gnews_urls 生成所有查詢 URL
        if cat == "自訂專家":
            expert_name = src_name.strip()
            if not expert_name:
                return []
            direct_rss = (src.get("url") or "").strip()
            if direct_rss and direct_rss.startswith("http"):
                # 直接從自訂 RSS URL 抓取；時間過濾在 client 端由 _filter_items_by_time_range 完成
                fetched = _fetch_rss_items(direct_rss, expert_name, limit=limit_per_source)
            else:
                # 無 rss_url → 用 expert_gnews_urls 產生所有語系的 Google News 查詢
                from utils.loaders import expert_gnews_urls as _expert_gnews_urls
                # 從 src dict 重建最小 expert dict 以供 helper 使用
                _exp_dict = {
                    "name_zh": src.get("name_zh") or ("" if all(ord(c) < 128 for c in expert_name.replace(" ", "")) else expert_name),
                    "name_en": src.get("name_en") or (expert_name if all(ord(c) < 128 for c in expert_name.replace(" ", "")) else ""),
                    "region": src.get("region", ""),
                    "rss_url": "",
                }
                _url_pairs = _expert_gnews_urls(_exp_dict)
                if not _url_pairs:
                    # fallback: single URL from display name
                    _is_ascii = all(ord(c) < 128 for c in expert_name.replace(" ", ""))
                    _p = "hl=en-US&gl=US&ceid=US:en" if _is_ascii else "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                    _url_pairs = [("", f"https://news.google.com/rss/search?q={quote(chr(34)+expert_name+chr(34))}&{_p}")]
                # Query all URLs, add time params, deduplicate by URL
                fetched = []
                _seen_urls: set = set()
                for _label, _base_url in _url_pairs:
                    # Inject time params into query string
                    _sep = "&" if "?" in _base_url else "?"
                    _timed_url = _base_url
                    if start_time or end_time:
                        # Extract existing q= param and append date range
                        import urllib.parse as _up
                        _parsed = _up.urlparse(_base_url)
                        _qs = _up.parse_qs(_parsed.query)
                        _q = _qs.get("q", [""])[0]
                        if start_time:
                            _q += f" after:{start_time.strftime('%Y/%m/%d')}"
                        if end_time:
                            _q += f" before:{end_time.strftime('%Y/%m/%d')}"
                        _qs["q"] = [_q]
                        _new_query = _up.urlencode({k: v[0] for k, v in _qs.items()})
                        _timed_url = _up.urlunparse(_parsed._replace(query=_new_query))
                    for _item in _fetch_rss_items(_timed_url, expert_name, limit=limit_per_source):
                        _key = (_item.get("url") or _item.get("link") or "").lower().strip()
                        if _key and _key not in _seen_urls:
                            _seen_urls.add(_key)
                            fetched.append(_item)
            # 時間過濾（補足 direct RSS 無法在 URL 帶時間的情況）
            fetched = _filter_items_by_time_range(fetched, start_time, end_time)
            for item in fetched:
                item["source"] = expert_name   # 來源標籤統一顯示專家名字
                item["source_category"] = cats
                item["source_region"] = src.get("region", "")
                item["source_type"] = "expert"
            return fetched

        src_type  = src.get("type", "rss")
        url_field = src.get("url", "")

        # ── 全球媒體：兩階段抓取 ──────────────────────────────────────────────
        is_global = any(c in _GLOBAL_MEDIA_CATEGORIES for c in cats)
        if is_global:
            src_lang    = (src.get("language") or "en").strip().lower()
            src_country = (src.get("country") or src.get("region") or "").strip()
            lang_label  = _LANG_DISPLAY_LABEL.get(src_lang, "")  # "" = zh/en（不標）
            lang_params = _LANG_NEWS_PARAMS.get(src_lang, _LANG_NEWS_PARAMS["en"])
            lang_kw     = _LANG_TW_KEYWORDS.get(src_lang, _LANG_TW_KEYWORDS["en"])

            seen_urls: set = set()
            all_phase: list = []

            def _tag_items(items, kw_match: bool):
                for item in items:
                    item["source"]          = src_name
                    item["source_category"] = cats
                    item["source_region"]   = src_country or src.get("region", "")
                    item["source_type"]     = src_type
                    item["source_language"] = src_lang
                    item["source_country"]  = src_country
                    item["tw_keyword_match"] = kw_match
                    # 非中文、非英文來源：在標題後加語言標籤
                    if lang_label and not item.get("_lang_tagged"):
                        title = item.get("title", "")
                        if title and not title.endswith(lang_label):
                            item["title"] = f"{title} {lang_label}"
                        item["_lang_tagged"] = True
                return items

            # Phase 1：關鍵字搜尋，只收涉台美中文章（用語言專屬關鍵字）
            domain = (src.get("domain") or src.get("site")
                      or _extract_news_domain(url_field) or url_field)
            if domain:
                domain = domain.lower().replace("www.", "")
                p1_url = _build_google_news_rss_for_domain(
                    domain, start_time=start_time, end_time=end_time,
                    keywords=lang_kw, lang_params=lang_params
                )
                p1_items = _fetch_rss_items(p1_url, src_name, limit=limit_per_source)
                _tag_items(p1_items, kw_match=True)
                for item in p1_items:
                    key = (item.get("original_url") or item.get("url") or "").lower().strip()
                    if key:
                        if key not in seen_urls:
                            seen_urls.add(key)
                            all_phase.append(item)
                    else:
                        # 無 URL 的條目仍保留（不用 key 做去重）
                        all_phase.append(item)

            # Phase 2：直接 RSS（一般新聞），不做關鍵字過濾
            direct_rss = (
                src.get("rss") or src.get("rss_url")
                or src.get("feed") or src.get("feed_url")
            )
            if not direct_rss and src_type == "rss" and url_field.startswith("http"):
                direct_rss = url_field
            if direct_rss:
                p2_items = _fetch_rss_items(direct_rss, src_name, limit=limit_per_source)
                _tag_items(p2_items, kw_match=False)
                for item in p2_items:
                    key = (item.get("original_url") or item.get("url") or "").lower().strip()
                    if key:
                        if key not in seen_urls:
                            seen_urls.add(key)
                            all_phase.append(item)
                    else:
                        # 無 URL 的條目仍保留
                        all_phase.append(item)

            return all_phase
        # ── 一般來源（非全球媒體）：原有邏輯 ────────────────────────────────

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
            # 無法解析日期（小型媒體常見，日期格式非標準）→ 預設保留
            # 避免直接 RSS Phase 2 的文章因日期缺失而全被丟棄
            filtered.append(item)
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

        # 用 url+title 組合鍵，允許同一 URL 不同標題的文章（如新聞聯播各則）各自建立尾註
        key = f"{url}||{title}".lower().strip()
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
            "edition": (item.get("edition") or "").strip(),
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


def _to_superscript(n: int) -> str:
    """Return citation marker as [n] bracket format.
    Bracket style avoids run-together ambiguity when multiple citations appear
    consecutively (e.g. [11][12][13] vs ¹¹¹²¹³).
    """
    return f"[{n}]"


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
    edition = src.get("edition", "").strip()

    sup = _to_superscript(idx)
    parts = []

    # Author (Last, First) – only if present
    if author:
        parts.append(author + ".")

    # Title of web page in double quotes
    if title:
        parts.append(f'"{title}."')

    # Website / publication name, optionally with edition (版/則)
    if source:
        source_label = f"{source}（{edition}）" if edition else source
        parts.append(source_label + ".")

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
    lines = ["", "", f"## {section_title}", ""]
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
    # 全球媒體 Phase 1 命中者（已以關鍵字過濾）直接視為台美中相關
    if item.get("tw_keyword_match"):
        return True
    text = _item_text(item)
    tw_keywords = [
        # 中文
        "taiwan", "台灣", "台湾", "taipei", "臺北", "台北",
        # 日文
        "台湾", "台湾海峡",
        # 韓文
        "대만", "대만해협",
        # 西班牙文
        "taiwán",
        # 法文
        "taïwan",
        # 俄文
        "тайвань",
        # 阿拉伯文
        "تايوان",
    ]
    us_keywords = [
        "united states", "u.s.", "us ", "usa", "美國", "美国", "washington",
        "アメリカ", "미국", "estados unidos", "états-unis", "etats-unis",
        "vereinigte staaten", "соединённые штаты", "сша",
        "الولايات المتحدة", "ארצות הברית",
    ]
    cn_keywords = [
        "china", "中國", "中国", "中共", "beijing", "北京",
        "中国", "중국", "베이징", "chine", "pékin", "kina",
        "cina", "китай", "пекин", "الصين", "סין",
    ]

    has_tw = _contains_any(text, tw_keywords)
    has_us = _contains_any(text, us_keywords)
    has_cn = _contains_any(text, cn_keywords)

    return (has_tw and has_us) or (has_tw and has_cn) or (has_us and has_cn)


def _is_taiwan_security(item):
    # 全球媒體 Phase 1 命中者加上安全關鍵字才算
    text = _item_text(item)
    tw_keywords = [
        "taiwan", "台灣", "台湾", "taipei", "臺北", "台北",
        "台湾", "대만", "taiwán", "taïwan", "тайвань", "تايوان",
    ]
    sec_keywords = [
        "security", "national security", "defense", "defence", "military",
        "國安", "安全", "國防", "防務", "防卫", "防衛",
        "軍事", "軍演", "演習", "灰色地帶", "灰色地带",
        "blockade", "封鎖", "封锁", "missile", "飛彈", "飞弹",
        "cyber", "網攻", "网攻", "資安", "网络安全", "網路安全",
        # 多語言安全詞
        "安全保障", "軍事演習",  # 日文
        "국방", "안보", "군사",  # 韓文
        "defensa", "seguridad", "militar",  # 西班牙文
        "défense", "sécurité", "militaire",  # 法文
        "sicherheit", "verteidigung", "militär",  # 德文
        "безопасность", "оборона", "военный",  # 俄文
        "الأمن", "الدفاع", "العسكري",  # 阿拉伯文
    ]
    return _contains_any(text, tw_keywords) and _contains_any(text, sec_keywords)


def _is_china_news(item):
    text = _item_text(item)
    keywords = [
        "china", "中國", "中国", "中共", "北京", "beijing",
        "pla", "解放軍", "解放军",
        "新華社", "新华社", "人民日報", "人民日报",
        "國台辦", "国台办", "中國外交部", "中国外交部",
        # 多語言
        "중국", "베이징",  # 韓文
        "中国", "北京",    # 日文（同中文字元，已涵蓋）
        "chine", "pékin", "péking",  # 法文
        "kina", "peking",  # 瑞典/挪威文
        "cina", "pechino",  # 義大利文
        "chinesisch", "volksrepublik china",  # 德文
        "китай", "пекин",  # 俄文/烏克蘭文
        "الصين", "بكين",  # 阿拉伯文
        "סין", "בייג'ינג",  # 希伯來文
        "çin", "pekin",  # 土耳其文
        "trung quốc", "bắc kinh",  # 越南文
        "จีน", "ปักกิ่ง",  # 泰文
        "चीन", "बीजिंग",  # 印地語
    ]
    return _contains_any(text, keywords)


def _is_china_external(item):
    """中國對外情勢：涉及中國外交、軍事行動、對外投資、對他國關係的報導。"""
    if not _is_china_news(item):
        return False
    text = _item_text(item)
    external_keywords = [
        # 外交
        "外交", "diplomacy", "diplomatic", "foreign policy", "外交部", "外長",
        "王毅", "外交關係", "雙邊", "多邊", "外交部長",
        # 軍事對外
        "解放軍", "pla", "軍事演習", "軍演", "演習", "武器", "飛彈", "導彈",
        "aircraft carrier", "航空母艦", "naval", "海軍", "warship", "軍艦",
        "air force", "空軍", "missile", "軍備",
        # 貿易/投資/制裁
        "trade", "貿易", "tariff", "關稅", "sanction", "制裁",
        "export", "import", "出口", "進口", "investment", "投資",
        "belt and road", "一帶一路", "bri",
        # 對台對美關係
        "taiwan strait", "台海", "taiwan", "台灣", "一中", "one china",
        "south china sea", "南海", "east china sea", "東海",
        # 國際組織
        "united nations", "聯合國", "un ", "brics", "金磚", "g20", "g7",
        "apec", "asean", "東盟", "東協",
        # 對外聲明
        "spokesperson", "發言人", "statement", "聲明",
    ]
    return _contains_any(text, external_keywords)


def _is_china_domestic(item):
    """中國內部情勢：涉及中國黨政、內政、經濟、社會、人權的報導。"""
    if not _is_china_news(item):
        return False
    text = _item_text(item)
    domestic_keywords = [
        # 黨政
        "ccp", "共產黨", "communist party", "中央", "習近平", "xi jinping",
        "politburo", "政治局", "national people's congress", "全國人民代表大會",
        "npc", "人大", "cppcc", "政協", "中央委員會",
        # 內政/治理
        "domestic", "內政", "governance", "治理", "regulation", "監管",
        "censorship", "審查", "crackdown", "鎮壓", "anti-corruption", "反腐",
        "propaganda", "宣傳",
        # 經濟
        "gdp", "economy", "經濟", "inflation", "通膨", "通脹", "deflation",
        "unemployment", "失業", "housing", "房地產", "real estate", "債務", "debt",
        "yuan", "人民幣", "renminbi", "pboc", "人民銀行",
        # 社會
        "protest", "抗議", "riot", "暴亂", "social unrest", "社會動盪",
        "human rights", "人權", "civil society", "公民社會",
        "xinjiang", "新疆", "tibet", "西藏", "hong kong", "香港",
        "uyghur", "維吾爾", "uighur",
        # 人口/科技
        "population", "人口", "birth rate", "出生率", "tech", "科技",
        "ai ", "artificial intelligence", "人工智慧",
    ]
    # 排除純外交/軍事對外的文章（避免與 external 重複），
    # 但如果文章同時有內政關鍵字，仍歸入 domestic
    return _contains_any(text, domestic_keywords)


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
        "中國對外情勢": [],
        "中國內部情勢": [],
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

        region = _detect_region_for_item(item)

        groups["國際要聞"].append(item)

        if _is_taiwan_us_china(item):
            groups["台美中要聞"].append(item)

        if _is_taiwan_security(item):
            groups["台灣國安要聞"].append(item)

        if _is_china_news(item):
            groups["中國要聞"].append(item)
            # 子節分類（一篇可同時歸入兩個子節）
            if _is_china_external(item):
                groups["中國對外情勢"].append(item)
            if _is_china_domestic(item):
                groups["中國內部情勢"].append(item)
            # 若兩個子節都沒命中，歸入外部情勢（預設）
            if not _is_china_external(item) and not _is_china_domestic(item):
                groups["中國對外情勢"].append(item)

        if _is_taiwan_china_related(item):
            groups["區域情勢"][region]["台灣與中國相關要聞"].append(item)
        else:
            groups["區域情勢"][region]["區域要聞"].append(item)

    # 各章節：選出被最多來源報導的前 5 個不同議題，每議題取 2 篇
    # → 每章最多 10 篇，涵蓋 4-5 個獨立話題，確保各章有足夠素材
    groups["國際要聞"]     = _select_diverse_topics(groups["國際要聞"],     n_topics=5, articles_per_topic=2)
    groups["台美中要聞"]   = _select_diverse_topics(groups["台美中要聞"],   n_topics=5, articles_per_topic=2)
    groups["台灣國安要聞"] = _select_diverse_topics(groups["台灣國安要聞"], n_topics=5, articles_per_topic=2)
    # 第五章：兩個子節各取 5 個議題 × 2 篇 = 各 10 篇
    groups["中國對外情勢"] = _select_diverse_topics(groups["中國對外情勢"], n_topics=5, articles_per_topic=2)
    groups["中國內部情勢"] = _select_diverse_topics(groups["中國內部情勢"], n_topics=5, articles_per_topic=2)
    # 整體中國要聞仍保留（供舊版 fallback 使用）
    groups["中國要聞"]     = _select_diverse_topics(groups["中國要聞"],     n_topics=5, articles_per_topic=2)

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
        src_country  = _safe_text(item.get("source_country"))
        src_language = _safe_text(item.get("source_language"))

        # 找對應的 Sx 代碼，讓 AI 知道引用哪個（與 _build_citation_source_map 使用相同鍵）
        sx_code = ""
        if item_to_sx:
            key = f"{url}||{title}".lower().strip()
            sx_code = item_to_sx.get(key, "")
        sx_label = f" [{sx_code}]" if sx_code else ""

        lines.append(f"{idx}.{sx_label} 標題: {title}")
        # 來源＋國家（供 AI 在報告中提及媒體國籍）
        source_meta = source
        if src_country:
            source_meta = f"{source}（{src_country}）" if source else f"（{src_country}）"
        if source_meta:
            lines.append(f"   來源: {source_meta}")
        if src_language and src_language not in ("zh", "zh-tw", "zh-cn", "en"):
            lang_display = _LANG_DISPLAY_LABEL.get(src_language, f"[{src_language}]")
            lines.append(f"   語言: {lang_display}")
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
    # 第五章：中國要聞分兩個子節
    blocks.append(_format_item_block("中國要聞｜（一）中國對外情勢", groups.get("中國對外情勢", []), item_to_sx))
    blocks.append(_format_item_block("中國要聞｜（二）中國內部情勢", groups.get("中國內部情勢", []), item_to_sx))

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


# =====================================================
# Segmented-report constants & helpers
# =====================================================

# 19 sections for the segmented report.
# Each section independently searches Google News with topic-specific keywords.
# kw_zh = primary Chinese-language query; kw_en = secondary English-language query.
_SEGMENTED_SECTIONS = [
    # ── Standalone sections (no parent hierarchy) ────────────────────────────
    {
        "id": "intl_news",
        "label": "二、國際要聞",
        "section_path": "二、國際要聞",
        "display_parent": None,   # standalone → emit ## label directly
        "display_sub": None,
        "display_leaf": None,
        "kw_zh": "國際局勢 OR 全球外交 OR 聯合國 OR 地緣政治 OR 國際安全 OR 戰爭 OR 武裝衝突",
        "kw_en": "\"international affairs\" OR \"global politics\" OR diplomacy OR \"United Nations\" OR geopolitics OR war OR conflict OR sanctions",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "tw_us_cn",
        "label": "三、台美中要聞",
        "section_path": "三、台美中要聞",
        "display_parent": None,
        "display_sub": None,
        "display_leaf": None,
        "kw_zh": "台灣 OR 台海 OR 兩岸 OR 台美關係 OR 美中關係 OR 中共 OR 台美中",
        "kw_en": "Taiwan OR \"Taiwan strait\" OR \"cross-strait\" OR \"US-China\" OR \"US-Taiwan\" OR CCP OR \"Sino-American\"",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "tw_security",
        "label": "四、台灣國安要聞",
        "section_path": "四、台灣國安要聞",
        "display_parent": None,
        "display_sub": None,
        "display_leaf": None,
        "kw_zh": "台灣 AND (國安 OR 國防 OR 解放軍 OR 軍演 OR 灰色地帶 OR 資安 OR 飛彈 OR 網路攻擊)",
        "kw_en": "Taiwan AND (security OR defense OR PLA OR military OR \"gray zone\" OR cyber OR missile OR drills)",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    # ── 五、中國要聞 (parent → sub, no leaf) ────────────────────────────────
    {
        "id": "cn_external",
        "label": "五（一）中國對外情勢",
        "section_path": "五、中國要聞｜（一）中國對外情勢",
        "display_parent": "五、中國要聞",
        "display_sub": "（一）中國對外情勢",
        "display_leaf": None,
        "kw_zh": "中國 AND (外交 OR 軍事 OR 南海 OR 東海 OR 台海 OR 制裁 OR 對外政策 OR 王毅 OR 軍演)",
        "kw_en": "China AND (diplomacy OR military OR \"South China Sea\" OR \"East China Sea\" OR Taiwan OR sanctions OR \"foreign policy\" OR PLA OR \"belt and road\")",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "cn_domestic",
        "label": "五（二）中國內部情勢",
        "section_path": "五、中國要聞｜（二）中國內部情勢",
        "display_parent": "五、中國要聞",
        "display_sub": "（二）中國內部情勢",
        "display_leaf": None,
        "kw_zh": "中國 AND (習近平 OR 共產黨 OR 國內經濟 OR 人權 OR 新疆 OR 香港 OR 西藏 OR 房地產 OR 政治局)",
        "kw_en": "China AND (\"Xi Jinping\" OR CCP OR economy OR \"human rights\" OR Xinjiang OR \"Hong Kong\" OR Tibet OR \"real estate\" OR Politburo OR crackdown)",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    # ── 六、區域情勢 (parent → sub → leaf) ──────────────────────────────────
    {
        "id": "asia_pacific_intl",
        "label": "六（一）亞太地區－國際要聞",
        "section_path": "六、區域情勢｜（一）亞太地區｜1. 國際要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（一）亞太地區",
        "display_leaf": "1. 國際要聞",
        "kw_zh": "亞太 OR 日本 OR 韓國 OR 澳洲 OR 印度 OR 東南亞 OR 東協 OR 印太 OR 菲律賓 OR 越南 OR 印尼",
        "kw_en": "Asia-Pacific OR Japan OR Korea OR Australia OR India OR \"Southeast Asia\" OR ASEAN OR \"Indo-Pacific\" OR Philippines OR Vietnam OR Indonesia",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "asia_pacific_twcn",
        "label": "六（一）亞太地區－台美中要聞",
        "section_path": "六、區域情勢｜（一）亞太地區｜2. 台美中要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（一）亞太地區",
        "display_leaf": "2. 台美中要聞",
        "kw_zh": "(日本 OR 韓國 OR 澳洲 OR 印度 OR 東南亞 OR 菲律賓 OR 越南) AND (台灣 OR 中國 OR 美國 OR 兩岸)",
        "kw_en": "(Japan OR Korea OR Australia OR India OR \"Southeast Asia\" OR Philippines OR Vietnam) AND (Taiwan OR China OR \"United States\" OR \"cross-strait\")",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "west_asia_intl",
        "label": "六（二）亞西地區－國際要聞",
        "section_path": "六、區域情勢｜（二）亞西地區｜1. 國際要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（二）亞西地區",
        "display_leaf": "1. 國際要聞",
        "kw_zh": "中東 OR 以色列 OR 伊朗 OR 沙烏地阿拉伯 OR 加薩 OR 伊拉克 OR 敘利亞 OR 土耳其 OR 葉門",
        "kw_en": "\"Middle East\" OR Israel OR Iran OR Saudi OR Gaza OR Iraq OR Syria OR Turkey OR Yemen OR Lebanon OR Palestine",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "west_asia_twcn",
        "label": "六（二）亞西地區－台美中要聞",
        "section_path": "六、區域情勢｜（二）亞西地區｜2. 台美中要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（二）亞西地區",
        "display_leaf": "2. 台美中要聞",
        "kw_zh": "(中東 OR 以色列 OR 伊朗 OR 沙烏地 OR 土耳其) AND (台灣 OR 中國 OR 美國)",
        "kw_en": "(\"Middle East\" OR Israel OR Iran OR Saudi OR Turkey OR Gulf) AND (Taiwan OR China OR \"United States\")",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "north_am_intl",
        "label": "六（三）北美地區－國際要聞",
        "section_path": "六、區域情勢｜（三）北美地區｜1. 國際要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（三）北美地區",
        "display_leaf": "1. 國際要聞",
        "kw_zh": "美國 OR 加拿大 OR 川普 OR 白宮 OR 美國國會 OR 美國外交",
        "kw_en": "\"United States\" OR Canada OR Trump OR \"White House\" OR Congress OR Pentagon OR \"State Department\" OR Washington",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "north_am_twcn",
        "label": "六（三）北美地區－台美中要聞",
        "section_path": "六、區域情勢｜（三）北美地區｜2. 台美中要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（三）北美地區",
        "display_leaf": "2. 台美中要聞",
        "kw_zh": "(美國 OR 加拿大) AND (台灣 OR 中國 OR 台海 OR 兩岸)",
        "kw_en": "(\"United States\" OR Canada) AND (Taiwan OR China OR \"Taiwan Strait\" OR \"cross-strait\" OR CCP)",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "latin_am_intl",
        "label": "六（四）拉丁美洲及加勒比海－國際要聞",
        "section_path": "六、區域情勢｜（四）拉丁美洲及加勒比海｜1. 國際要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（四）拉丁美洲及加勒比海",
        "display_leaf": "1. 國際要聞",
        "kw_zh": "拉丁美洲 OR 巴西 OR 阿根廷 OR 哥倫比亞 OR 委內瑞拉 OR 古巴 OR 加勒比海 OR 智利 OR 秘魯",
        "kw_en": "\"Latin America\" OR Brazil OR Argentina OR Colombia OR Venezuela OR Cuba OR Caribbean OR Chile OR Peru OR Mexico",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "latin_am_twcn",
        "label": "六（四）拉丁美洲及加勒比海－台美中要聞",
        "section_path": "六、區域情勢｜（四）拉丁美洲及加勒比海｜2. 台美中要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（四）拉丁美洲及加勒比海",
        "display_leaf": "2. 台美中要聞",
        "kw_zh": "(拉丁美洲 OR 巴西 OR 阿根廷 OR 哥倫比亞 OR 智利) AND (台灣 OR 中國 OR 美國)",
        "kw_en": "(\"Latin America\" OR Brazil OR Argentina OR Chile OR Colombia OR Peru) AND (Taiwan OR China OR \"United States\")",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "europe_intl",
        "label": "六（五）歐洲地區－國際要聞",
        "section_path": "六、區域情勢｜（五）歐洲地區｜1. 國際要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（五）歐洲地區",
        "display_leaf": "1. 國際要聞",
        "kw_zh": "歐洲 OR 歐盟 OR 北約 OR 烏克蘭 OR 俄羅斯 OR 英國 OR 德國 OR 法國 OR 波蘭",
        "kw_en": "Europe OR EU OR NATO OR Ukraine OR Russia OR UK OR Germany OR France OR Poland OR Macron OR Scholz OR Zelensky",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "europe_twcn",
        "label": "六（五）歐洲地區－台美中要聞",
        "section_path": "六、區域情勢｜（五）歐洲地區｜2. 台美中要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（五）歐洲地區",
        "display_leaf": "2. 台美中要聞",
        "kw_zh": "(歐洲 OR 歐盟 OR 北約 OR 英國 OR 德國 OR 法國 OR 烏克蘭) AND (台灣 OR 中國)",
        "kw_en": "(Europe OR EU OR NATO OR UK OR Germany OR France OR Poland OR Ukraine) AND (Taiwan OR China)",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "africa_intl",
        "label": "六（六）非洲地區－國際要聞",
        "section_path": "六、區域情勢｜（六）非洲地區｜1. 國際要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（六）非洲地區",
        "display_leaf": "1. 國際要聞",
        "kw_zh": "非洲 OR 奈及利亞 OR 南非 OR 肯亞 OR 衣索比亞 OR 埃及 OR 蘇丹 OR 剛果",
        "kw_en": "Africa OR Nigeria OR \"South Africa\" OR Kenya OR Ethiopia OR Egypt OR Sudan OR Congo OR Ghana OR Tanzania OR Sahel",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "africa_twcn",
        "label": "六（六）非洲地區－台美中要聞",
        "section_path": "六、區域情勢｜（六）非洲地區｜2. 台美中要聞研析",
        "display_parent": "六、區域情勢",
        "display_sub": "（六）非洲地區",
        "display_leaf": "2. 台美中要聞",
        "kw_zh": "(非洲 OR 奈及利亞 OR 南非 OR 肯亞 OR 埃及 OR 衣索比亞) AND (台灣 OR 中國 OR 美國)",
        "kw_en": "(Africa OR Nigeria OR \"South Africa\" OR Kenya OR Egypt OR Ethiopia OR Sudan) AND (Taiwan OR China OR \"United States\")",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    # ── 七、專家研析 (parent → sub, no leaf) ────────────────────────────────
    {
        "id": "expert_intl",
        "label": "七（一）專家研析－國際要聞",
        "section_path": "七、專家研析｜1. 國際要聞研析",
        "display_parent": "七、專家研析",
        "display_sub": "1. 國際要聞研析",
        "display_leaf": None,
        "kw_zh": "(智庫 OR 研析 OR 學者 OR 評論員) AND (國際 OR 全球 OR 安全 OR 外交 OR 地緣政治)",
        "kw_en": "(\"think tank\" OR analysis OR scholar OR commentary OR expert OR analyst) AND (international OR global OR security OR diplomacy OR geopolitics)",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
    {
        "id": "expert_twcn",
        "label": "七（二）專家研析－台美中要聞",
        "section_path": "七、專家研析｜2. 台美中要聞研析",
        "display_parent": "七、專家研析",
        "display_sub": "2. 台美中要聞研析",
        "display_leaf": None,
        "kw_zh": "(智庫 OR 研析 OR 學者 OR 評論員) AND (台灣 OR 中國 OR 台海 OR 兩岸 OR 美中)",
        "kw_en": "(\"think tank\" OR analysis OR expert OR scholar OR commentary) AND (Taiwan OR China OR \"Taiwan strait\" OR \"cross-strait\" OR \"US-China\")",
        "lp_zh": {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
        "lp_en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    },
]


def _fetch_items_for_section(section: dict, start_time=None, end_time=None,
                              limit_per_query: int = 6) -> list:
    """
    Independently fetch articles for one segmented-report section.
    Runs two Google News searches (zh + en) and deduplicates by URL.
    """
    results: list = []
    seen_urls: set = set()

    def _run_query(kw: str, lp: dict):
        # Use a bare Google News search (no site: restriction) so every section
        # gets results even if the user's configured sources are narrow.
        p = lp or {"hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
        hl   = p.get("hl", "zh-TW")
        gl   = p.get("gl", "TW")
        ceid = p.get("ceid", "TW:zh-Hant")

        when_str = "when:3d"
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

        query = f"({kw})"
        if when_str:
            query += f" {when_str}"
        query_encoded = quote(query, safe=':/')
        url = f"https://news.google.com/rss/search?q={query_encoded}&hl={hl}&gl={gl}&ceid={ceid}"
        return _fetch_rss_items(url, section["label"], limit=limit_per_query)

    # Chinese query
    for item in _run_query(section["kw_zh"], section.get("lp_zh")):
        key = (item.get("original_url") or item.get("url") or "").lower().strip()
        if key:
            if key not in seen_urls:
                seen_urls.add(key)
                item["section_id"] = section["id"]
                results.append(item)
        else:
            item["section_id"] = section["id"]
            results.append(item)

    # English query (supplement if still < limit)
    for item in _run_query(section["kw_en"], section.get("lp_en")):
        key = (item.get("original_url") or item.get("url") or "").lower().strip()
        if key:
            if key not in seen_urls:
                seen_urls.add(key)
                item["section_id"] = section["id"]
                results.append(item)
        else:
            item["section_id"] = section["id"]
            results.append(item)

    return results


def _generate_segmented_final_report(
    section_mini_reports: list,
    language_label: str,
    insights_block: str = "",
    status_callback=None,
) -> str:
    """
    Given 19 section mini-reports, ask AI to write:
    - 一、摘要
    - 八、研析 / 1. 國際要聞研析 / 2. 台美中要聞研析
    then assemble the full report.
    """
    from utils.ai_briefing import get_client

    def _cb(msg):
        if status_callback:
            try:
                status_callback("stage", msg)
            except Exception:
                pass

    _cb("🤖 AI 綜整 19 份章節小報告，撰寫摘要與研析…")

    # Build a compact summary of all 19 mini-reports for the AI
    mini_block = "\n\n".join(
        f"═══ {label} ═══\n{text}"
        for label, text in section_mini_reports
    )

    synthesis_prompt = f"""You are a senior strategic intelligence analyst.

Below are 19 section mini-reports covering all chapters of a strategic intelligence briefing.
Your task is to write ONLY the following three parts (do NOT re-write the other chapters):

1. 一、摘要（1-2 paragraphs: most important strategic judgements of this issue）
2. 八、研析
   1. 國際要聞研析 — Analyse international developments EXCLUDING Taiwan-US-China (台美中) dynamics.
      Cover: global multilateral affairs, regional flashpoints (Europe, Middle East, Asia-Pacific, Americas,
      Africa), major-power competition among parties other than the Taiwan-US-China triangle.
      Do NOT discuss Taiwan independence/sovereignty, cross-strait relations, or US-China rivalry here.
   2. 台美中要聞研析 — Analyse ONLY Taiwan-US-China strategic dynamics:
      cross-strait relations, US-Taiwan ties, US-China rivalry, PRC military posture toward Taiwan,
      Taiwan's national security environment. Do NOT repeat content already in 國際要聞研析.

Write in {language_label}, in formal analytical prose (NOT bullet points).
Do NOT repeat the content of the 19 section reports — synthesize and elevate.
Do NOT place URLs in the body text.
MANDATORY: cite specific outlet names (Chinese + English on first mention).
MANDATORY: cite specific people with their titles.

Strategic Context:
{insights_block or "None"}

Section mini-reports:
{mini_block}

Output format (ONLY these three sections):
一、摘要

八、研析
1. 國際要聞研析

2. 台美中要聞研析
"""

    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ── 章節關鍵字比對輔助函式 ─────────────────────────────────────────────────

def _split_top_level(s: str, sep: str) -> list:
    """Split string by sep only at top level (not inside parentheses)."""
    parts: list = []
    depth = 0
    current: list = []
    i = 0
    while i < len(s):
        if s[i] == '(':
            depth += 1
            current.append(s[i])
            i += 1
        elif s[i] == ')':
            depth -= 1
            current.append(s[i])
            i += 1
        elif depth == 0 and s[i:i + len(sep)] == sep:
            parts.append("".join(current))
            current = []
            i += len(sep)
        else:
            current.append(s[i])
            i += 1
    if current:
        parts.append("".join(current))
    return parts


def _eval_section_query(text: str, query: str) -> bool:
    """
    Evaluate a Google-News-style AND/OR keyword query against text.
    Handles:
      "A OR B OR C"
      "A AND B"
      "A AND (B OR C)"
      "(A OR B) AND (C OR D)"
    """
    query = query.strip()
    if not query:
        return False
    text_lower = text.lower()

    # Try top-level OR split
    or_parts = _split_top_level(query, " OR ")
    if len(or_parts) > 1:
        return any(_eval_section_query(text_lower, p.strip()) for p in or_parts)

    # Try top-level AND split
    and_parts = _split_top_level(query, " AND ")
    if len(and_parts) > 1:
        return all(_eval_section_query(text_lower, p.strip()) for p in and_parts)

    # Single term or parenthesised group
    inner = query.strip("() \"'")
    if " OR " in inner or " AND " in inner:
        return _eval_section_query(text_lower, inner)

    kw = inner.lower()
    return bool(kw) and kw in text_lower


def _classify_items_to_sections(all_items: list, sections: list) -> dict:
    """
    Classify items into section buckets by matching kw_zh / kw_en queries
    against each item's title + summary.  An item may appear in multiple sections.
    Title-matched items are sorted first within each bucket so that the most
    directly relevant articles lead each section.
    Returns: {section_id: [item, ...]}
    """
    # Each bucket entry is (title_matched: bool, item)
    raw_buckets: dict = {sec["id"]: [] for sec in sections}
    for item in all_items:
        title     = item.get("title")   or ""
        summary   = item.get("summary") or ""
        full_text = f"{title} {summary}"
        for sec in sections:
            kw_zh = sec.get("kw_zh", "")
            kw_en = sec.get("kw_en", "")
            # Only add if full text (title+summary) matches
            if (_eval_section_query(full_text, kw_zh)
                    or _eval_section_query(full_text, kw_en)):
                # Title-only match = higher priority
                title_match = bool(
                    _eval_section_query(title, kw_zh)
                    or _eval_section_query(title, kw_en)
                )
                raw_buckets[sec["id"]].append((title_match, item))

    # Sort: title-matched items first (sort key 0), summary-only matches second (sort key 1)
    result: dict = {}
    for sec_id, entries in raw_buckets.items():
        entries.sort(key=lambda x: 0 if x[0] else 1)
        result[sec_id] = [item for _, item in entries]
    return result


def _cap_items_per_source(items: list, max_per_source: int = 3,
                           max_total: int = 12) -> list:
    """
    Select up to max_total items from a sorted list, ensuring no single source
    contributes more than max_per_source items.  The input order (title-match
    first) is preserved so the best items are kept.
    """
    source_counts: dict = {}
    result: list = []
    for item in items:
        src = (item.get("source") or "").strip().lower()
        count = source_counts.get(src, 0)
        if count >= max_per_source:
            continue
        source_counts[src] = count + 1
        result.append(item)
        if len(result) >= max_total:
            break
    return result


def _enrich_items_with_content(items: list, max_workers: int = 12) -> None:
    """
    Fetch full article text for every item that lacks content, in parallel.
    Updates each item dict in-place with "content".  Network failures are
    silently swallowed — the item retains its RSS summary as fallback.
    """
    to_fetch = [
        item for item in items
        if not item.get("content")
        and (item.get("original_url") or item.get("url"))
    ]
    if not to_fetch:
        return

    def _one(item):
        url = (item.get("original_url") or item.get("url") or "").strip()
        content = _fetch_article_content(url, max_chars=6000)
        if content:
            item["content"] = content

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_one, to_fetch))


def generate_segmented_report(
    start_time,
    end_time,
    selected_sources=None,
    selected_experts=None,
    language: str = "zh",
    insights_text: str = "",
    format_options=None,
    status_callback=None,
):
    """
    分段報告：先從設定的 RSS 來源統一抓取文章，再按各章節關鍵字分類，
    各章節獨立生成小報告，最後 AI 撰寫「一、摘要」和「八、研析」，組裝成完整報告。
    """
    from utils.ai_briefing import generate_section_mini_report

    def _cb(event, detail=None, *args):
        if status_callback:
            try:
                status_callback(event, detail, *args)
            except Exception:
                pass

    language_label = _normalize_language_label(language)
    insights_block = insights_text or ""
    total_sections = len(_SEGMENTED_SECTIONS)
    section_mini_reports: list = []  # [(label, text), ...]
    format_options = format_options or _load_format_options()

    # ── 1. 從 RSS 來源抓取所有文章（與 generate_report 相同流程） ───────────
    sources = load_sources()
    _sel = selected_sources or []
    n_src = len(_normalize_selected_sources(_sel, all_sources=sources))
    _cb("stage", f"⏳ 分段報告：從 {n_src} 個來源抓取 RSS…")

    all_items = fetch_items_from_sources(
        selected_sources=_sel,
        all_sources=sources,
        limit_per_source=30,
        start_time=start_time,
        end_time=end_time,
        status_callback=status_callback,
    )
    _cb("stage", f"✅ RSS 抓取完成，共取得 {len(all_items)} 篇文章")

    # ── 1a. 抓取專家 RSS / Google News（若有選取專家）────────────────────────
    if selected_experts:
        _cb("stage", f"🔍 抓取專家文章…")
        try:
            expert_items = fetch_expert_items(selected_experts)
            if expert_items:
                for _ei in expert_items:
                    if isinstance(_ei, dict):
                        _ei.setdefault("source_region", "")
                        _ei.setdefault("source_category", ["自訂專家"])
                all_items = expert_items + all_items   # expert items優先排序
                _cb("stage", f"✅ 專家文章：{len(expert_items)} 篇已加入分析池")
        except Exception as _e:
            print(f"[Segmented] Expert fetch failed: {_e}")

    # ── 1b. 時間區間過濾（對 direct RSS 來源補做，Google News 搜尋已在 URL 帶時間）──
    all_items = _filter_items_by_time_range(all_items, start_time, end_time)
    _cb("stage", f"🕐 時間過濾後剩 {len(all_items)} 篇")

    # ── 1c. 建立全局 source_map（同 _generate_multiphase_synthesis 做法）──
    #   先建 source_map，再傳給 _format_item_block，讓 AI 能使用 [Sx] 引用
    source_map = _build_citation_source_map(all_items, max_sources=30)
    item_to_sx: dict[str, str] = {}
    for sx, info in source_map.items():
        key = f"{info.get('url') or ''}||{info.get('title') or ''}".lower().strip()
        if key:
            item_to_sx[key] = sx

    # ── 2. 按章節關鍵字分類（標題命中優先） ──────────────────────────────
    _cb("stage", "📂 按章節關鍵字分類文章…")
    section_buckets = _classify_items_to_sections(all_items, _SEGMENTED_SECTIONS)
    classified = sum(1 for item in all_items
                     if any(item in section_buckets[s["id"]] for s in _SEGMENTED_SECTIONS))
    _cb("stage", f"📂 分類完成：{classified}/{len(all_items)} 篇分入各章節")

    # ── 2b. 每章節套用來源多樣性上限（每來源最多 3 篇），收集所有入選文章 ──
    selected_per_section: dict = {}
    all_selected_unique: list = []
    _seen_selected_urls: set = set()
    for _sec in _SEGMENTED_SECTIONS:
        _raw = section_buckets.get(_sec["id"], [])
        _chosen = _cap_items_per_source(_raw, max_per_source=3, max_total=12)
        selected_per_section[_sec["id"]] = _chosen
        for _item in _chosen:
            _url = (_item.get("original_url") or _item.get("url") or "").lower().strip()
            if _url and _url not in _seen_selected_urls:
                _seen_selected_urls.add(_url)
                all_selected_unique.append(_item)

    # ── 2c. 並行抓取所有入選文章的全文（Title-first → 全文 pipeline）──────
    _cb("stage", f"📄 並行抓取 {len(all_selected_unique)} 篇入選文章全文…")
    _enrich_items_with_content(all_selected_unique, max_workers=12)
    _with_content = sum(1 for i in all_selected_unique if i.get("content"))
    _cb("stage", f"✅ 全文取得：{_with_content}/{len(all_selected_unique)} 篇有全文，其餘用摘要")

    # ── 3. 各章節生成小報告 ────────────────────────────────────────────────
    _cb("stage", f"📰 分段報告：共 {total_sections} 個章節，各自生成小報告…")

    section_mini_secs: list = []   # parallel list of section dicts (for assembly hierarchy)

    for idx, sec in enumerate(_SEGMENTED_SECTIONS, 1):
        label = sec["label"]
        items = selected_per_section.get(sec["id"], [])

        _cb("rss", f"{label}：{len(items)} 篇", idx, total_sections, len(items))

        # Build news block for this section — pass item_to_sx so AI gets [Sx] codes
        news_block = _format_item_block(label, items, item_to_sx)

        # Generate mini-report for this section
        _cb("stage", f"✍️  [{idx}/{total_sections}] AI 撰寫：{label}…")
        try:
            mini_text = generate_section_mini_report(
                section_path=sec["section_path"],
                section_label=label,
                news_block=news_block,
                language=language_label,
            )
        except Exception as e:
            print(f"[Segmented] Section mini-report failed for {label}: {e}")
            mini_text = f"本期資料不足，無法生成{label}小報告。"

        section_mini_reports.append((label, mini_text))
        section_mini_secs.append(sec)
        _cb("stage", f"✅ [{idx}/{total_sections}] 完成：{label}")

    # ── 4. AI 撰寫「一、摘要」和「八、研析」 ─────────────────────────────
    synthesis_text = _generate_segmented_final_report(
        section_mini_reports=section_mini_reports,
        language_label=language_label,
        insights_block=insights_block,
        status_callback=status_callback,
    )

    # ── 5. 組裝最終報告 ────────────────────────────────────────────────────
    _cb("stage", "📄 組裝完整分段報告…")
    report_lines = ["公情綜整報告", ""]

    # Extract 一、摘要 / 八、研析 from synthesis
    summary_text = ""
    analysis_text = ""
    if "八、研析" in synthesis_text:
        parts = synthesis_text.split("八、研析", 1)
        summary_text = parts[0].strip()
        analysis_text = "八、研析\n" + parts[1].strip()
    else:
        summary_text = synthesis_text.strip()
        analysis_text = "八、研析\n1. 國際要聞研析\n本期無相關分析。\n\n2. 台美中要聞研析\n本期無相關分析。"

    # Strip AI's own "一、摘要" header (if present) to avoid duplication with our ## heading
    for _prefix in ("一、摘要\n\n", "一、摘要\n", "一、摘要"):
        if summary_text.startswith(_prefix):
            summary_text = summary_text[len(_prefix):].strip()
            break

    # 一、摘要 with explicit ## heading
    report_lines.append("## 一、摘要")
    report_lines.append("")
    report_lines.append(summary_text)
    report_lines.append("")

    # Insert section mini-reports with hierarchy-aware headings
    # display_parent → "## " (14pt bold, emitted once per unique parent)
    # display_sub    → "### " (13pt bold, emitted once per unique sub within a parent)
    # display_leaf   → "#### " (12pt bold italic, always emitted)
    # standalone (display_parent is None) → "## label"
    _last_parent: str | None = "__UNSET__"
    _last_sub: str | None = "__UNSET__"
    for sec, (label, text) in zip(section_mini_secs, section_mini_reports):
        dp = sec.get("display_parent")
        ds = sec.get("display_sub")
        dl = sec.get("display_leaf")

        if dp is None:
            # Standalone section — emit ## label heading, reset hierarchy tracking
            report_lines.append(f"## {label}")
            _last_parent = None
            _last_sub = None
        else:
            # Emit parent heading if it changed
            if dp != _last_parent:
                report_lines.append(f"## {dp}")
                report_lines.append("")
                _last_parent = dp
                _last_sub = "__UNSET__"  # reset sub when parent changes
            # Emit sub heading if present and changed
            if ds and ds != _last_sub:
                report_lines.append(f"### {ds}")
                report_lines.append("")
                _last_sub = ds
            # Emit leaf heading if present
            if dl:
                report_lines.append(f"#### {dl}")

        report_lines.append("")
        report_lines.append(text)
        report_lines.append("")

    report_lines.append(f"## 八、研析")
    report_lines.append("")
    # analysis_text already starts with "八、研析\n..." so strip that prefix to avoid duplication
    analysis_body = analysis_text.replace("八、研析\n", "", 1).strip()
    report_lines.append(analysis_body)

    final_report = "\n".join(report_lines)

    # ── 6. 套用來源引用渲染（同 generate_report / _generate_multiphase_synthesis）
    # 先清除 AI 可能自行寫入的 [文字] 標記（字母開頭），再清除純數字括號如 [1] [12]
    # （純數字括號是 AI 自行編號而非合法 [Sx] 代碼，若保留會與尾註編號對不上）
    final_report = re.sub(r'\[\s*(?!S\d+\s*\])([A-Za-z][^\]]{0,40})\]', '', final_report)
    final_report = re.sub(r'\[(\d{1,3})\]', '', final_report)   # strip AI-written [1] [2]…[999]
    final_report = re.sub(r'[ \t]+', ' ', final_report)
    final_report = _render_citations(final_report, source_map, format_options)

    # ── 7. 強制補上尾註（確保即使 AI 未引用 [Sx]，仍在報告末附上來源清單）──
    # 先判斷 _render_citations 是否已加入 Notes/Sources；若已加入則不重複
    if source_map and "## Notes" not in final_report and "## Sources" not in final_report:
        endnote_lines = ["", "## 尾註", ""]
        # 使用循序編號 1, 2, 3… 而非原始 S-number（S1=1, S5=5 等無意義映射）
        for seq_idx, (_sx, info) in enumerate(source_map.items(), start=1):
            endnote_lines.append(_format_chicago_note(seq_idx, info))
        final_report = final_report + "\n" + "\n".join(endnote_lines)

    return final_report, all_items


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
    Section-by-section report generation (replaces the old source-group sub-report pipeline).

    For each of the 19 report sections, the FULL item pool is classified by the same
    keyword queries used in generate_segmented_report.  Each section gets its own
    dedicated AI call (generate_section_mini_report), so every section has access to
    every relevant article regardless of source group.

    Finally, 一、摘要 and 八、研析 are written from all 19 section texts and the
    complete report is assembled.
    """
    from utils.ai_briefing import generate_section_mini_report

    def _cb(event, detail=None, *args):
        if status_callback:
            try:
                status_callback(event, detail, *args)
            except Exception:
                pass

    language_label = _normalize_language_label(language)
    format_options = format_options or _load_format_options()

    # ── 1. Separate news items from expert items ─────────────────────────
    news_items = [i for i in items if i.get("source_type") != "expert"]

    if not news_items:
        return "No news items found.", []

    # ── 2. Build global source_map (same as single / segmented report) ───
    source_map = _build_citation_source_map(news_items, max_sources=30)
    item_to_sx: dict[str, str] = {}
    for sx, info in source_map.items():
        key = f"{info.get('url') or ''}||{info.get('title') or ''}".lower().strip()
        if key:
            item_to_sx[key] = sx

    # ── 3. Classify all items into the 19 sections by keyword matching ───
    _cb("stage", "📂 按章節關鍵字分類文章…")
    section_buckets = _classify_items_to_sections(news_items, _SEGMENTED_SECTIONS)
    classified = sum(1 for item in news_items
                     if any(item in section_buckets[s["id"]] for s in _SEGMENTED_SECTIONS))
    _cb("stage", f"📂 分類完成：{classified}/{len(news_items)} 篇分入各章節")

    # Per-section source-diversity cap (same as generate_segmented_report)
    selected_per_section: dict = {}
    for _sec in _SEGMENTED_SECTIONS:
        _raw = section_buckets.get(_sec["id"], [])
        _chosen = _cap_items_per_source(_raw, max_per_source=3, max_total=12)
        # For expert sections, also append actual expert feed items
        if _sec["id"] in ("expert_intl", "expert_twcn") and expert_items:
            expert_pool = [it for it in expert_items if isinstance(it, dict)]
            # Avoid duplicates already in chosen
            chosen_urls = {(it.get("url") or "").lower() for it in _chosen}
            for ei in expert_pool:
                if (ei.get("url") or "").lower() not in chosen_urls:
                    _chosen.append(ei)
                    chosen_urls.add((ei.get("url") or "").lower())
        selected_per_section[_sec["id"]] = _chosen

    # ── 4. Generate each section mini-report ────────────────────────────
    total_sections = len(_SEGMENTED_SECTIONS)
    section_mini_reports: list = []
    section_mini_secs: list = []

    for idx, sec in enumerate(_SEGMENTED_SECTIONS, 1):
        label = sec["label"]
        sec_items = selected_per_section.get(sec["id"], [])

        _cb("rss", f"{label}：{len(sec_items)} 篇", idx, total_sections, len(sec_items))
        _cb("stage", f"✍️  [{idx}/{total_sections}] AI 撰寫：{label}（{len(sec_items)} 篇）…")

        news_block = _format_item_block(label, sec_items, item_to_sx)

        try:
            mini_text = generate_section_mini_report(
                section_path=sec["section_path"],
                section_label=label,
                news_block=news_block,
                language=language_label,
            )
        except Exception as e:
            print(f"[Multiphase] Section mini-report failed for {label}: {e}")
            mini_text = f"本期資料不足，無法生成{label}小報告。"

        section_mini_reports.append((label, mini_text))
        section_mini_secs.append(sec)
        _cb("stage", f"✅ [{idx}/{total_sections}] 完成：{label}")

    # ── 5. AI writes 摘要 + 八、研析 from all 19 section mini-reports ────
    synthesis_text = _generate_segmented_final_report(
        section_mini_reports=section_mini_reports,
        language_label=language_label,
        insights_block=insights_block,
        status_callback=status_callback,
    )

    # ── 6. Assemble final report ─────────────────────────────────────────
    _cb("stage", "📄 組裝完整綜合報告…")
    report_lines = ["【戰略情報簡報】", ""]

    # Extract 一、摘要 and 八、研析 from synthesis
    summary_text = ""
    analysis_text = ""
    if "八、研析" in synthesis_text:
        parts = synthesis_text.split("八、研析", 1)
        summary_text = parts[0].strip()
        analysis_text = "八、研析\n" + parts[1].strip()
    else:
        summary_text = synthesis_text.strip()
        analysis_text = "八、研析\n1. 國際要聞研析\n本期無相關分析。\n\n2. 台美中要聞研析\n本期無相關分析。"

    # Strip AI's own "一、摘要" header if present to avoid duplication
    for _prefix in ("一、摘要\n\n", "一、摘要\n", "一、摘要"):
        if summary_text.startswith(_prefix):
            summary_text = summary_text[len(_prefix):].strip()
            break

    report_lines.append("## 一、摘要")
    report_lines.append("")
    report_lines.append(summary_text)
    report_lines.append("")

    # Insert section mini-reports with hierarchy-aware headings
    _last_parent: str | None = "__UNSET__"
    _last_sub: str | None = "__UNSET__"
    for sec, (label, text) in zip(section_mini_secs, section_mini_reports):
        dp = sec.get("display_parent")
        ds = sec.get("display_sub")
        dl = sec.get("display_leaf")

        if dp is None:
            report_lines.append(f"## {label}")
            _last_parent = None
            _last_sub = None
        else:
            if dp != _last_parent:
                report_lines.append(f"## {dp}")
                report_lines.append("")
                _last_parent = dp
                _last_sub = "__UNSET__"
            if ds and ds != _last_sub:
                report_lines.append(f"### {ds}")
                report_lines.append("")
                _last_sub = ds
            if dl:
                report_lines.append(f"#### {dl}")

        report_lines.append("")
        report_lines.append(text)
        report_lines.append("")

    report_lines.append("## 八、研析")
    report_lines.append("")
    analysis_body = analysis_text.replace("八、研析\n", "", 1).strip()
    report_lines.append(analysis_body)

    final_report = "\n".join(report_lines)

    # ── 7. Clean up and render citations ────────────────────────────────
    final_report = re.sub(r'\[\s*(?!S\d+\s*\])([A-Za-z][^\]]{0,40})\]', '', final_report)
    final_report = re.sub(r'\[(\d{1,3})\]', '', final_report)
    final_report = re.sub(r'[ \t]+', ' ', final_report)
    final_report = _render_citations(final_report, source_map, format_options)

    # Force endnotes if not already present
    if source_map and "## Notes" not in final_report and "## Sources" not in final_report:
        endnote_lines = ["", "## 尾註", ""]
        for seq_idx, (_sx, info) in enumerate(source_map.items(), start=1):
            endnote_lines.append(_format_chicago_note(seq_idx, info))
        final_report = final_report + "\n" + "\n".join(endnote_lines)

    return final_report, items


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
        limit_per_source=30,
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
                callback=_cb,
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
    # cn_official / 已有全文的文章直接保留，其餘按時間排序取前 150 篇補抓全文；
    # 150 篇以後的文章仍保留在 items 中（僅有標題/摘要），確保不遺失。
    # -------------------------------------------------

    cn_items  = [i for i in items if i.get("source_type") == "cn_official" or i.get("content")]
    web_items = [i for i in items if i.get("source_type") != "cn_official" and not i.get("content")]

    # 按發佈時間排序（新 → 舊），最多補抓 150 篇
    web_items.sort(key=lambda x: x.get("published") or "", reverse=True)
    to_enrich = web_items[:150]
    leftover = web_items[150:]   # 超過 150 篇的部分：保留標題/摘要，不補全文
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
        items = enriched + leftover + cn_items
    except Exception as e:
        print(f"[Briefings] Article enrichment failed (using summaries): {e}")
        items = to_enrich + leftover + cn_items

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
12a. MANDATORY — Media country attribution: When citing a non-Chinese / non-English language media outlet, you MUST note which country it is from on first mention. Format: 「[媒體名稱]（[國家名稱]）」. Examples: 《朝日新聞》（日本）、《韓聯社》（韓國）、《明鏡週刊》（德國）、《費加羅報》（法國）。For articles that carry a language tag such as [日文], [韓文], [德文] etc. in their title, treat them as coming from the corresponding country. The news data also includes "來源" with a country in parentheses — use that country when provided. This rule has NO exceptions.
13. MANDATORY — People: Every person mentioned must be preceded by their full official title or role. Use the conventionally established Chinese name form (e.g., 川普、岸田文雄、習近平、賴清德). ENGLISH NAME RULES BY NATIONALITY — (A) Taiwan/ROC officials and PRC/China officials: DO NOT add a parenthetical English name. Write the Chinese name only (e.g., 行政院長卓榮泰, NOT 卓榮泰（Cho Jung-tai）; 國家主席習近平, NOT 習近平（Xi Jinping））. (B) Western figures: on first mention, follow with the common English name in parentheses — surname only. e.g. 美國總統川普（Donald Trump）、美國國務卿魯比歐（Marco Rubio）. (C) Japanese, Korean, Vietnamese: add the romanised form in square brackets after the Chinese name, then English in parentheses. e.g. 日本首相石破茂[Ishiba Shigeru]（Shigeru Ishiba）、韓國總統尹錫悅[Yoon Suk-yeol]（Yoon Suk-yeol）. (D) Other East Asian (Singapore, Thailand, Malaysia, etc.): use the person's internationally recognised English name in parentheses. CRITICAL — Before writing any parenthetical English name, be 100% certain. Known errors to avoid: 黃循財 = Lawrence Wong (Singapore PM since 2024), NOT Heng Swee Keat (who is 王瑞杰, former DPM). If uncertain of a name, OMIT the parenthetical entirely rather than guess. CRITICAL — The name inside the parentheses must contain ONLY the English name, no titles (correct: 川普（Donald Trump）; WRONG: 川普（President Donald Trump）). CRITICAL — Only assign a ministerial/official title to a person if the provided news articles explicitly confirm they currently hold that position; DO NOT rely on memory for current cabinet assignments. This rule has NO exceptions.
13a. MANDATORY — Expert names: whenever an expert or analyst is cited in 七、專家研析, render their name in bold (**Name**) and include their full title and affiliation on first mention. E.g. **美國智庫戰略與國際研究中心（CSIS）資深研究員王大維（David Wang）**。
14. MANDATORY — Organizations and institutions: On first mention, always provide both Chinese and English names. Format: Chinese name（English Name）. Examples: 北大西洋公約組織（NATO）、美國國務院（U.S. Department of State）、歐盟委員會（European Commission）、美國在台協會（American Institute in Taiwan, AIT）、中華民國國防部（Ministry of National Defense, ROC）. This rule has NO exceptions.

Output structure:
【戰略情報簡報】

一、摘要

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞
（一）中國對外情勢

（二）中國內部情勢

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
- 「五、中國要聞」必須包含兩個子節：「（一）中國對外情勢」聚焦中國外交、軍事對外、對外貿易與制裁、涉外聲明等；「（二）中國內部情勢」聚焦中國黨政內鬥、國內經濟、社會民情、人權、新疆西藏香港等內部議題。若某子節本期無相關新聞，請寫「本期無相關新聞。」，但兩個子節都不可省略。
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