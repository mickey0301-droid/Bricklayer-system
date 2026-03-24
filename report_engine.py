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

import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime
from bs4 import BeautifulSoup


from utils.loaders import load_sources

USER_AGENT = "Mozilla/5.0 (Briefings Local Research Tool)"

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
            headers={"User-Agent": USER_AGENT},
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

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()

        items.append({
            "title": title,
            "url": link,
            "published": pub_date,
            "summary": description,
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
            headers={"User-Agent": USER_AGENT},
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


def _fetch_rss_items(rss_url, source_name, limit=20):
    try:
        r = requests.get(
            rss_url,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        r.raise_for_status()
        parsed = _parse_rss(r.text)
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


def _build_google_news_rss_for_domain(domain, start_time=None, end_time=None, keywords=None):
    """
    建立 Google News RSS 查詢 URL。
    若傳入 start_time / end_time，會加上 after:/before: 日期篩選。
    若傳入 keywords（字串），會以 (keywords) 附加在 site: 後，
    讓 Google 直接在 RSS 層篩選相關報導，避免抓到大量無關文章。
    全球媒體（熱度排名邏輯）不傳 keywords，以免遺漏重大國際新聞。
    """
    query = f"site:{domain}"
    if keywords:
        query += f" ({keywords})"
    if start_time:
        query += f" after:{start_time.strftime('%Y/%m/%d')}"
    if end_time:
        query += f" before:{end_time.strftime('%Y/%m/%d')}"
    query_encoded = quote(query)
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

def fetch_items_from_sources(selected_sources, all_sources=None, limit_per_source=20,
                              start_time=None, end_time=None, status_callback=None):
    """
    從各來源抓取新聞條目。
    start_time / end_time 會傳入 domain 類來源（Google News RSS），
    確保查詢結果已在 Google 端預先依日期篩選，減少漏抓問題。
    status_callback(msg: str) 若提供，會在每個來源完成時被呼叫。
    """

    normalized_sources = _normalize_selected_sources(selected_sources, all_sources=all_sources)
    total = len(normalized_sources)
    all_items = []

    # 每次抓取時動態載入關鍵字設定，讓使用者在 UI 修改後立即生效
    category_keywords = load_category_keywords()

    def fetch_single(src):

        name = src.get("name", "Unknown Source")

        rss_url = (
            src.get("rss")
            or src.get("rss_url")
            or src.get("feed")
            or src.get("feed_url")
        )

        domain = (
            src.get("domain")
            or src.get("site")
        )

        if not domain and src.get("url"):
            try:
                domain = src.get("url").replace("http://", "").replace("https://", "").split("/")[0]
            except:
                domain = None

        source_items = []

        if rss_url:
            source_items = _fetch_rss_items(rss_url, name, limit=limit_per_source)

        elif domain:
            # 依來源類別選擇對應的關鍵字組，讓 Google News 在 RSS 層預先篩選。
            # 關鍵字由使用者在 Sources 頁面設定，儲存後立即生效（category_keywords 已在此次呼叫開始時載入）。
            # 若來源無對應類別，則不加關鍵字（抓全部，由後段邏輯過濾）。
            cats = src.get("category", []) or []
            if isinstance(cats, str):
                cats = [cats]
            domain_keywords = None
            for cat in cats:
                if cat in category_keywords:
                    domain_keywords = category_keywords[cat]
                    break
            source_items = _fetch_domain_items(
                domain, name, limit=limit_per_source,
                start_time=start_time, end_time=end_time,
                keywords=domain_keywords,
            )

        else:
            print(f"[Briefings] Source skipped: {name}")

        enriched_items = []
        for item in source_items:
            enriched = dict(item)
            enriched["source_region"] = src.get("region", "")
            enriched["source_category"] = src.get("category", []) or []
            enriched["source_description"] = src.get("description", "")
            enriched["source_type"] = src.get("type", enriched.get("source_type", "rss"))
            enriched_items.append(enriched)

        return name, enriched_items

    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {executor.submit(fetch_single, src): src for src in normalized_sources}

        for future in as_completed(futures):
            try:
                name, items = future.result()
                all_items.extend(items)
                completed += 1
                if status_callback and items:
                    status_callback(
                        f"rss",
                        f"{name}：{len(items)} 篇",
                        completed, total, len(all_items),
                    )
                elif status_callback:
                    status_callback("rss_progress", None, completed, total, len(all_items))
            except Exception as e:
                completed += 1
                print("Fetch error:", e)

    return all_items

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

        key = (url or title).lower().strip()
        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        ordered.append({
            "title": title,
            "source": source,
            "url": url
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
    Chicago Notes & Bibliography style:
    ¹ "Title," Source Name, Date. URL.
    """
    title = src.get("title", "").strip()
    source = src.get("source", "").strip()
    published_at = src.get("published_at", "").strip()
    url = src.get("url", "").strip()

    sup = _to_superscript(idx)
    parts = []
    if title:
        parts.append(f'"{title},"')
    if source:
        parts.append(source + ("," if published_at else "."))
    if published_at:
        parts.append(published_at + ".")
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

    groups["國際要聞"] = _dedupe_items(groups["國際要聞"], limit=12)
    groups["台美中要聞"] = _dedupe_items(groups["台美中要聞"], limit=8)
    groups["台灣國安要聞"] = _dedupe_items(groups["台灣國安要聞"], limit=8)
    groups["中國要聞"] = _dedupe_items(groups["中國要聞"], limit=8)

    for region in REGION_ORDER:
        groups["區域情勢"][region]["區域要聞"] = _dedupe_items(
            groups["區域情勢"][region]["區域要聞"], limit=6
        )
        groups["區域情勢"][region]["台灣與中國相關要聞"] = _dedupe_items(
            groups["區域情勢"][region]["台灣與中國相關要聞"], limit=6
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
    # 兩段式過濾 + 全文補抓
    # 路徑 A（台灣／中國議題）：關鍵字過濾 → 補抓全文（最多 40 篇）
    # 路徑 B（國際要聞 / 全球媒體）：熱度排名 → 取 top 20 → 補抓全文
    # -------------------------------------------------

    tw_cn_items = []
    global_items = []
    other_items  = []

    for it in items:
        cats = set(it.get("source_category") or [])
        if it.get("source_type") == "cn_official" or it.get("content"):
            other_items.append(it)
        elif cats & _GLOBAL_MEDIA_CATEGORIES:
            global_items.append(it)
        else:
            tw_cn_items.append(it)

    tw_cn_matched = [i for i in tw_cn_items if _matches_tw_cn(i)]
    _cb("stage", f"🔍 台灣/中國關鍵字命中：{len(tw_cn_matched)} / {len(tw_cn_items)} 篇")

    global_top = _rank_by_coverage(global_items, top_n=20)
    _cb("stage", f"🌍 國際熱點排名：取前 {len(global_top)} / {len(global_items)} 篇最多來源報導")

    to_enrich = tw_cn_matched[:40] + global_top[:20]
    _cb("stage", f"📄 補抓 {len(to_enrich)} 篇全文（10 個並行連線）…")

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
        items = enriched + other_items
    except Exception as e:
        print(f"[Briefings] Article enrichment failed (using summaries): {e}")
        items = to_enrich + other_items

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

    expert_section = ""
    if has_expert_data:
        expert_section = """
七、專家研析
1. 國際情勢解讀
2. 台美中情勢解讀

八、研析"""
    else:
        expert_section = "\n七、研析"

    expert_guidance = ""
    if has_expert_data:
        expert_guidance = f"""- 「專家研析」必須引用 Expert Analysis Data 中具名專家（{', '.join(expert_names)}）的實際觀點，並標注是哪位專家的看法。若某專家無明確觀點可引用，請省略該專家。勿憑空虛構專家言論。"""
    else:
        expert_guidance = "- 本期無專家資料，請省略「專家研析」章節。"

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
12. Whenever you mention a media outlet, always name it specifically — never use vague terms like "歐洲媒體", "西方媒體", "美國媒體". Write the actual outlet name with both Chinese and English on first mention, e.g. 德國之聲（Deutsche Welle）、法新社（Agence France-Presse, AFP）、路透社（Reuters）. For widely recognized outlets where one name is dominant, you may use just that name, e.g. CNN、BBC、紐約時報（New York Times）.
13. Whenever you mention a person, always include their specific title or role immediately before their name. For the Chinese name, use only the conventionally established form: for Western figures use surname only (e.g., 川普、拜登、奧斯汀、馬克宏、梅洛尼); for Japanese, Korean, or other East Asian figures use the full name in Chinese characters (e.g., 岸田文雄、尹錫悅、習近平、普廷). Always follow with the person's full English name in parentheses on first mention. Examples: 美國總統川普（Donald Trump）、日本首相岸田文雄（Fumio Kishida）、韓國總統尹錫悅（Yoon Suk-yeol）、國防部長奧斯汀（Lloyd Austin）.
14. Whenever you mention a non-media organization or institution for the first time, always include both its Chinese name and English name in parentheses, e.g. 北大西洋公約組織（NATO）、美國國務院（U.S. Department of State）、歐盟委員會（European Commission）.

Output structure:
【戰略情報簡報】

一、摘要

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞

六、區域情勢
（一）亞太地區
1. 區域要聞（僅在有亞太地區相關新聞時撰寫，否則省略）
2. 台灣與中國相關要聞（僅在有亞太地區涉台涉中新聞時撰寫，否則省略）

（二）亞西地區
1. 區域要聞（僅在有亞西地區相關新聞時撰寫，否則省略）
2. 台灣與中國相關要聞（僅在有亞西地區涉台涉中新聞時撰寫，否則省略）

（三）北美地區
1. 區域要聞（僅在有北美地區相關新聞時撰寫，否則省略）
2. 台灣與中國相關要聞（僅在有北美地區涉台涉中新聞時撰寫，否則省略）

（四）拉丁美洲及加勒比海
1. 區域要聞（僅在有拉丁美洲相關新聞時撰寫，否則省略）
2. 台灣與中國相關要聞（僅在有拉丁美洲涉台涉中新聞時撰寫，否則省略）

（五）歐洲地區
1. 區域要聞（僅在有歐洲地區相關新聞時撰寫，否則省略）
2. 台灣與中國相關要聞（僅在有歐洲地區涉台涉中新聞時撰寫，否則省略）

（六）非洲地區
1. 區域要聞（僅在有非洲地區相關新聞時撰寫，否則省略）
2. 台灣與中國相關要聞（僅在有非洲地區涉台涉中新聞時撰寫，否則省略）
{expert_section}

Writing guidance:
- 「摘要」請用一小段說明本期最重要判斷。
- 「國際要聞」聚焦全球戰略層次的重要發展。
- 「台美中要聞」聚焦台灣、美國、中國三角互動及其戰略意涵。
- 「台灣國安要聞」聚焦軍事、灰帶、資安、國防、國安治理等。
- 「中國要聞」聚焦中國政治、外交、軍事、經濟、對外作為。
- 「區域情勢」各區域的子段只在有明確對應新聞時才撰寫，若無相關新聞請直接省略，不要寫「無相關新聞」。
{expert_guidance}
- 「研析」請提出跨章節的整體判斷、風險、趨勢、可能後續觀察重點。

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