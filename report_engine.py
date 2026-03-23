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

from utils.loaders import load_sources, load_experts, load_formats, load_insights
from utils.expert_loader import fetch_expert_items

try:
    from utils.cn_official import fetch_official_media_for_range
except Exception:
    def fetch_official_media_for_range(start_time=None, end_time=None, requested_subsources=None):
        return {}

import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime

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
    If resolution fails, return the original URL.
    """
    if not url:
        return url

    if "news.google.com" not in url:
        return url

    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
            allow_redirects=True,
        )
        if r.url:
            return r.url
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


def _fetch_rss_items(rss_url, source_name, limit=20):
    try:
        r = requests.get(
            rss_url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        r.raise_for_status()
        parsed = _parse_rss(r.text)
    except Exception as e:
        print(f"[Briefings] RSS fetch failed for {source_name}: {e}")
        return []

    output = []
    for item in parsed[:limit]:
        raw_url = item.get("url", "")
        resolved_url = _resolve_google_news_url(raw_url)

        output.append({
            "title": item.get("title", "").strip(),
            "url": raw_url,
            "original_url": resolved_url,
            "source": source_name,
            "published": item.get("published", ""),
            "summary": item.get("summary", ""),
            "source_type": "rss",
        })
    return output


def _build_google_news_rss_for_domain(domain):
    query = quote(f"site:{domain}")
    return f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _fetch_domain_items(domain, source_name, limit=20):
    rss_url = _build_google_news_rss_for_domain(domain)
    items = _fetch_rss_items(rss_url, source_name, limit=limit)

    for item in items:
        item["source_type"] = "domain"
        if not item.get("source"):
            item["source"] = source_name

    return items


from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_items_from_sources(selected_sources, all_sources=None, limit_per_source=20):

    normalized_sources = _normalize_selected_sources(selected_sources, all_sources=all_sources)

    all_items = []

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
            source_items = _fetch_domain_items(domain, name, limit=limit_per_source)

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

        print(f"[Briefings] Source fetched: {name} -> {len(enriched_items)}")

        return enriched_items

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [executor.submit(fetch_single, src) for src in normalized_sources]

        for future in as_completed(futures):
            try:
                items = future.result()
                all_items.extend(items)
            except Exception as e:
                print("Fetch error:", e)

    return all_items

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


def _render_citations(report_text, source_map, format_options):

    notes_style = format_options.get("notes", {}).get("style", "endnote")
    link_mode = format_options.get("links", {}).get("placement", "none")

    if notes_style == "none" and link_mode == "none":
        return _strip_ai_link_markers(report_text)

    text = _strip_ai_link_markers(report_text)

    used_codes = []

    def replace_code(match):
        code = match.group(1)
        if code not in source_map:
            return ""

        if code not in used_codes:
            used_codes.append(code)

        idx = used_codes.index(code) + 1

        if notes_style == "footnote":
            return f"{idx}"
        return f"[{idx}]"

    text = re.sub(r'\[(S\d+)\]', replace_code, text)

    # 把重複空白整理一下
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if not used_codes:
        return text

    if notes_style == "footnote":
        lines = ["", "", "Footnotes", ""]
        for idx, code in enumerate(used_codes, start=1):
            src = source_map[code]
            url = src.get("url", "")
            if url:
                lines.append(f"{idx}. {url}")
        return text + "\n" + "\n".join(lines)

    if notes_style == "endnote":
        lines = ["", "", "Sources", ""]
        for idx, code in enumerate(used_codes, start=1):
            src = source_map[code]
            title = src.get("title", "")
            source = src.get("source", "")
            url = src.get("url", "")

            if title and source and url:
                lines.append(f"[{idx}] {title} | {source} | {url}")
            elif url:
                lines.append(f"[{idx}] {url}")

        return text + "\n" + "\n".join(lines)

    return text


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


def _format_item_block(label, items):
    lines = [f"【{label}】"]
    if not items:
        lines.append("- 無明顯代表性新聞")
        return "\n".join(lines)

    for idx, item in enumerate(items, start=1):
        title = _safe_text(item.get("title"))
        source = _safe_text(item.get("source"))
        summary = _safe_text(item.get("summary"))
        url = _safe_text(item.get("original_url") or item.get("url"))
        region = _safe_text(item.get("source_region"))

        lines.append(f"{idx}. 標題: {title}")
        if source:
            lines.append(f"   來源: {source}")
        if region:
            lines.append(f"   區域: {region}")
        if summary:
            lines.append(f"   摘要: {summary}")
        if url:
            lines.append(f"   連結: {url}")

    return "\n".join(lines)


def _build_news_data_block(groups):
    blocks = []

    blocks.append(_format_item_block("國際要聞", groups["國際要聞"]))
    blocks.append(_format_item_block("台美中要聞", groups["台美中要聞"]))
    blocks.append(_format_item_block("台灣國安要聞", groups["台灣國安要聞"]))
    blocks.append(_format_item_block("中國要聞", groups["中國要聞"]))

    for region in REGION_ORDER:
        region_block = groups["區域情勢"][region]
        blocks.append(_format_item_block(f"{region}｜區域要聞", region_block["區域要聞"]))
        blocks.append(_format_item_block(f"{region}｜台灣與中國相關要聞", region_block["台灣與中國相關要聞"]))

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
    topic=None
):

    sources = load_sources()

    items = fetch_items_from_sources(
        selected_sources=selected_sources,
        all_sources=sources,
        limit_per_source=20
    )

    items = filter_items_by_topic(items, topic)

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

    groups = _group_items_for_report(items)
    news_data_block = _build_news_data_block(groups)
    language_label = _normalize_language_label(language)

    source_map = _build_citation_source_map(items, max_sources=12)

    prompt = f"""
You are a senior strategic intelligence analyst.

Write a polished strategic intelligence briefing in {language}.
The output must read like a real analytical report, not like bullet-point news notes.

Requirements:
1. Write in formal report style with coherent paragraphs.
2. Do NOT place URLs anywhere in the body text.
3. Do NOT write the string 【連結】 anywhere.
4. Use ONLY the provided news items as source material.
5. Strategic Context is analyst guidance only. Do NOT cite it as evidence.
6. Synthesize multiple news items into broader analysis instead of summarizing one article at a time.
7. When making a factual claim based on a source, append source markers like [S1], [S2].
8. You may cite multiple sources together, for example [S1][S3].
9. Only use source markers that exist in the provided News data.
10. Keep citations light and readable. Do not attach a citation to every single sentence unless necessary.

Output structure:
【戰略情報簡報】

一、摘要

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞

六、區域情勢
（一）亞太地區
1. 區域要聞
2. 台灣與中國相關要聞

（二）亞西地區
1. 區域要聞
2. 台灣與中國相關要聞

（三）北美地區
1. 區域要聞
2. 台灣與中國相關要聞

（四）拉丁美洲及加勒比海
1. 區域要聞
2. 台灣與中國相關要聞

（五）歐洲地區
1. 區域要聞
2. 台灣與中國相關要聞

（六）非洲地區
1. 區域要聞
2. 台灣與中國相關要聞

七、研析

Writing guidance:
- 「摘要」請用一小段說明本期最重要判斷。
- 「國際要聞」聚焦全球戰略層次的重要發展。
- 「台美中要聞」聚焦台灣、美國、中國三角互動及其戰略意涵。
- 「台灣國安要聞」聚焦軍事、灰帶、資安、國防、國安治理等。
- 「中國要聞」聚焦中國政治、外交、軍事、經濟、對外作為。
- 「區域情勢」請按各區域分寫，每區一定要有兩個子段。
- 「研析」請提出跨章節的整體判斷、風險、趨勢、可能後續觀察重點。

Strategic Context:
{insights_block or "None"}

News data:
{news_data_block}
"""

    client = _get_openai_client()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    report = response.output_text

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