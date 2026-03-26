import json
import os
from pathlib import Path
from copy import deepcopy

try:
    from opencc import OpenCC
    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"

SOURCES_PATH = CONFIG_DIR / "sources.json"
SOURCES_USER_PATH = CONFIG_DIR / "sources_user.json"
EXPERTS_PATH = CONFIG_DIR / "experts.json"
EXPERTS_USER_PATH = CONFIG_DIR / "experts_user.json"   # user-local, not git-tracked → survives updates
PROFILES_PATH = CONFIG_DIR / "profiles.json"
INSIGHTS_PATH = CONFIG_DIR / "insights.txt"
AUTO_EXPORT_PATH = CONFIG_DIR / "auto_export.json"
AUTO_EXPORT_STATE_PATH = CONFIG_DIR / "auto_export_state.json"
FORMATS_PATH = CONFIG_DIR / "formats.json"
GLOBAL_MEDIA_PATH = CONFIG_DIR / "global_media.json"

# 固定抓法（特殊 scraper，不可編輯）
FIXED_CN_OFFICIAL_SOURCES = [
    {
        "name": "人民日報",
        "subsource": "people_daily",
        "type": "cn_official",
        "url": "",
        "category": ["中共官媒"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
    {
        "name": "新聞聯播",
        "subsource": "xinwen_lianbo",
        "type": "cn_official",
        "url": "",
        "category": ["中共官媒"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
    {
        "name": "解放軍報",
        "subsource": "pla_daily",
        "type": "cn_official",
        "url": "",
        "category": ["中共官媒", "軍事"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
]

# 可編輯的中共官媒預設值（type=rss，由網域生成 Google News zh-CN RSS URL）
# 若使用者的 sources 中已有同名來源，則不重複注入。
CN_OFFICIAL_EDITABLE_DEFAULTS = [
    {
        "name": "新華社",
        "domain": "xinhuanet.com",
        "type": "rss",
        "url": "https://news.google.com/rss/search?q=site%3Axinhuanet.com&hl=zh-CN&gl=SG&ceid=SG:zh-Hans",
        "category": ["中共官媒"],
        "region": "CN",
        "enabled": True,
        "description": "Google News site:xinhuanet.com zh-CN",
    },
    {
        "name": "中國外交部記者會",
        "domain": "fmprc.gov.cn",
        "type": "rss",
        "url": "https://news.google.com/rss/search?q=site%3Afmprc.gov.cn&hl=zh-CN&gl=SG&ceid=SG:zh-Hans",
        "category": ["中共官媒", "外交"],
        "region": "CN",
        "enabled": True,
        "description": "Google News site:fmprc.gov.cn zh-CN",
    },
    {
        "name": "中國國防部記者會",
        "domain": "mod.gov.cn",
        "type": "rss",
        "url": "https://news.google.com/rss/search?q=site%3Amod.gov.cn&hl=zh-CN&gl=SG&ceid=SG:zh-Hans",
        "category": ["中共官媒", "國防"],
        "region": "CN",
        "enabled": True,
        "description": "Google News site:mod.gov.cn zh-CN",
    },
    {
        "name": "國台辦",
        "domain": "gwytb.gov.cn",
        "type": "rss",
        "url": "https://news.google.com/rss/search?q=site%3Agwytb.gov.cn&hl=zh-CN&gl=SG&ceid=SG:zh-Hans",
        "category": ["中共官媒", "涉台"],
        "region": "CN",
        "enabled": True,
        "description": "Google News site:gwytb.gov.cn zh-CN",
    },
]


def ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _default_json(path: Path):
    if path.name == "sources.json":
        return []
    if path.name == "experts.json":
        return []
    if path.name == "profiles.json":
        return []
    if path.name == "formats.json":
        return [
            {
                "name": "default",
                "fonts": {
                    "zh": "Arial",
                    "en": "Arial"
                },
                "title": {
                    "font_size": 16,
                    "bold": True,
                    "align": "center"
                },
                "section_heading": {
                    "font_size": 14,
                    "bold": True
                },
                "body": {
                    "font_size": 12,
                    "line_spacing": 1.15
                },
                "notes": {
                    "style": "footnote"
                },
                "links": {
                    "placement": "inline"
                }
            }
        ]
    if path.name == "auto_export.json":
        return {
            "enabled": False,
            "mode": "daily_times",
            "daily_times": ["09:00"],
            "coverage_hours": 24,
            "interval_hours": 2,
            "window_start": "08:00",
            "window_end": "22:00",
            "source_categories": [],
            "source_names": [],
            "expert_categories": [],
            "expert_names": [],
            "language": "繁體中文",
            "profile": "",
            "extra_insights": "",
            "output_formats": ["docx"],
            "output_targets": ["local"],
            "google_drive_folder_id": "",
        }
    if path.name == "auto_export_state.json":
        return {"last_runs": {}}
    return []


def read_json(path: Path):
    ensure_config_dir()
    if not path.exists():
        default = _default_json(path)
        write_json(path, default)
        return deepcopy(default)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        default = _default_json(path)
        write_json(path, default)
        return deepcopy(default)


def write_json(path: Path, data):
    ensure_config_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(path: Path, default=""):
    ensure_config_dir()
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write(default)
        return default
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, content: str):
    ensure_config_dir()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def normalize_listish(value):
    if value is None:
        return []

    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for sep in ["\n", "；", ";", "、"]:
            text = text.replace(sep, ",")
        parts = [p.strip() for p in text.split(",")]
        return [p for p in parts if p]

    return [str(value).strip()] if str(value).strip() else []


def normalize_category(value):
    return normalize_listish(value)


def normalize_aliases(value):
    return normalize_listish(value)


def list_to_csv(value):
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join([str(x).strip() for x in value if str(x).strip()])
    return str(value)


def tw_to_simplified(text: str) -> str:
    if not text:
        return ""
    if _OPENCC is None:
        return text
    try:
        return _OPENCC.convert(text)
    except Exception:
        return text


def unique_keep_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_expert_search_names(expert: dict):
    name_zh = (expert.get("name_zh") or "").strip()
    name_sc = (expert.get("name_sc") or "").strip()   # 簡體字名（中國大陸專家）
    name_en = (expert.get("name_en") or "").strip()
    aliases = normalize_aliases(expert.get("aliases"))

    # Use explicit name_sc if provided; otherwise auto-convert from name_zh
    zh_simplified = name_sc or (tw_to_simplified(name_zh) if name_zh else "")
    alias_expanded = []
    for a in aliases:
        alias_expanded.append(a)
        if any("\u4e00" <= ch <= "\u9fff" for ch in a):
            alias_expanded.append(tw_to_simplified(a))

    search_names = unique_keep_order(
        [name_zh, zh_simplified, name_en] + aliases + alias_expanded
    )
    return search_names


def display_expert_name(expert: dict) -> str:
    name_zh = (expert.get("name_zh") or "").strip()
    name_en = (expert.get("name_en") or "").strip()
    return name_zh or name_en or "Unnamed Expert"


# ── Google News locale map ───────────────────────────────────────────────────
# Maps language code → (hl, gl, ceid) parameters for Google News RSS URLs.
_LOCALE_MAP: dict[str, tuple[str, str, str]] = {
    "zh-TW": ("zh-TW", "TW", "TW:zh-Hant"),
    "zh-CN": ("zh-CN", "SG", "SG:zh-Hans"),
    "en-US": ("en-US", "US", "US:en"),
    "en-GB": ("en-GB", "GB", "GB:en"),
    "ja":    ("ja",    "JP", "JP:ja"),
    "ko":    ("ko",    "KR", "KR:ko"),
    "fr":    ("fr",    "FR", "FR:fr"),
    "de":    ("de",    "DE", "DE:de"),
    "es":    ("es",    "ES", "ES:es"),
}
LOCALE_OPTIONS: list[str] = list(_LOCALE_MAP.keys())


def _gnews_url(query: str, lang: str = "zh-TW") -> str:
    import urllib.parse as _up
    hl, gl, ceid = _LOCALE_MAP.get(lang, _LOCALE_MAP["zh-TW"])
    return (
        "https://news.google.com/rss/search?q="
        + _up.quote(query)
        + f"&hl={hl}&gl={gl}&ceid={ceid}"
    )


def gnews_url_from_domain(domain: str, lang: str = "zh-TW") -> str:
    """Generate a Google News RSS URL for a domain with the given locale."""
    domain = domain.strip().lstrip("https://").lstrip("http://").split("/")[0]
    return _gnews_url(f"site:{domain}", lang)


def cn_gnews_url_from_domain(domain: str) -> str:
    """Backward-compat wrapper: generate zh-CN Google News URL from domain."""
    return gnews_url_from_domain(domain, "zh-CN")


def normalize_source(item: dict) -> dict:
    item = item or {}
    source_type = (item.get("type") or "rss").strip().lower()
    if source_type not in ["rss", "domain", "cn_official"]:
        source_type = "rss"

    raw_domain = (item.get("domain") or "").strip()
    lang       = (item.get("language") or "").strip()

    normalized = {
        "name":      (item.get("name")      or "").strip(),
        "name_sc":   (item.get("name_sc")   or "").strip(),  # 簡體字名（中共官媒用）
        "domain":    raw_domain,
        "language":  lang,
        "subsource": (item.get("subsource") or "").strip(),
        "type":      source_type,
        "url":       (item.get("url")       or "").strip(),
        "category":  normalize_category(item.get("category")),
        "region":    (item.get("region")    or "").strip(),
        "enabled":   bool(item.get("enabled", True)),
        "description": (item.get("description") or "").strip(),
        "readonly":  bool(item.get("readonly", False)),
        "fixed":     bool(item.get("fixed", False)),
    }
    # Auto-generate RSS URL from domain when url is empty.
    # Uses the source's language setting; falls back to zh-TW.
    if normalized["domain"] and not normalized["url"]:
        normalized["url"] = gnews_url_from_domain(
            normalized["domain"], normalized["language"] or "zh-TW"
        )
    return normalized


def normalize_expert(item: dict) -> dict:
    item = item or {}
    normalized = {
        "name_zh":     (item.get("name_zh")     or "").strip(),
        "name_sc":     (item.get("name_sc")     or "").strip(),  # 簡體字名（中國大陸專家）
        "name_en":     (item.get("name_en")     or "").strip(),
        "aliases":     normalize_aliases(item.get("aliases")),
        "category":    normalize_category(item.get("category")),
        "affiliation": (item.get("affiliation") or "").strip(),
        "region":      (item.get("region")      or "").strip(),
        "language":    (item.get("language")    or "").strip(),  # 覆蓋自動語言偵測
        "enabled":     bool(item.get("enabled", True)),
        "description": (item.get("description") or "").strip(),
        "rss_url":     (item.get("rss_url")     or "").strip(),
    }
    normalized["name"] = display_expert_name(normalized)
    normalized["search_names"] = build_expert_search_names(normalized)
    return normalized


def normalize_global_media_source(item: dict) -> dict:
    item = item or {}

    name = str(item.get("name", "")).strip()
    domain = str(item.get("domain", "")).strip()
    continent = str(item.get("continent", "")).strip()
    country = str(item.get("country", "")).strip()
    language = str(item.get("language", "")).strip()
    media_type = str(item.get("type", "")).strip()
    rss = str(item.get("rss", "")).strip()

    if not name or not domain:
        return {}

    # 有 RSS 就直接用 RSS feed，否則退回 domain 查詢
    if rss:
        src_type = "rss"
        url = rss
    else:
        src_type = "domain"
        url = domain

    return {
        "name": name,
        "subsource": name,
        "type": src_type,
        "url": url,
        "category": ["全球媒體", continent] if continent else ["全球媒體"],
        "region": country or continent,
        "enabled": True,
        "description": f"global media | language={language} | type={media_type}",
        "language": language,
        "country": country,
        "readonly": True,
        "fixed": False,
    }


def load_global_media_sources():
    data = read_json(GLOBAL_MEDIA_PATH)
    if not isinstance(data, list):
        return []

    results = []
    seen = set()

    for item in data:
        normalized = normalize_global_media_source(item)
        if not normalized:
            continue

        key = normalized.get("name", "").strip()
        if not key or key in seen:
            continue

        seen.add(key)
        results.append(normalized)

    return results


def _restore_from_github(local_path: Path, repo_path: str) -> None:
    """
    If *local_path* does not exist, attempt to pull it from the GitHub repo.
    Called at the top of every load_* function so data survives Streamlit
    Cloud restarts (ephemeral filesystem).  Silently does nothing if
    credentials are absent or the file isn't in the repo yet.
    """
    if local_path.exists():
        return
    try:
        from utils.github_storage import fetch_file
        fetch_file(local_path, repo_path)
    except Exception:
        pass


def load_sources(editable_only=False):
    # 若本機檔案不存在（Streamlit Cloud 重啟後），先從 GitHub 拉回
    _restore_from_github(SOURCES_USER_PATH, "config/sources_user.json")
    # 優先讀取使用者本地版本（不被 git pull 覆蓋）
    active_path = SOURCES_USER_PATH if SOURCES_USER_PATH.exists() else SOURCES_PATH
    data = read_json(active_path)
    if not isinstance(data, list):
        data = []

    editable = []
    for item in data:
        n = normalize_source(item)
        if n["type"] == "cn_official":
            n["readonly"] = True
            n["fixed"] = True
        editable.append(n)

    editable = [x for x in editable if not x.get("fixed")]

    # 注入可編輯中共官媒預設值（若使用者尚未自訂同名來源）
    existing_names = {s["name"] for s in editable}
    for item in CN_OFFICIAL_EDITABLE_DEFAULTS:
        if item["name"] not in existing_names:
            editable.append(normalize_source(item))

    global_sources = load_global_media_sources()

    # 把專家清單注入為 '自訂專家' 類別的來源條目（readonly，不會被寫回）
    expert_srcs = experts_as_sources()

    if editable_only:
        return editable + global_sources + expert_srcs

    all_sources = editable + global_sources + deepcopy(FIXED_CN_OFFICIAL_SOURCES) + expert_srcs

    return all_sources


def save_sources(sources) -> bool:
    """Save sources locally and commit to GitHub.  Returns True if GitHub
    sync succeeded, False if local-only (credentials missing or API error)."""
    normalized = []
    for item in (sources or []):
        n = normalize_source(item)
        if n.get("fixed") or n.get("readonly") or n["type"] == "cn_official":
            continue
        normalized.append(n)
    write_json(SOURCES_USER_PATH, normalized)
    try:
        from utils.github_storage import commit_file
        return commit_file(SOURCES_USER_PATH, "config/sources_user.json",
                           "chore: update sources_user.json via app")
    except Exception:
        return False


def load_experts():
    # 若本機檔案不存在（Streamlit Cloud 重啟後），先從 GitHub 拉回
    _restore_from_github(EXPERTS_USER_PATH, "config/experts_user.json")
    # 優先讀取使用者本地版本（不被 git pull 覆蓋），與 load_sources 邏輯一致
    active_path = EXPERTS_USER_PATH if EXPERTS_USER_PATH.exists() else EXPERTS_PATH
    data = read_json(active_path)
    if not isinstance(data, list):
        data = []
    return [normalize_expert(item) for item in data]


def expert_gnews_urls(expert: dict) -> list:
    """
    Return a list of (label, url) tuples for Google News RSS queries
    based on the expert's name fields.

    If `language` is set on the expert, all name-based queries use that
    single locale.  Otherwise the automatic rules apply:
      - name_en → en-US
      - name_zh → zh-TW, plus zh-CN when region indicates mainland China
    If rss_url is set, returns [] (direct RSS takes priority).
    """
    if (expert.get("rss_url") or "").strip():
        return []

    results = []
    name_zh  = (expert.get("name_zh")  or "").strip()
    name_en  = (expert.get("name_en")  or "").strip()
    name_sc  = (expert.get("name_sc")  or "").strip()
    region   = (expert.get("region")   or "").strip().upper()
    language = (expert.get("language") or "").strip()

    def _url(name: str, lang: str) -> str:
        import urllib.parse as _up
        return _gnews_url(f'"{name}"', lang)

    if language:
        # User chose an explicit locale — apply it to every available name
        if name_en:
            results.append((f"英文名（{language}）", _url(name_en, language)))
        if name_zh:
            results.append((f"中文名（{language}）", _url(name_zh, language)))
        elif name_sc:
            results.append((f"簡體名（{language}）", _url(name_sc, language)))
    else:
        # Auto-detect locale from name type and region
        if name_en:
            results.append(("英文名（en-US）", _url(name_en, "en-US")))

        if name_zh:
            results.append(("中文名（zh-TW）", _url(name_zh, "zh-TW")))
            _cn_regions = {"CN", "中國", "中国", "大陸", "大陆", "CHINA", "MAINLAND"}
            is_cn = any(r in region for r in _cn_regions)
            if is_cn:
                sc_name = name_sc or tw_to_simplified(name_zh)
                if sc_name:
                    results.append(("簡體名（zh-CN）", _url(sc_name, "zh-CN")))
        elif name_sc:
            results.append(("簡體名（zh-CN）", _url(name_sc, "zh-CN")))

    return results


def experts_as_sources() -> list:
    """
    Convert enabled experts into synthetic source-like dicts so they appear
    in the source selection UI under the '自訂專家' category.

    - If the expert has rss_url → type='rss', url=rss_url (direct fetch)
    - Otherwise                 → type='domain', url='' (Google News name search)
    These entries are marked readonly=True so they are never written to sources.json.
    """
    experts = load_experts()
    sources = []
    for e in experts:
        if not e.get("enabled", True):
            continue
        name = (e.get("name") or e.get("name_zh") or e.get("name_en") or "").strip()
        if not name:
            continue
        rss_url = (e.get("rss_url") or "").strip()
        # 合併「自訂專家」與專家自身的 category，讓來源分類選單可以用專家分類篩選
        expert_cats = normalize_category(e.get("category"))
        merged_cats = ["自訂專家"] + [c for c in expert_cats if c and c != "自訂專家"]
        sources.append({
            "name": name,
            "type": "rss" if rss_url else "domain",
            "url": rss_url,
            "category": merged_cats,
            "region": (e.get("region") or "").strip(),
            "enabled": True,
            "description": (e.get("description") or "").strip(),
            "readonly": True,       # prevent save_sources from writing these back
            "from_expert": True,    # marker so UI can identify them
            # keep expert name fields so fetch_items_from_sources can build URLs
            "name_zh": (e.get("name_zh") or "").strip(),
            "name_en": (e.get("name_en") or "").strip(),
        })
    return sources


def save_experts(experts) -> bool:
    """Save experts locally and commit to GitHub.  Returns True if GitHub
    sync succeeded, False if local-only (credentials missing or API error)."""
    normalized = [normalize_expert(item) for item in (experts or [])]
    write_json(EXPERTS_USER_PATH, normalized)
    try:
        from utils.github_storage import commit_file
        return commit_file(EXPERTS_USER_PATH, "config/experts_user.json",
                           "chore: update experts_user.json via app")
    except Exception:
        return False


_INSIGHTS_USER_PATH = "config/insights_user.json"
_INSIGHTS_DEFAULT_PATH = "config/insights.json"


def load_insights(path=None):
    # 優先讀取使用者本地版本（不被 git pull 覆蓋）
    if path is None:
        path = _INSIGHTS_USER_PATH if os.path.exists(_INSIGHTS_USER_PATH) else _INSIGHTS_DEFAULT_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        cleaned = []

        for item in data:
            if not isinstance(item, dict):
                continue

            cleaned.append({
                "title": str(item.get("title", "")).strip(),
                "content": str(item.get("content", "")).strip(),
                "tags": item.get("tags", []),
            })

        return cleaned

    except Exception:
        return []


def save_insights(insights, path=None):
    # 始終寫到 _user 版本，不動 git 追蹤的預設檔
    if path is None:
        path = _INSIGHTS_USER_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cleaned = []

    for item in insights:

        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()

        tags = item.get("tags", [])

        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        elif isinstance(tags, list):
            tags = [str(t).strip() for t in tags if str(t).strip()]

        else:
            tags = []

        cleaned.append({
            "title": title,
            "content": content,
            "tags": tags
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)


def load_profiles():
    data = read_json(PROFILES_PATH)
    if isinstance(data, list):
        return data
    return []


def save_profiles(profiles):
    write_json(PROFILES_PATH, profiles or [])


def load_formats():
    data = read_json(FORMATS_PATH)
    if isinstance(data, list):
        return data
    return []


def save_formats(formats):
    write_json(FORMATS_PATH, formats or [])


AUTO_EXPORT_PATH = "config/auto_export.json"
AUTO_EXPORT_USER_PATH = "config/auto_export_user.json"


def load_auto_export():

    import json
    import os

    # 優先讀取使用者本地版本（不被 git pull 覆蓋）
    path = AUTO_EXPORT_USER_PATH if os.path.exists(AUTO_EXPORT_USER_PATH) else AUTO_EXPORT_PATH

    if not os.path.exists(path):

        return {
            "enabled": True,
            "schedules": [],
            "drive_folders": [],
        }

    try:

        with open(path, "r", encoding="utf-8") as f:

            data = json.load(f)

        if "schedules" not in data:
            data["schedules"] = []

        if "enabled" not in data:
            data["enabled"] = True

        if "drive_folders" not in data:
            data["drive_folders"] = []

        return data

    except:

        return {
            "enabled": True,
            "schedules": [],
            "drive_folders": [],
        }


def save_auto_export(config):

    import json
    import os

    os.makedirs("config", exist_ok=True)

    # 始終寫到 _user 版本，不動 git 追蹤的預設檔
    with open(AUTO_EXPORT_USER_PATH, "w", encoding="utf-8") as f:

        json.dump(config, f, ensure_ascii=False, indent=2)


def load_auto_export_state():
    data = read_json(AUTO_EXPORT_STATE_PATH)
    if not isinstance(data, dict):
        data = {"last_runs": {}}
    data.setdefault("last_runs", {})
    data.setdefault("running_now", [])
    data.setdefault("running_started_at", {})
    data.setdefault("run_history", [])
    return data


def save_auto_export_state(data: dict):
    write_json(AUTO_EXPORT_STATE_PATH, data or {"last_runs": {}})


# ── 各類別 Google News RSS 關鍵字 ────────────────────────────────────────────

_CATEGORY_KEYWORDS_USER_PATH = CONFIG_DIR / "category_keywords_user.json"

# 預設關鍵字（code-level fallback，不會被 git pull 覆蓋使用者設定）
DEFAULT_CATEGORY_KEYWORDS: dict = {
    "自訂台灣媒體": (
        "中國 OR 中共 OR 解放軍 OR 共軍 OR 習近平 OR 兩岸 OR 台海 OR 北京 OR "
        "台積電 OR 半導體 OR 印太 OR 南海 OR 東海 OR 美中 OR 中美"
    ),
    "自訂國際媒體": (
        "taiwan OR 台灣 OR china OR 中國 OR \"cross-strait\" OR 兩岸 OR "
        "\"xi jinping\" OR 習近平 OR tsmc OR 台積電 OR \"chinese military\" OR 解放軍 OR "
        "\"indo-pacific\" OR 印太 OR \"south china sea\" OR \"taiwan strait\" OR ccp OR 中共"
    ),
    "自訂專家": (
        "taiwan OR 台灣 OR china OR 中國 OR \"cross-strait\" OR 兩岸 OR "
        "\"xi jinping\" OR 習近平 OR tsmc OR 台積電 OR \"chinese military\" OR 解放軍 OR "
        "\"indo-pacific\" OR 印太 OR \"south china sea\" OR \"taiwan strait\" OR ccp OR 中共"
    ),
    "全球媒體": (
        "taiwan OR 台灣 OR china OR 中國 OR \"cross-strait\" OR 兩岸 OR "
        "\"xi jinping\" OR 習近平 OR tsmc OR 台積電 OR \"chinese military\" OR 解放軍 OR "
        "\"indo-pacific\" OR 印太 OR \"south china sea\" OR \"taiwan strait\" OR ccp OR 中共"
    ),
    "中國媒體": (
        "台湾 OR 台独 OR 台岛 OR 两岸 OR 台海 OR 涉台 OR \"一中原则\" OR \"一中政策\" OR "
        "\"一个中国原则\" OR \"一个中国政策\" OR 国民党 OR 民进党 OR 台企 OR 台胞 OR "
        "宝岛 OR 陈水扁 OR 马英九 OR 蔡英文 OR 赖清德"
    ),
}


def load_category_keywords() -> dict:
    """
    載入各類別 Google News RSS 關鍵字設定。
    優先讀取使用者自訂檔（category_keywords_user.json），
    不存在時回傳 DEFAULT_CATEGORY_KEYWORDS 的副本。
    """
    # 若本機檔案不存在（Streamlit Cloud 重啟後），先從 GitHub 拉回
    _restore_from_github(
        _CATEGORY_KEYWORDS_USER_PATH, "config/category_keywords_user.json"
    )
    if _CATEGORY_KEYWORDS_USER_PATH.exists():
        try:
            with open(_CATEGORY_KEYWORDS_USER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                # 以預設值補齊缺少的類別
                merged = dict(DEFAULT_CATEGORY_KEYWORDS)
                merged.update(data)
                return merged
        except Exception:
            pass
    return dict(DEFAULT_CATEGORY_KEYWORDS)


def save_category_keywords(keywords: dict) -> bool:
    """Save category keywords locally and commit to GitHub.  Returns True if
    GitHub sync succeeded, False if local-only (credentials missing or error)."""
    ensure_config_dir()
    with open(_CATEGORY_KEYWORDS_USER_PATH, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)
    try:
        from utils.github_storage import commit_file
        return commit_file(Path(_CATEGORY_KEYWORDS_USER_PATH),
                           "config/category_keywords_user.json",
                           "chore: update category_keywords_user.json via app")
    except Exception:
        return False


def get_source_categories(sources=None):
    sources = sources if sources is not None else load_sources()
    cats = []
    for s in sources:
        cats.extend(normalize_category(s.get("category")))
    return unique_keep_order(cats)


def get_expert_categories(experts=None):
    experts = experts if experts is not None else load_experts()
    cats = []
    for e in experts:
        cats.extend(normalize_category(e.get("category")))
    return unique_keep_order(cats)


def source_to_editor_row(source: dict):
    return {
        "name":        source.get("name", ""),
        "type":        source.get("type", "rss"),
        "domain":      source.get("domain", ""),
        "language":    source.get("language", ""),
        "url":         source.get("url", ""),
        "category":    list_to_csv(source.get("category")),
        "region":      source.get("region", ""),
        "enabled":     bool(source.get("enabled", True)),
        "description": source.get("description", ""),
    }


def editor_row_to_source(row: dict):
    return normalize_source(
        {
            "name":        row.get("name", ""),
            "type":        row.get("type", "rss"),
            "domain":      row.get("domain", ""),
            "language":    row.get("language", ""),
            "url":         row.get("url", ""),
            "category":    row.get("category", ""),
            "region":      row.get("region", ""),
            "enabled":     row.get("enabled", True),
            "description": row.get("description", ""),
        }
    )


def expert_to_editor_row(expert: dict):
    return {
        "name_zh":     expert.get("name_zh", ""),
        "name_sc":     expert.get("name_sc", ""),
        "name_en":     expert.get("name_en", ""),
        "aliases":     list_to_csv(expert.get("aliases")),
        "category":    list_to_csv(expert.get("category")),
        "affiliation": expert.get("affiliation", ""),
        "region":      expert.get("region", ""),
        "language":    expert.get("language", ""),
        "enabled":     bool(expert.get("enabled", True)),
        "description": expert.get("description", ""),
        "rss_url":     expert.get("rss_url", ""),
    }


def editor_row_to_expert(row: dict):
    return normalize_expert(
        {
            "name_zh":     row.get("name_zh", ""),
            "name_sc":     row.get("name_sc", ""),
            "name_en":     row.get("name_en", ""),
            "aliases":     row.get("aliases", ""),
            "category":    row.get("category", ""),
            "affiliation": row.get("affiliation", ""),
            "region":      row.get("region", ""),
            "language":    row.get("language", ""),
            "enabled":     row.get("enabled", True),
            "description": row.get("description", ""),
            "rss_url":     row.get("rss_url", ""),
        }
    )