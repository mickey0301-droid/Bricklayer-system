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
EXPERTS_PATH = CONFIG_DIR / "experts.json"
PROFILES_PATH = CONFIG_DIR / "profiles.json"
INSIGHTS_PATH = CONFIG_DIR / "insights.txt"
AUTO_EXPORT_PATH = CONFIG_DIR / "auto_export.json"
AUTO_EXPORT_STATE_PATH = CONFIG_DIR / "auto_export_state.json"
FORMATS_PATH = CONFIG_DIR / "formats.json"
GLOBAL_MEDIA_PATH = CONFIG_DIR / "global_media.json"

FIXED_CN_OFFICIAL_SOURCES = [
    {
        "name": "人民日報",
        "subsource": "people_daily",
        "type": "cn_official",
        "url": "",
        "category": ["中國官媒"],
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
        "category": ["中國官媒"],
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
        "category": ["中國官媒", "軍事"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
    {
        "name": "新華社",
        "subsource": "xinhua",
        "type": "cn_official",
        "url": "",
        "category": ["中國官媒"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
    {
        "name": "中國外交部記者會",
        "subsource": "mfa_press",
        "type": "cn_official",
        "url": "",
        "category": ["中國官方", "外交"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
    {
        "name": "中國國防部記者會",
        "subsource": "mod_press",
        "type": "cn_official",
        "url": "",
        "category": ["中國官方", "國防"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
    },
    {
        "name": "國台辦",
        "subsource": "taiwan_affairs_office",
        "type": "cn_official",
        "url": "",
        "category": ["中國官方", "涉台"],
        "region": "CN",
        "enabled": True,
        "description": "固定抓法（唯讀）",
        "readonly": True,
        "fixed": True,
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
    name_en = (expert.get("name_en") or "").strip()
    aliases = normalize_aliases(expert.get("aliases"))

    zh_simplified = tw_to_simplified(name_zh) if name_zh else ""
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


def normalize_source(item: dict) -> dict:
    item = item or {}
    source_type = (item.get("type") or "rss").strip().lower()
    if source_type not in ["rss", "domain", "cn_official"]:
        source_type = "rss"

    normalized = {
        "name": (item.get("name") or "").strip(),
        "subsource": (item.get("subsource") or "").strip(),
        "type": source_type,
        "url": (item.get("url") or "").strip(),
        "category": normalize_category(item.get("category")),
        "region": (item.get("region") or "").strip(),
        "enabled": bool(item.get("enabled", True)),
        "description": (item.get("description") or "").strip(),
        "readonly": bool(item.get("readonly", False)),
        "fixed": bool(item.get("fixed", False)),
    }
    return normalized


def normalize_expert(item: dict) -> dict:
    item = item or {}
    normalized = {
        "name_zh": (item.get("name_zh") or "").strip(),
        "name_en": (item.get("name_en") or "").strip(),
        "aliases": normalize_aliases(item.get("aliases")),
        "category": normalize_category(item.get("category")),
        "affiliation": (item.get("affiliation") or "").strip(),
        "region": (item.get("region") or "").strip(),
        "enabled": bool(item.get("enabled", True)),
        "description": (item.get("description") or "").strip(),
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

    if not name or not domain:
        return {}

    return {
        "name": f"[Global] {name}",
        "subsource": name,
        "type": "domain",
        "url": domain,
        "category": ["全球媒體", continent] if continent else ["全球媒體"],
        "region": country or continent,
        "enabled": True,
        "description": f"global media | language={language} | type={media_type}",
        "readonly": False,
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


def load_sources(editable_only=False):
    data = read_json(SOURCES_PATH)
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

    global_sources = load_global_media_sources()

    if editable_only:
        return editable + global_sources

    all_sources = editable + global_sources + deepcopy(FIXED_CN_OFFICIAL_SOURCES)

    return all_sources


def save_sources(sources):
    normalized = []
    for item in (sources or []):
        n = normalize_source(item)
        if n.get("fixed") or n.get("readonly") or n["type"] == "cn_official":
            continue
        normalized.append(n)
    write_json(SOURCES_PATH, normalized)


def load_experts():
    data = read_json(EXPERTS_PATH)
    if not isinstance(data, list):
        data = []
    return [normalize_expert(item) for item in data]


def save_experts(experts):
    normalized = [normalize_expert(item) for item in (experts or [])]
    write_json(EXPERTS_PATH, normalized)


def load_insights(path="config/insights.json"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
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


def save_insights(insights, path="config/insights.json"):
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


def load_auto_export():

    import json
    import os

    if not os.path.exists(AUTO_EXPORT_PATH):

        return {
            "enabled": True,
            "schedules": []
        }

    try:

        with open(AUTO_EXPORT_PATH, "r", encoding="utf-8") as f:

            data = json.load(f)

        if "schedules" not in data:
            data["schedules"] = []

        if "enabled" not in data:
            data["enabled"] = True

        return data

    except:

        return {
            "enabled": True,
            "schedules": []
        }


def save_auto_export(config):

    import json
    import os

    os.makedirs("config", exist_ok=True)

    with open(AUTO_EXPORT_PATH, "w", encoding="utf-8") as f:

        json.dump(config, f, ensure_ascii=False, indent=2)


def load_auto_export_state():
    data = read_json(AUTO_EXPORT_STATE_PATH)
    if not isinstance(data, dict):
        data = {"last_runs": {}}
    data.setdefault("last_runs", {})
    return data


def save_auto_export_state(data: dict):
    write_json(AUTO_EXPORT_STATE_PATH, data or {"last_runs": {}})


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
        "name": source.get("name", ""),
        "type": source.get("type", "rss"),
        "url": source.get("url", ""),
        "category": list_to_csv(source.get("category")),
        "region": source.get("region", ""),
        "enabled": bool(source.get("enabled", True)),
        "description": source.get("description", ""),
    }


def editor_row_to_source(row: dict):
    return normalize_source(
        {
            "name": row.get("name", ""),
            "type": row.get("type", "rss"),
            "url": row.get("url", ""),
            "category": row.get("category", ""),
            "region": row.get("region", ""),
            "enabled": row.get("enabled", True),
            "description": row.get("description", ""),
        }
    )


def expert_to_editor_row(expert: dict):
    return {
        "name_zh": expert.get("name_zh", ""),
        "name_en": expert.get("name_en", ""),
        "aliases": list_to_csv(expert.get("aliases")),
        "category": list_to_csv(expert.get("category")),
        "affiliation": expert.get("affiliation", ""),
        "region": expert.get("region", ""),
        "enabled": bool(expert.get("enabled", True)),
        "description": expert.get("description", ""),
    }


def editor_row_to_expert(row: dict):
    return normalize_expert(
        {
            "name_zh": row.get("name_zh", ""),
            "name_en": row.get("name_en", ""),
            "aliases": row.get("aliases", ""),
            "category": row.get("category", ""),
            "affiliation": row.get("affiliation", ""),
            "region": row.get("region", ""),
            "enabled": row.get("enabled", True),
            "description": row.get("description", ""),
        }
    )