import os
import json
import base64

DATA_FOLDER = "data"
LANGUAGES_FILE = os.path.join(DATA_FOLDER, "languages.json")
GH_PATH = "data/languages.json"

DEFAULT_LANGUAGES = [
    {
        "key": "japanese",
        "label": "Japanese",
        "reading_label": "Hiragana",
        "supports_reading": True
    },
    {
        "key": "spanish",
        "label": "Spanish",
        "reading_label": "Reading",
        "supports_reading": False
    },
    {
        "key": "korean",
        "label": "Korean",
        "reading_label": "Romanization",
        "supports_reading": True
    }
]


def ensure_data_folder():
    os.makedirs(DATA_FOLDER, exist_ok=True)


# ── GitHub API helpers ────────────────────────────────────

def _github_config():
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "mickey0301-droid/Bricklayer-system")
        return token, repo
    except Exception:
        return "", ""


def _github_read(gh_path: str):
    import requests
    token, repo = _github_config()
    if not token:
        return None, None
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{gh_path}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            d = r.json()
            content = base64.b64decode(d["content"]).decode("utf-8")
            return json.loads(content), d["sha"]
    except Exception:
        pass
    return None, None


def _github_write(gh_path: str, content_str: str, sha, message: str) -> bool:
    import requests
    token, repo = _github_config()
    if not token:
        return False
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{gh_path}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        body = {
            "message": message,
            "content": base64.b64encode(content_str.encode("utf-8")).decode("utf-8"),
        }
        if sha:
            body["sha"] = sha
        r = requests.put(url, headers=headers, json=body, timeout=15)
        return r.status_code in (200, 201)
    except Exception:
        return False


# ── 語言讀寫 ──────────────────────────────────────────────

def load_languages():
    ensure_data_folder()

    # 1. 嘗試讀本地檔案
    local_data = None
    if os.path.exists(LANGUAGES_FILE):
        try:
            with open(LANGUAGES_FILE, "r", encoding="utf-8") as f:
                local_data = json.load(f)
        except Exception:
            local_data = None

    # 2. 本地為空 → 從 GitHub 拉取
    if not local_data:
        gh_data, _ = _github_read(GH_PATH)
        if gh_data:
            with open(LANGUAGES_FILE, "w", encoding="utf-8") as f:
                json.dump(gh_data, f, ensure_ascii=False, indent=2)
            local_data = gh_data

    # 3. 仍然為空 → 寫入預設語言並同步 GitHub
    if not local_data:
        local_data = DEFAULT_LANGUAGES
        _save_locally(local_data)
        _sync_to_github(local_data, "Initialize default languages")

    if not isinstance(local_data, list):
        return DEFAULT_LANGUAGES

    normalized = []
    for item in local_data:
        key = str(item.get("key", "")).strip().lower()
        label = str(item.get("label", "")).strip()
        reading_label = str(item.get("reading_label", "Reading")).strip() or "Reading"
        supports_reading = bool(item.get("supports_reading", False))
        if not key or not label:
            continue
        normalized.append({
            "key": key,
            "label": label,
            "reading_label": reading_label,
            "supports_reading": supports_reading
        })

    return normalized if normalized else DEFAULT_LANGUAGES


def _save_locally(languages):
    ensure_data_folder()
    with open(LANGUAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(languages, f, ensure_ascii=False, indent=2)


def _sync_to_github(languages, message: str):
    content_str = json.dumps(languages, ensure_ascii=False, indent=2)
    _, sha = _github_read(GH_PATH)
    _github_write(GH_PATH, content_str, sha, message)


def save_languages(languages):
    _save_locally(languages)
    _sync_to_github(languages, f"Update languages ({len(languages)} entries)")


def make_language_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace(" ", "_").replace("-", "_")
    key = "".join(ch for ch in key if ch.isalnum() or ch == "_")
    return key


def add_language(label: str, reading_label: str = "Reading", supports_reading: bool = False):
    languages = load_languages()

    key = make_language_key(label)
    if not key:
        raise ValueError("Language name cannot be empty.")

    for lang in languages:
        if lang["key"] == key:
            raise ValueError("This language already exists.")

    languages.append({
        "key": key,
        "label": label.strip(),
        "reading_label": reading_label.strip() or "Reading",
        "supports_reading": supports_reading
    })

    save_languages(languages)
    return key


def get_language_config(language_key: str):
    languages = load_languages()
    for lang in languages:
        if lang["key"] == language_key:
            return lang
    return None
