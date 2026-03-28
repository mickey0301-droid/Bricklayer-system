import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER   = os.path.normpath(os.path.join(_HERE, "..", "data"))
CACHE_FILE    = os.path.join(DATA_FOLDER, "sentence_cache.json")
GH_CACHE_PATH = "data/sentence_cache.json"


def _ensure_data_folder():
    os.makedirs(DATA_FOLDER, exist_ok=True)


# ── GitHub helpers（reuse from vocab_manager）──────────────

def _gh_read(gh_path: str):
    try:
        from utils.vocab_manager import _github_read
        data, _ = _github_read(gh_path)
        return data
    except Exception:
        return None


def _gh_write(gh_path: str, data: dict, message: str):
    try:
        from utils.vocab_manager import _github_write
        content_str = json.dumps(data, ensure_ascii=False, indent=2)
        ok, err = _github_write(gh_path, content_str, None, message)
        # 句子快取失敗只記 log，不打擾使用者
        if not ok:
            print(f"[sentence_cache] GitHub sync failed: {err}")
    except Exception as e:
        print(f"[sentence_cache] GitHub sync error: {e}")


# ── Cache load/save ────────────────────────────────────────

def load_sentence_cache() -> dict:
    """本地優先，本地無資料時才從 GitHub 下載。"""
    # 1. 本地優先（絕對路徑，最可靠）
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                return data
        except Exception:
            pass

    # 2. 本地無資料時從 GitHub 下載（首次安裝 / 換電腦）
    gh_data = _gh_read(GH_CACHE_PATH)
    if gh_data is not None:
        _ensure_data_folder()
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(gh_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return gh_data

    return {}


def save_sentence_cache(cache: dict):
    """同時存到本地和 GitHub。"""
    _ensure_data_folder()
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _gh_write(GH_CACHE_PATH, cache, "Update sentence cache")


def _make_key(language: str, code: str) -> str:
    return f"{language}::{code}"


def get_cached_sentence(language: str, code: str) -> dict:
    cache = load_sentence_cache()
    key = _make_key(language, str(code))
    return cache.get(key, {
        "sentence":    "",
        "reading":     "",
        "translation": "",
        "grammar":     "",
    })


def set_cached_sentence(language: str, code: str, sentence_data: dict):
    cache = load_sentence_cache()
    key = _make_key(language, str(code))
    cache[key] = {
        "sentence":    sentence_data.get("sentence", ""),
        "reading":     sentence_data.get("reading", ""),
        "translation": sentence_data.get("translation", ""),
        "grammar":     sentence_data.get("grammar", ""),  # 包含 grammar
    }
    save_sentence_cache(cache)
