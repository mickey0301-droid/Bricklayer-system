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


# ── 從 GitHub 強制同步到本地 ──────────────────────────────

def sync_cache_from_github() -> bool:
    """從 GitHub 強制拉取例句快取並覆寫本地。"""
    _ensure_data_folder()
    gh_data = _gh_read(GH_CACHE_PATH)
    if gh_data is not None:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(gh_data, f, ensure_ascii=False, indent=2)
        return True
    return False


# ── 模組級記憶體快取（Streamlit 每次 rerun 不會重新載入模組）──────
_MEM_CACHE: dict | None = None


def _invalidate_mem_cache():
    global _MEM_CACHE
    _MEM_CACHE = None


# ── Cache load/save ────────────────────────────────────────

def load_sentence_cache() -> dict:
    """本地優先，本地無資料時才從 GitHub 下載。結果存於模組記憶體，避免重複讀磁碟。"""
    global _MEM_CACHE
    if _MEM_CACHE is not None:
        return _MEM_CACHE

    # 1. 本地優先（絕對路徑，最可靠）
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                _MEM_CACHE = data
                return _MEM_CACHE
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
        _MEM_CACHE = gh_data
        return _MEM_CACHE

    _MEM_CACHE = {}
    return _MEM_CACHE


def save_sentence_cache(cache: dict):
    """同時存到記憶體、本地和 GitHub。"""
    global _MEM_CACHE
    _MEM_CACHE = cache          # 更新記憶體快取
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


def set_cached_sentences_bulk(language: str, results: dict):
    """一次儲存多筆例句，只呼叫一次 save_sentence_cache（一次 GitHub write）。
    results: {code_str: sentence_data_dict}
    """
    if not results:
        return
    cache = load_sentence_cache()
    for code, sentence_data in results.items():
        key = _make_key(language, str(code))
        cache[key] = {
            "sentence":    sentence_data.get("sentence", ""),
            "reading":     sentence_data.get("reading", ""),
            "translation": sentence_data.get("translation", ""),
            "grammar":     sentence_data.get("grammar", ""),
        }
    save_sentence_cache(cache)
