import os
import json
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER   = os.path.normpath(os.path.join(_HERE, "..", "data"))
PROGRESS_FILE = os.path.join(DATA_FOLDER, "study_progress.json")
HISTORY_FILE  = os.path.join(DATA_FOLDER, "learning_history.json")

GH_PROGRESS_PATH = "data/study_progress.json"
GH_HISTORY_PATH  = "data/learning_history.json"


def _ensure_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)


# ── 模組級記憶體快取（Streamlit 每次 rerun 不會重新載入模組）──────────
_MEM_CACHE_PROGRESS: dict | None = None


def _invalidate_progress_cache():
    global _MEM_CACHE_PROGRESS
    _MEM_CACHE_PROGRESS = None


# ── GitHub helpers（reuse from vocab_manager）──────────────

def _gh_read(gh_path: str):
    try:
        from utils.vocab_manager import _github_read
        data, sha = _github_read(gh_path)
        return data
    except Exception:
        return None


def _gh_write(gh_path: str, data: dict, message: str):
    # 資料已存本地（local-first），GitHub 只是雲端備份，失敗不影響使用，靜默忽略
    try:
        from utils.vocab_manager import _github_write
        content_str = json.dumps(data, ensure_ascii=False, indent=2)
        _github_write(gh_path, content_str, None, message)
    except Exception:
        pass


# ── 從 GitHub 強制同步到本地 ──────────────────────────────

def sync_progress_from_github() -> bool:
    """從 GitHub 強制拉取學習進度並覆寫本地。"""
    _ensure_data_folder()
    gh_data = _gh_read(GH_PROGRESS_PATH)
    if gh_data is not None:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(gh_data, f, indent=2)
        return True
    return False


def sync_history_from_github() -> bool:
    """從 GitHub 強制拉取學習記錄並覆寫本地。"""
    _ensure_data_folder()
    gh_data = _gh_read(GH_HISTORY_PATH)
    if gh_data is not None:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(gh_data, f, ensure_ascii=False, indent=2)
        return True
    return False


# ── Progress ───────────────────────────────────────────────

def load_progress() -> dict:
    """有 token → 首次從 GitHub 載入並記憶體快取；後續 rerun 直接走快取，不重複呼叫 GitHub API。"""
    global _MEM_CACHE_PROGRESS
    if _MEM_CACHE_PROGRESS is not None:
        return _MEM_CACHE_PROGRESS

    # 1. 有 token → 從 GitHub 拿最新版本（雲端環境，僅首次呼叫）
    gh_data = _gh_read(GH_PROGRESS_PATH)
    if gh_data is not None:
        _ensure_data_folder()
        try:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(gh_data, f, indent=2)
        except Exception:
            pass
        _MEM_CACHE_PROGRESS = gh_data
        return _MEM_CACHE_PROGRESS

    # 2. 無 token 或 GitHub 讀取失敗 → 本地檔案
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                _MEM_CACHE_PROGRESS = data
                return _MEM_CACHE_PROGRESS
        except Exception:
            pass

    _MEM_CACHE_PROGRESS = {}
    return _MEM_CACHE_PROGRESS


def save_progress(progress: dict):
    """同時更新記憶體快取、存到本地和 GitHub。"""
    global _MEM_CACHE_PROGRESS
    _MEM_CACHE_PROGRESS = progress          # 立即更新記憶體，下一次 rerun 不需重讀
    _ensure_data_folder()
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        pass
    _gh_write(GH_PROGRESS_PATH, progress, "Update study progress")


def get_language_progress(language: str) -> int:
    progress = load_progress()
    return progress.get(language, 0)


def update_language_progress(language: str, index: int):
    progress = load_progress()
    progress[language] = index
    save_progress(progress)
    _log_daily_progress(language, index)


# ── History ────────────────────────────────────────────────

def _load_history() -> dict:
    # 1. 本地優先
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                return data
        except Exception:
            pass
    # 2. 本地無資料時從 GitHub 下載
    gh_data = _gh_read(GH_HISTORY_PATH)
    if gh_data is not None:
        return gh_data
    return {}


def _save_history(history: dict):
    _ensure_data_folder()
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _gh_write(GH_HISTORY_PATH, history, "Update learning history")


def _log_daily_progress(language: str, index: int):
    today   = str(date.today())
    history = _load_history()

    if language not in history:
        history[language] = {}

    current_today = history[language].get(today, -1)
    if index > current_today:
        history[language][today] = index
        _save_history(history)


def get_learning_history(language: str) -> dict:
    history = _load_history()
    return dict(sorted(history.get(language, {}).items()))
