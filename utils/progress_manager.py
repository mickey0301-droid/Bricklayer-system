import os
import json
from datetime import date

DATA_FOLDER = "data"
PROGRESS_FILE = os.path.join(DATA_FOLDER, "study_progress.json")
HISTORY_FILE  = os.path.join(DATA_FOLDER, "learning_history.json")

GH_PROGRESS_PATH = "data/study_progress.json"
GH_HISTORY_PATH  = "data/learning_history.json"


def _ensure_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)


# ── GitHub helpers（reuse from vocab_manager）──────────────

def _gh_read(gh_path: str):
    try:
        from utils.vocab_manager import _github_read
        data, sha = _github_read(gh_path)
        return data
    except Exception:
        return None


def _gh_write(gh_path: str, data: dict, message: str):
    try:
        from utils.vocab_manager import _github_write
        content_str = json.dumps(data, ensure_ascii=False, indent=2)
        _github_write(gh_path, content_str, None, message)
    except Exception:
        pass


# ── Progress ───────────────────────────────────────────────

def load_progress() -> dict:
    """GitHub 優先，失敗時讀本地檔案。"""
    gh_data = _gh_read(GH_PROGRESS_PATH)
    if gh_data is not None:
        # 同步寫回本地（加速後續讀取）
        _ensure_data_folder()
        try:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(gh_data, f, indent=2)
        except Exception:
            pass
        return gh_data

    # 本地 fallback
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(progress: dict):
    """同時存到本地和 GitHub。"""
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
    gh_data = _gh_read(GH_HISTORY_PATH)
    if gh_data is not None:
        return gh_data
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
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
