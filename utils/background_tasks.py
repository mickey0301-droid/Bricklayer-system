import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from utils.vocab_manager import load_vocab, save_vocab, ensure_min_rows
from utils.sentence_cache import get_cached_sentence, set_cached_sentence


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TASKS_FILE = DATA_DIR / "background_tasks.json"

_LOCK = threading.Lock()
_WORKERS: Dict[str, threading.Thread] = {}
_SENTENCE_PREFETCH_KEYS: set[str] = set()


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_tasks() -> Dict[str, Any]:
    _ensure_data_dir()
    if not TASKS_FILE.exists():
        return {"tasks": {}}
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tasks": {}}


def _save_tasks(data: Dict[str, Any]):
    _ensure_data_dir()
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_task(task_id: str, updates: Dict[str, Any]):
    with _LOCK:
        data = _load_tasks()
        tasks = data.setdefault("tasks", {})
        if task_id not in tasks:
            return
        tasks[task_id].update(updates)
        tasks[task_id]["updated_at"] = time.time()
        _save_tasks(data)


def create_autocomplete_task(language: str) -> str:
    task_id = str(uuid.uuid4())

    with _LOCK:
        data = _load_tasks()
        tasks = data.setdefault("tasks", {})

        tasks[task_id] = {
            "task_id": task_id,
            "type": "autocomplete_vocab",
            "language": language,
            "status": "queued",   # queued / running / done / error
            "error": "",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        _save_tasks(data)

    return task_id


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    data = _load_tasks()
    return data.get("tasks", {}).get(task_id)


def get_latest_task_for_language(language: str, task_type: str = "autocomplete_vocab") -> Optional[Dict[str, Any]]:
    data = _load_tasks()
    tasks = list(data.get("tasks", {}).values())

    filtered = [
        t for t in tasks
        if t.get("language") == language and t.get("type") == task_type
    ]

    if not filtered:
        return None

    filtered.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return filtered[0]


def has_running_task(language: str, task_type: str = "autocomplete_vocab") -> bool:
    latest = get_latest_task_for_language(language, task_type)
    if not latest:
        return False
    return latest.get("status") in ("queued", "running")


def _run_autocomplete_task(task_id: str, language: str):
    try:
        _update_task(task_id, {"status": "running", "error": ""})

        from utils.ai_tools import autocomplete_dataframe

        df = load_vocab(language)
        completed_df = autocomplete_dataframe(df.copy(), language)
        completed_df = ensure_min_rows(completed_df, min_rows=10)
        save_vocab(language, completed_df)

        _update_task(task_id, {"status": "done"})
    except Exception as e:
        _update_task(task_id, {"status": "error", "error": str(e)})


def start_autocomplete_task(language: str) -> str:
    if has_running_task(language, "autocomplete_vocab"):
        latest = get_latest_task_for_language(language, "autocomplete_vocab")
        return latest["task_id"]

    task_id = create_autocomplete_task(language)

    worker = threading.Thread(
        target=_run_autocomplete_task,
        args=(task_id, language),
        daemon=True,
    )
    worker.start()
    _WORKERS[task_id] = worker

    return task_id


def start_sentence_prefetch_task(
    language: str,
    code: str,
    current_term: str,
    allowed_terms: list,
    term_meaning: str = "",
    term_reading: str = "",
    term_pos: str = "",
    current_code: int = 0,
    allowed_vocab: list | None = None,
    ai_provider: str = "",
    ai_model: str = "",
) -> bool:
    cache_key = f"{language}::{code}"
    if get_cached_sentence(language, code).get("sentence"):
        return False

    with _LOCK:
        if cache_key in _SENTENCE_PREFETCH_KEYS:
            return False
        _SENTENCE_PREFETCH_KEYS.add(cache_key)

    def _worker():
        try:
            from utils.study_engine import generate_example_sentence

            result = generate_example_sentence(
                language=language,
                current_term=current_term,
                allowed_terms=allowed_terms,
                term_meaning=term_meaning,
                term_reading=term_reading,
                term_pos=term_pos,
                current_code=current_code,
                allowed_vocab=allowed_vocab,
                ai_provider=ai_provider,
                ai_model=ai_model,
            )
            if result.get("sentence"):
                set_cached_sentence(language, code, result)
        except Exception:
            pass
        finally:
            with _LOCK:
                _SENTENCE_PREFETCH_KEYS.discard(cache_key)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    _WORKERS[cache_key] = worker
    return True
