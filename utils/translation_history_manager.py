import json
import os
import time
import uuid

from utils.vocab_manager import DATA_FOLDER, ensure_data_folder, _github_read, _github_write


TRANSLATION_HISTORY_FILE = os.path.join(DATA_FOLDER, "translation_history.json")
GH_TRANSLATION_HISTORY_PATH = "data/translation_history.json"


def _normalize_entry(entry: dict) -> dict:
    return {
        "id": str(entry.get("id") or uuid.uuid4().hex),
        "original_text": str(entry.get("original_text", "") or "").strip(),
        "translated_sentence": str(entry.get("translated_sentence", "") or "").strip(),
        "target_language": str(entry.get("target_language", "") or "").strip(),
        "target_mode": str(entry.get("target_mode", "") or "").strip(),
        "updated_at": float(entry.get("updated_at") or time.time()),
    }


def load_translation_history() -> list:
    ensure_data_folder()
    data = None
    if os.path.exists(TRANSLATION_HISTORY_FILE):
        try:
            with open(TRANSLATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    if data is None:
        gh_data, _ = _github_read(GH_TRANSLATION_HISTORY_PATH)
        if isinstance(gh_data, list):
            data = gh_data
            _save_locally(data)
    if data is None:
        return []
    if not isinstance(data, list):
        return []
    normalized = [_normalize_entry(item) for item in data]
    return [
        item for item in normalized
        if item["original_text"] and item["translated_sentence"] and item["target_language"]
    ]


def _save_locally(entries: list):
    ensure_data_folder()
    with open(TRANSLATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def save_translation_history(entries: list):
    normalized = [_normalize_entry(item) for item in entries]
    normalized = [
        item for item in normalized
        if item["original_text"] and item["translated_sentence"] and item["target_language"]
    ]
    _save_locally(normalized)
    content = json.dumps(normalized, ensure_ascii=False, indent=2)
    _github_write(
        GH_TRANSLATION_HISTORY_PATH,
        content,
        None,
        f"Update translation history ({len(normalized)} entries)",
    )


def upsert_translation_history_entry(
    original_text: str,
    translated_sentence: str,
    target_language: str,
    target_mode: str = "",
) -> list:
    original_text = str(original_text or "").strip()
    translated_sentence = str(translated_sentence or "").strip()
    target_language = str(target_language or "").strip()
    target_mode = str(target_mode or "").strip()
    if not original_text or not translated_sentence or not target_language:
        return load_translation_history()

    entries = load_translation_history()
    now = time.time()
    for item in entries:
        if (
            item.get("original_text") == original_text
            and item.get("target_language") == target_language
            and item.get("target_mode", "") == target_mode
        ):
            item["translated_sentence"] = translated_sentence
            item["updated_at"] = now
            save_translation_history(entries)
            return entries

    entries.append({
        "id": uuid.uuid4().hex,
        "original_text": original_text,
        "translated_sentence": translated_sentence,
        "target_language": target_language,
        "target_mode": target_mode,
        "updated_at": now,
    })
    save_translation_history(entries)
    return entries
