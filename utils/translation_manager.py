import json
import os
import time
import uuid

from utils.vocab_manager import DATA_FOLDER, ensure_data_folder, _github_read, _github_write


TRANSLATION_SENTENCES_FILE = os.path.join(DATA_FOLDER, "translation_sentences.json")
GH_TRANSLATION_SENTENCES_PATH = "data/translation_sentences.json"


def _normalize_entry(entry: dict) -> dict:
    translations = entry.get("translations", {})
    if not isinstance(translations, dict):
        translations = {}
    return {
        "id": str(entry.get("id") or uuid.uuid4().hex),
        "source": str(entry.get("source", "") or "").strip(),
        "translations": translations,
        "created_at": float(entry.get("created_at") or time.time()),
    }


def load_translation_sentences() -> list:
    ensure_data_folder()
    data = None
    if os.path.exists(TRANSLATION_SENTENCES_FILE):
        try:
            with open(TRANSLATION_SENTENCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

    if data is None:
        gh_data, _ = _github_read(GH_TRANSLATION_SENTENCES_PATH)
        if isinstance(gh_data, list):
            data = gh_data
            _save_locally(data)

    if not isinstance(data, list):
        return []
    return [_normalize_entry(item) for item in data if str(item.get("source", "")).strip()]


def _save_locally(sentences: list):
    ensure_data_folder()
    with open(TRANSLATION_SENTENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)


def save_translation_sentences(sentences: list):
    normalized = [_normalize_entry(item) for item in sentences if str(item.get("source", "")).strip()]
    _save_locally(normalized)
    content = json.dumps(normalized, ensure_ascii=False, indent=2)
    _github_write(
        GH_TRANSLATION_SENTENCES_PATH,
        content,
        None,
        f"Update translation sentences ({len(normalized)} entries)",
    )


def add_translation_sentence(source: str, translations: dict) -> list:
    source = source.strip()
    if not source:
        return load_translation_sentences()

    sentences = load_translation_sentences()
    for entry in sentences:
        if entry["source"] == source:
            entry["translations"].update(translations)
            save_translation_sentences(sentences)
            return sentences

    sentences.append({
        "id": uuid.uuid4().hex,
        "source": source,
        "translations": translations,
        "created_at": time.time(),
    })
    save_translation_sentences(sentences)
    return sentences


def update_translation_grammar(sentence_id: str, language: str, grammar: str) -> list:
    sentences = load_translation_sentences()
    for entry in sentences:
        if entry["id"] != sentence_id:
            continue
        translations = entry.setdefault("translations", {})
        lang_data = translations.setdefault(language, {})
        lang_data["grammar"] = str(grammar or "").strip()
        save_translation_sentences(sentences)
        return sentences
    return sentences


def sync_translation_sentences_from_github() -> bool:
    ensure_data_folder()
    gh_data, _ = _github_read(GH_TRANSLATION_SENTENCES_PATH)
    if isinstance(gh_data, list):
        _save_locally([_normalize_entry(item) for item in gh_data])
        return True
    return False
