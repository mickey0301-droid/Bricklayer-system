import csv
import json
import re
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LANGUAGES_FILE = DATA_DIR / "languages.json"
PROGRESS_FILE = DATA_DIR / "progress.json"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_languages():
    ensure_data_dir()
    return load_json(LANGUAGES_FILE, {})


def save_languages(languages: dict):
    save_json(LANGUAGES_FILE, languages)


def load_progress():
    ensure_data_dir()
    return load_json(PROGRESS_FILE, {})


def save_progress(progress: dict):
    save_json(PROGRESS_FILE, progress)


def slugify_language_key(name: str) -> str:
    """
    Convert a display name like 'Japanese' or 'Latin American Spanish'
    into a stable language key like 'japanese' or 'latin_american_spanish'.
    """
    value = name.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def vocab_file(language_key: str) -> Path:
    return DATA_DIR / f"{language_key}_vocab.csv"


def sentences_file(language_key: str) -> Path:
    return DATA_DIR / f"{language_key}_sentences.json"


def ensure_vocab_file(language_key: str):
    path = vocab_file(language_key)
    if not path.exists():
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "word", "reading", "meaning"])


def ensure_sentences_file(language_key: str):
    path = sentences_file(language_key)
    if not path.exists():
        save_json(path, {})


def ensure_language_files(language_key: str):
    ensure_data_dir()
    ensure_vocab_file(language_key)
    ensure_sentences_file(language_key)


def create_language(language_name: str, supports_reading: bool, reading_label: str):
    languages = load_languages()

    language_key = slugify_language_key(language_name)
    if not language_key:
        raise ValueError("Language name is invalid.")

    if language_key in languages:
        raise ValueError(f"Language '{language_key}' already exists.")

    languages[language_key] = {
        "label": language_name.strip(),
        "reading_label": reading_label.strip() if supports_reading else "",
        "supports_reading": bool(supports_reading),
    }

    save_languages(languages)
    ensure_language_files(language_key)

    progress = load_progress()
    if language_key not in progress:
        progress[language_key] = {}
        save_progress(progress)

    return language_key


def update_language(language_key: str, label: str, supports_reading: bool, reading_label: str):
    languages = load_languages()
    if language_key not in languages:
        raise ValueError(f"Language '{language_key}' not found.")

    languages[language_key] = {
        "label": label.strip(),
        "reading_label": reading_label.strip() if supports_reading else "",
        "supports_reading": bool(supports_reading),
    }

    save_languages(languages)
    ensure_language_files(language_key)


def delete_language(language_key: str, delete_files: bool = True):
    languages = load_languages()
    if language_key not in languages:
        raise ValueError(f"Language '{language_key}' not found.")

    del languages[language_key]
    save_languages(languages)

    progress = load_progress()
    if language_key in progress:
        del progress[language_key]
        save_progress(progress)

    if delete_files:
        vf = vocab_file(language_key)
        sf = sentences_file(language_key)

        if vf.exists():
            vf.unlink()
        if sf.exists():
            sf.unlink()


def render_language_manager_page():
    st.title("🌍 Language Manager")
    st.caption("Create, edit, and delete languages for Bricklayer.")

    languages = load_languages()

    # ---------- Create New Language ----------
    st.subheader("Create New Language")

    with st.form("create_language_form", clear_on_submit=True):
        new_language_name = st.text_input("Language name", placeholder="e.g. Korean")
        new_supports_reading = st.checkbox("Supports readings", value=False)
        new_reading_label = st.text_input(
            "Reading label",
            placeholder="e.g. Hiragana / Romanization",
            disabled=not new_supports_reading,
        )

        submitted = st.form_submit_button("Create Language", use_container_width=True)

        if submitted:
            try:
                if not new_language_name.strip():
                    st.error("Please enter a language name.")
                else:
                    create_language(
                        language_name=new_language_name,
                        supports_reading=new_supports_reading,
                        reading_label=new_reading_label,
                    )
                    st.success(f"Language '{new_language_name}' created successfully.")
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ---------- Existing Languages ----------
    st.subheader("Existing Languages")

    if not languages:
        st.info("No languages found yet.")
        return

    for language_key, config in languages.items():
        label = config.get("label", language_key)
        supports_reading = config.get("supports_reading", False)
        reading_label = config.get("reading_label", "")

        with st.expander(f"{label}  ({language_key})", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Language key:** `{language_key}`")
                st.write(f"**Supports reading:** {'Yes' if supports_reading else 'No'}")
                st.write(f"**Reading label:** {reading_label if reading_label else '—'}")

                vf = vocab_file(language_key)
                sf = sentences_file(language_key)

                st.write(f"**Vocab file:** `{vf.name}` {'✅' if vf.exists() else '❌'}")
                st.write(f"**Sentence file:** `{sf.name}` {'✅' if sf.exists() else '❌'}")

            with col2:
                st.markdown("#### Edit Settings")

                edit_label = st.text_input(
                    "Display label",
                    value=label,
                    key=f"edit_label_{language_key}",
                )
                edit_supports_reading = st.checkbox(
                    "Supports readings",
                    value=supports_reading,
                    key=f"edit_supports_{language_key}",
                )
                edit_reading_label = st.text_input(
                    "Reading label",
                    value=reading_label,
                    key=f"edit_reading_{language_key}",
                    disabled=not edit_supports_reading,
                )

                save_col, delete_col = st.columns(2)

                with save_col:
                    if st.button("Save Changes", key=f"save_{language_key}", use_container_width=True):
                        try:
                            update_language(
                                language_key=language_key,
                                label=edit_label,
                                supports_reading=edit_supports_reading,
                                reading_label=edit_reading_label,
                            )
                            st.success(f"Updated '{language_key}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                with delete_col:
                    confirm_delete = st.checkbox(
                        "Confirm delete",
                        key=f"confirm_delete_{language_key}",
                    )
                    if st.button("Delete Language", key=f"delete_{language_key}", use_container_width=True):
                        if not confirm_delete:
                            st.warning("Please tick 'Confirm delete' first.")
                        else:
                            try:
                                delete_language(language_key, delete_files=True)
                                st.success(f"Deleted '{language_key}'.")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))