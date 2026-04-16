import io
import os
import zipfile
import random
import subprocess
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from utils.vocab_manager import (
    load_vocab, save_vocab, ensure_min_rows,
    load_pattern_vocab, save_pattern_vocab,
    DATA_FOLDER, ensure_data_folder,
    sync_vocab_from_github, sync_pattern_vocab_from_github,
    _github_write, _github_config,
)
from utils.study_engine import (
    prepare_study_df,
    get_allowed_vocab,
    get_current_row,
    get_next_index,
    get_prev_index,
    generate_example_sentence,
    generate_recombination_sentence,
    generate_fsi_sentence,
    generate_passage,
    GRAMMAR_PATTERNS,
)
from utils.tts_engine import (
    generate_tts_audio, audio_player, audio_player_dual, audio_player_pausable,
    get_cached_tts, set_cached_tts,
)
from utils.sentence_cache import get_cached_sentence, set_cached_sentence, set_cached_sentences_bulk, sync_cache_from_github
from utils.progress_manager import (
    get_language_progress,
    update_language_progress,
    get_learning_history,
    sync_progress_from_github,
    sync_history_from_github,
    get_progress_sync_status,
)
from utils.app_logger import log_error
from utils.language_manager import load_languages, add_language, get_language_config
from utils.ai_tools import explain_translated_sentence_grammar, translate_chinese_sentence
from utils.translation_manager import (
    add_translation_sentence,
    load_translation_sentences,
    sync_translation_sentences_from_github,
    update_translation_grammar,
)
from utils.familiarity_manager import (
    get_familiarity, set_familiarity, get_sample_weights,
    FAMILIAR, UNFAMILIAR,
)
from utils.background_tasks import (
    start_autocomplete_task,
    get_latest_task_for_language,
    start_sentence_prefetch_task,
)

st.set_page_config(page_title="Bricklayer", layout="wide")

# ── 語言對應國旗 ──────────────────────────────────────────
LANGUAGE_FLAGS = {
    "japanese": "🇯🇵",
    "korean":   "🇰🇷",
    "spanish":  "🇪🇸",
    "french":   "🇫🇷",
    "german":   "🇩🇪",
    "italian":  "🇮🇹",
    "portuguese": "🇵🇹",
    "chinese":  "🇨🇳",
    "russian":  "🇷🇺",
    "arabic":   "🇸🇦",
}

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding-top: 4.5rem;   /* 需蓋過 Streamlit 固定 header（約 3.5rem）*/
    padding-bottom: 2rem;
}
h1 { font-size: 2.4rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.3rem !important; }

div[data-testid="stButton"] > button {
    min-height: 3.2rem;
    font-size: 1.1rem;
    border-radius: 12px;
}
div[data-testid="stDataEditor"] * {
    font-size: 1rem !important;
}

/* 詞彙卡 / 例句卡 */
.study-card {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    background: #f7f9fc;
    border: 1px solid #dfe7f3;
    margin-bottom: 1rem;
    height: 100%;
}
.study-label {
    font-size: 0.85rem;
    color: #5b6575;
    margin-bottom: 0.15rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.study-value-lg {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    line-height: 1.2;
}
.study-value-md {
    font-size: 1.15rem;
    margin-bottom: 0.6rem;
}
.grammar-box {
    background: #eef4fb;
    border-left: 3px solid #4F8BF9;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.95rem;
    color: #334;
    margin-top: 0.4rem;
    margin-bottom: 0.8rem;
    line-height: 1.7;
}

/* 首頁語言按鈕加大 */
.lang-btn > button {
    height: 4rem !important;
    font-size: 1.2rem !important;
}

/* 統計進度條 */
.stat-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.stat-label { min-width: 130px; font-size: 1rem; }
.stat-bar-bg {
    flex: 1;
    height: 10px;
    background: #dfe7f3;
    border-radius: 99px;
    overflow: hidden;
}
.stat-bar-fill {
    height: 100%;
    background: #4F8BF9;
    border-radius: 99px;
}
.stat-count { font-size: 0.9rem; color: #5b6575; min-width: 60px; text-align: right; }

/* ── 手機：兩欄改成上下堆疊 + 字體略縮 ── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stColumn"] {
        width: 100% !important;
        flex: 0 0 auto !important;
        min-width: 0 !important;
    }
    .study-value-lg { font-size: 1.5rem !important; }
    .study-value-md { font-size: 0.88rem !important; }
    .study-label    { font-size: 0.7rem !important; }
    .grammar-box    { font-size: 0.78rem !important; padding: 0.4rem 0.5rem !important; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.15rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state 初始化 ───────────────────────────────────
_defaults = {
    "page": "home",
    "language": None,
    "vocab_df": None,
    "vocab_loaded_language": None,
    "study_index": 0,
    "study_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": ""},
    "study_sentence_term": "",
    "study_current_code": "",
    "autocomplete_task_id": None,
    "autocomplete_task_seen_done": False,
    "tts_term_audio": None,
    "tts_term_for": "",
    "tts_sentence_audio": None,
    "tts_sentence_for": "",
    "auto_played_for": "",
    "review_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": ""},
    "review_term": "",
    "review_meaning": "",
    "review_term_code": 0,
    "review_term_reading": "",
    "review_term_pos": "",
    "review_show_answer": False,
    "review_auto_played_for": "",
    # FSI drill sentences
    "review_sub_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""},
    "review_trans_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""},
    "review_sub_prev_notes": [],
    "review_trans_prev_notes": [],
    # 重組練習模式
    "combo_words": [],
    "combo_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": ""},
    "combo_show_answer": False,
    "combo_pattern": {"label": "", "instruction": ""},
    "combo_auto_played_for": "",
    # 句型學習模式（逐筆瀏覽，與字彙學習同版面）
    "pattern_vocab_df": None,
    "pattern_vocab_loaded_language": None,
    "pattern_study_index": 0,
    "pattern_study_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": ""},
    "pattern_study_sentence_term": "",
    "pattern_study_current_code": "",
    "pattern_tts_term_audio": None,
    "pattern_tts_term_for": "",
    "pattern_tts_sentence_audio": None,
    "pattern_tts_sentence_for": "",
    "pattern_review_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": ""},
    "pattern_review_term": "",
    "pattern_review_meaning": "",
    "pattern_review_term_code": 0,
    "pattern_review_term_reading": "",
    "pattern_review_term_pos": "",
    "pattern_review_sub_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""},
    "pattern_review_trans_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""},
    "pattern_review_sub_prev_notes": [],
    "pattern_review_trans_prev_notes": [],
    "pattern_combo_words": [],
    "pattern_combo_sentence": {"sentence": "", "reading": "", "translation": "", "grammar": ""},
    "pattern_combo_show_answer": False,
    "pattern_combo_auto_played_for": "",
    "pattern_combo_pattern": {"label": "", "instruction": ""},
    "pattern_review_code_min": None,
    "pattern_review_code_max": None,
    "pattern_review_show_answer": False,
    "pattern_study_auto_played_for": "",
    "pattern_review_auto_played_for": "",
    # 短文練習模式
    "passage_type": "短文",
    "passage_data": {"passage": "", "reading": "", "translation": "", "vocab_notes": [], "grammar_notes": "", "char_count": 0, "unit_label": "字元"},
    # 複習練習範圍
    "review_code_min": None,
    "review_code_max": None,
    # 批次例句預生成（記錄最近已生成的 batch 索引，-1 代表尚未生成）
    "study_batch_generated": -1,
    "pattern_study_batch_generated": -1,
    "translation_input": "",
    "translation_review_index": 0,
    "translation_review_order": "依序",
    "translation_tts_audio": None,
    "translation_tts_for": "",
    # AI 設定
    "ai_provider": "openai",
    "ai_model": "",
    # 詞彙顯示偵錯：顯示被規則忽略的高編號詞
    "debug_vocab_filter": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _build_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True
        ).strip()
    except Exception:
        return "unknown"


# ── 例句詞彙查找 ───────────────────────────────────────────
def find_used_vocab(sentence: str, vocab_df, current_code: int = 9999) -> list:
    """
    搜尋 vocab_df 中出現在 sentence 的詞彙，依 code 排序。
    code_num > current_code 的詞彙標記為 is_extra=True（超出範圍）。
    使用「長字優先」原則：若一個詞完全被另一個更長的詞覆蓋，則忽略它。
    """
    if vocab_df is None or vocab_df.empty or not sentence:
        return []

    # 第一步：找出所有匹配詞彙及其在句子中的位置
    candidates = []
    seen_terms = set()
    for _, row in vocab_df.iterrows():
        term = str(row.get("term", "")).strip()
        if not term or term in seen_terms:
            continue
        seen_terms.add(term)
        code_num = int(row.get("code_num", 0))
        pos = 0
        while True:
            idx = sentence.find(term, pos)
            if idx == -1:
                break
            candidates.append({
                "code":     str(row.get("code", "")),
                "code_num": code_num,
                "term":     term,
                "meaning":  str(row.get("meaning", "")),
                "pos":      str(row.get("pos", "")),
                "is_extra": code_num > current_code,
                "start":    idx,
                "end":      idx + len(term),
            })
            pos = idx + 1

    # 第二步：移除被更長詞彙完全覆蓋的匹配（長字優先）
    filtered = [
        c for c in candidates
        if not any(
            o["start"] <= c["start"] and o["end"] >= c["end"] and len(o["term"]) > len(c["term"])
            for o in candidates
        )
    ]

    # 第三步：每個詞只保留一次，依 code 排序
    seen2, result = set(), []
    for c in sorted(filtered, key=lambda x: x["code_num"]):
        if c["term"] not in seen2:
            seen2.add(c["term"])
            result.append({k: v for k, v in c.items() if k not in ("start", "end")})
    return result


def _df_to_allowed_vocab(df) -> list:
    """將 prepared vocab DataFrame 轉為供 AI 使用的詞彙陣列。"""
    if df is None or df.empty:
        return []
    return [
        {
            "code": int(row["code_num"]),
            "term": str(row["term"]),
            "pos": str(row.get("pos", "")),
            "meaning": str(row.get("meaning", "")),
            "reading": str(row.get("reading", "")),
        }
        for _, row in df.iterrows()
    ]


def _sent_tts_text(sentence_data: dict, language: str) -> str:
    """
    回傳要傳給 TTS 的例句文字。
    日語有平假名標注時，用平假名（reading）取代漢字句子，
    確保 TTS 發音與畫面顯示的平假名完全一致。
    其他語言直接回傳 sentence。
    """
    if language == "japanese":
        reading = (sentence_data.get("reading") or "").strip()
        if reading:
            return reading
    return (sentence_data.get("sentence") or "").strip()


def _vocab_by_codes(codes: list, vocab_df, current_code: int) -> list:
    """依 vocab_codes 編號查找詞彙，用於 AI 已回傳編號的情況。"""
    result, seen = [], set()
    for c in codes:
        try:
            c_int = int(c)
        except (ValueError, TypeError):
            continue
        if c_int in seen:
            continue
        seen.add(c_int)
        rows = vocab_df[vocab_df["code_num"] == c_int]
        if not rows.empty:
            r = rows.iloc[0]
            result.append({
                "code":     str(r["code"]),
                "code_num": c_int,
                "term":     str(r["term"]),
                "meaning":  str(r.get("meaning", "")),
                "pos":      str(r.get("pos", "")),
                "is_extra": c_int > current_code,
            })
    return sorted(result, key=lambda x: x["code_num"])


def _is_ignorable_extra_vocab(item: dict, language: str) -> bool:
    pos = str(item.get("pos", "")).strip().lower()
    function_pos_keywords = (
        "助詞", "助动词", "助動詞", "連接詞", "接続詞", "冠詞", "介詞", "前置詞", "後置詞",
        "代名詞", "語尾", "詞尾", "접속", "조사", "어미", "보조", "관형사", "감탄사", "대명사",
        "particle", "auxiliary", "article", "preposition", "conjunction", "pronoun",
        "determiner", "postposition", "suffix",
    )
    if any(k in pos for k in function_pos_keywords):
        return True

    term = str(item.get("term", "")).strip()
    if language in ("japanese", "korean", "chinese") and len(term) <= 1:
        return True
    return False


def _merge_used_vocab(sentence: str, vocab_df, current_code: int, vocab_codes: list = None) -> list:
    """整合 AI 回傳編號與字串掃描結果，回傳去重後的詞彙命中清單。"""
    code_hits = _vocab_by_codes(vocab_codes or [], vocab_df, current_code) if vocab_codes else []
    string_hits = find_used_vocab(sentence, vocab_df, current_code)
    seen_code_nums = {v["code_num"] for v in code_hits}
    merged = list(code_hits)
    for v in string_hits:
        if v["code_num"] not in seen_code_nums:
            merged.append(v)
    return merged


def _grammar_has_new_format(grammar: str) -> bool:
    g = str(grammar or "").strip()
    if not g:
        return False
    return ("變化規則" in g) and ("• " in g or "\n•" in g)


def _strip_focus_section(grammar: str) -> str:
    g = str(grammar or "")
    i_focus = g.find("文法重點")
    if i_focus < 0:
        return g
    i_rules = g.find("變化規則")
    if i_rules > i_focus:
        left = g[:i_focus].rstrip()
        right = g[i_rules:].lstrip()
        return f"{left}\n{right}".strip()
    return g[:i_focus].rstrip()


def _ensure_display_grammar_format(grammar: str) -> str:
    g = _strip_focus_section(str(grammar or "")).strip()
    if not g:
        return ""
    if not _grammar_has_new_format(g):
        parts = [p.strip() for p in g.split("+") if p.strip()]
        verb_tokens, particle_tokens, noun_tokens, case_tokens = [], [], [], []
        for p in parts:
            token = p.split("(", 1)[0].split("[", 1)[0].strip()
            if not token:
                continue
            if "[動詞" in p or "[形容詞" in p:
                verb_tokens.append(token)
            if "助詞" in p or "介詞" in p or "連接詞" in p or "冠詞" in p:
                particle_tokens.append(token)
            if "[名詞" in p:
                noun_tokens.append(token)
            if any(k in p for k in ("主格", "受格", "與格", "屬格", "奪格", "工具格")):
                case_tokens.append(token)

        specific_rules, seen = [], set()
        if verb_tokens:
            t = verb_tokens[0]
            if t.endswith(("다", "る", "ir", "ar", "er")):
                specific_rules.append(f"• {t}：本句使用原形/辭書形，因為句型採簡潔敘述，未加入過去、敬語或否定標記。")
            else:
                specific_rules.append(f"• {t}：本句使用這個變化形，是為了配合句子的時態與語氣需求。")
            seen.add(t)
        for t in particle_tokens:
            if t in seen:
                continue
            specific_rules.append(f"• {t}：此成分在本句有明確語意功能，並且可接在前詞後標示其語法角色。")
            seen.add(t)
            if len(specific_rules) >= 3:
                break
        if len(specific_rules) < 3 and case_tokens:
            t = case_tokens[0]
            if t not in seen:
                specific_rules.append(f"• {t}：此格標記的選擇是為了對應該名詞在本句中的格功能。")
                seen.add(t)
        if len(specific_rules) < 3:
            for t in noun_tokens:
                if t in seen:
                    continue
                specific_rules.append(f"• {t}：此名詞本身不活用，透過後接成分在句中承擔角色。")
                seen.add(t)
                if len(specific_rules) >= 3:
                    break
        if not specific_rules:
            specific_rules = ["• 目標詞彙：此詞在本句依語境採用當前形式。"]
        g = (
            f"{g}\n變化規則：\n"
            + "\n".join(specific_rules)
        )
    return g


def _render_grammar_box(grammar: str):
    g = _ensure_display_grammar_format(grammar)
    if not g:
        return
    safe = (
        g.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace("\n", "<br>")
    )
    st.markdown(f'<div class="grammar-box">{safe}</div>', unsafe_allow_html=True)


def _is_cache_sentence_valid(sentence_data: dict, vocab_df, current_code: int, language: str) -> bool:
    sentence = str(sentence_data.get("sentence", "")).strip()
    # 臨時測試模式：先確認「能穩定生成句子」，
    # 不再用詞彙規則/文法格式把已生成句再判無效。
    return bool(sentence)


def render_used_vocab(sentence: str, vocab_df, current_code: int = 9999, vocab_codes: list = None):
    """在例句下方顯示使用到的詞彙 chip 清單。超出範圍的詞以橘色標示。
    合併 vocab_codes（AI 回傳，處理變形）與 string matching（掃描全句，補漏），
    以確保不遺漏任何使用到的詞彙。
    """
    merged = _merge_used_vocab(
        sentence=sentence,
        vocab_df=vocab_df,
        current_code=current_code,
        vocab_codes=vocab_codes,
    )

    language = st.session_state.get("language", "")
    ignored_extras = [
        v for v in merged
        if v.get("is_extra") and _is_ignorable_extra_vocab(v, language)
    ]
    used = sorted(
        [
            v for v in merged
            if not (v.get("is_extra") and _is_ignorable_extra_vocab(v, language))
        ],
        key=lambda x: x["code_num"]
    )
    if not used:
        return
    st.markdown('<div class="study-label" style="margin-top:0.5rem;font-size:0.72rem;">例句使用的詞彙</div>', unsafe_allow_html=True)
    chips = []
    for v in used:
        if v["is_extra"]:
            bg, border, code_color, meaning_color, badge = "#fff3e0", "#ffb74d", "#e65100", "#e65100", "&nbsp;↑"
        else:
            bg, border, code_color, meaning_color, badge = "#eef4fb", "#c5d8f0", "#999", "#4F8BF9", ""
        chips.append(
            f'<span style="display:inline-block;margin:0.1rem 0.2rem;background:{bg};'
            f'border:1px solid {border};border-radius:6px;padding:0.05rem 0.4rem;font-size:0.78rem;">'
            f'<b style="color:{code_color};">#{v["code"]}{badge}</b>&nbsp;'
            f'<span>{v["term"]}</span>&nbsp;'
            f'<span style="color:{meaning_color};">{v["meaning"]}</span></span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)

    if st.session_state.get("debug_vocab_filter", False) and ignored_extras:
        ignored_extras = sorted(ignored_extras, key=lambda x: x["code_num"])
        with st.expander("🔍 偵錯：已忽略的高編號詞（功能詞 / 疑似斷詞）", expanded=False):
            for v in ignored_extras:
                st.caption(f"#{v['code']} {v['term']}（{v.get('meaning','')}｜{v.get('pos','')}）")


# ── 導航函數 ───────────────────────────────────────────────
def go_home():
    st.session_state.page = "home"

def go_language(lang: str):
    st.session_state.language = lang
    st.session_state.page = "language_home"

def go_custom_vocab():
    st.session_state.page = "custom_vocab"

def go_translation_practice():
    st.session_state.language = None
    st.session_state.page = "translation_practice"

def go_study():
    language = st.session_state.language
    st.session_state.study_index = get_language_progress(language)
    st.session_state.study_batch_generated = -1
    st.session_state.page = "study"

def go_review():
    st.session_state.page = "review"
    st.session_state.review_show_answer = False
    st.session_state.review_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}
    st.session_state.review_term = ""
    st.session_state.review_sub_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}
    st.session_state.review_trans_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}
    st.session_state.review_sub_prev_notes = []
    st.session_state.review_trans_prev_notes = []
    st.session_state.combo_words = []
    st.session_state.combo_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}
    st.session_state.combo_show_answer = False
    st.session_state.combo_pattern = {"label": "", "instruction": ""}

def go_combo():
    go_review()
    st.session_state["review_mode_radio"] = "🔀 重組練習"

def go_pattern_vocab():
    st.session_state.page = "pattern_vocab"

def go_pattern_study():
    st.session_state.page = "pattern_study"
    st.session_state.pattern_study_index = 0
    st.session_state.pattern_study_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}
    st.session_state.pattern_study_sentence_term = ""
    st.session_state.pattern_study_current_code = ""
    st.session_state.pattern_tts_term_audio = None
    st.session_state.pattern_tts_term_for = ""
    st.session_state.pattern_tts_sentence_audio = None
    st.session_state.pattern_tts_sentence_for = ""
    st.session_state.pattern_study_batch_generated = -1

def go_pattern_review():
    st.session_state.page = "pattern_review"
    st.session_state.pattern_review_term = ""
    st.session_state.pattern_review_meaning = ""
    st.session_state.pattern_review_term_code = 0
    st.session_state.pattern_review_term_reading = ""
    st.session_state.pattern_review_term_pos = ""
    st.session_state.pattern_review_sub_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}
    st.session_state.pattern_review_trans_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}
    st.session_state.pattern_review_sub_prev_notes = []
    st.session_state.pattern_review_trans_prev_notes = []
    st.session_state.pattern_combo_words = []
    st.session_state.pattern_combo_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}
    st.session_state.pattern_combo_show_answer = False
    st.session_state.pattern_combo_pattern = {"label": "", "instruction": ""}
    st.session_state.pattern_review_code_min = None
    st.session_state.pattern_review_code_max = None

def go_settings():
    st.session_state.page = "settings"

def reset_vocab_state():
    st.session_state.vocab_df = None
    st.session_state.vocab_loaded_language = None

def reset_study_state():
    st.session_state.study_index = 0
    st.session_state.study_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}
    st.session_state.study_sentence_term = ""
    st.session_state.study_current_code = ""
    st.session_state.tts_term_audio = None
    st.session_state.tts_term_for = ""
    st.session_state.tts_sentence_audio = None
    st.session_state.tts_sentence_for = ""


# ── 詞庫工具 ───────────────────────────────────────────────
def load_vocab_into_state(language: str):
    df = load_vocab(language)
    df = ensure_min_rows(df, min_rows=10)
    st.session_state.vocab_df = df
    st.session_state.vocab_loaded_language = language

def get_current_vocab_df(language: str) -> pd.DataFrame:
    if (st.session_state.vocab_df is None
            or st.session_state.vocab_loaded_language != language):
        load_vocab_into_state(language)
    return st.session_state.vocab_df

def refresh_vocab_from_disk_if_task_finished(language: str):
    latest_task = get_latest_task_for_language(language, "autocomplete_vocab")
    if not latest_task:
        return
    if latest_task.get("status") == "done":
        current_seen = st.session_state.get("autocomplete_task_id")
        already_seen = st.session_state.get("autocomplete_task_seen_done", False)
        if current_seen == latest_task.get("task_id") and not already_seen:
            load_vocab_into_state(language)
            st.session_state.autocomplete_task_seen_done = True

def render_autocomplete_task_status(language: str):
    latest_task = get_latest_task_for_language(language, "autocomplete_vocab")
    if not latest_task:
        return
    status = latest_task.get("status")
    error = latest_task.get("error", "")
    if status in ("queued", "running"):
        st.info("AI 補全正在背景執行中。你可以先離開這一頁，完成後結果會自動存檔。")
    elif status == "done":
        st.success("背景 AI 補全已完成，結果已存入詞庫。")
    elif status == "error":
        st.error(f"背景 AI 補全失敗：{error}")

def load_pattern_vocab_into_state(language: str):
    df = load_pattern_vocab(language)
    df = ensure_min_rows(df, min_rows=10)
    st.session_state.pattern_vocab_df = df
    st.session_state.pattern_vocab_loaded_language = language

def get_current_pattern_vocab_df(language: str) -> pd.DataFrame:
    if (st.session_state.pattern_vocab_df is None
            or st.session_state.pattern_vocab_loaded_language != language):
        load_pattern_vocab_into_state(language)
    return st.session_state.pattern_vocab_df


def save_current_sentence_before_leaving():
    language = st.session_state.get("language")
    current_code = st.session_state.get("study_current_code")
    sentence_data = st.session_state.get("study_sentence", {})
    if not language or current_code in (None, ""):
        return
    if sentence_data and sentence_data.get("sentence"):
        set_cached_sentence(language, str(current_code), sentence_data)


# ══════════════════════════════════════════════════════════
# 首頁
# ══════════════════════════════════════════════════════════
def home_page():
    st.markdown("# 🧱 Bricklayer")

    languages = load_languages()
    # ── 語言按鈕（含國旗）──────────────────────────────────
    st.subheader("Select a language")
    for lang in languages:
        flag = LANGUAGE_FLAGS.get(lang["key"], "🌐")
        if st.button(
            f"{flag}  {lang['label']}",
            use_container_width=True,
            key=f"lang_btn_{lang['key']}"
        ):
            reset_vocab_state()
            reset_study_state()
            go_language(lang["key"])
            st.rerun()

    # ── 學習統計 ────────────────────────────────────────────
    if False and languages:
        st.divider()
        st.subheader("📊 學習進度")

        for lang in languages:
            key = lang["key"]
            flag = LANGUAGE_FLAGS.get(key, "🌐")
            progress_idx = get_language_progress(key)
            total = total_counts.get(key, 0)
            learned = min(progress_idx + 1, total) if total > 0 else 0
            pct = (learned / total * 100) if total > 0 else 0

            # 進度條
            stat_html = f"""
<div class="stat-row">
  <span class="stat-label">{flag} {lang['label']}</span>
  <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{pct:.1f}%"></div></div>
  <span class="stat-count">{learned}/{total}</span>
</div>"""
            st.markdown(stat_html, unsafe_allow_html=True)

            # 每日學習折線圖
            history = get_learning_history(key)
            if history:
                dates_sorted = sorted(history.keys())
                rows = []
                prev_idx = -1
                for d in dates_sorted:
                    idx = history[d]
                    new_words = max(0, idx - prev_idx)
                    rows.append({"日期": d, "每日新學字數": new_words})
                    prev_idx = idx
                trend_df = pd.DataFrame(rows).set_index("日期")
                st.line_chart(trend_df[["每日新學字數"]], height=120)
            else:
                st.caption("　　尚無學習記錄")

            st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("中文句子翻譯練習")
    if st.button("進入句子練習", use_container_width=True, key="home_translation_practice"):
        go_translation_practice(); st.rerun()

    # ── 資料備份 / 還原 ──────────────────────────────────────
    st.divider()
    st.subheader("💾 資料備份 / 還原")

    # ── GitHub 雙向同步 ──────────────────────────────────────
    gh_col1, gh_col2 = st.columns(2)
    with gh_col1:
        if st.button("☁️ 從 GitHub 同步資料到本機", use_container_width=True, key="home_gh_pull"):
            _langs = load_languages()
            _results = []
            for _l in _langs:
                _k = _l["key"]
                _ok_v = sync_vocab_from_github(_k)
                _ok_p = sync_pattern_vocab_from_github(_k)
                if _ok_v or _ok_p:
                    _results.append(f"{_k}: 詞庫✅" if _ok_v else f"{_k}: 句型詞庫✅")
                else:
                    _results.append(f"{_k}: GitHub 上無資料")
            _ok_prog = sync_progress_from_github()
            _ok_hist = sync_history_from_github()
            _ok_cache = sync_cache_from_github()
            _ok_trans = sync_translation_sentences_from_github()
            _extra = []
            if _ok_prog: _extra.append("學習進度✅")
            if _ok_hist: _extra.append("學習歷史✅")
            if _ok_cache: _extra.append("例句快取✅")
            if _ok_trans: _extra.append("中文句子✅")
            _all = _results + _extra
            if any("✅" in r for r in _all):
                st.success("同步完成！" + "　".join(_all) + "　⟶ 請重新整理頁面（F5）。")
            else:
                st.warning("GitHub bricklayer 分支上目前沒有資料檔案（首次使用屬正常）。")
    with gh_col2:
        st.caption("把 GitHub bricklayer 分支上的最新資料拉到本機，適合換電腦或重新安裝後使用。")

    st.caption("⬇️ 本機備份：匯出 ZIP 後可離線保存；匯入 ZIP 可還原。")

    exp_col, imp_col = st.columns(2)

    with exp_col:
        # 建立 ZIP（每次渲染時執行，資料量小速度快）
        _buf = io.BytesIO()
        with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
            if os.path.exists(DATA_FOLDER):
                for _fname in sorted(os.listdir(DATA_FOLDER)):
                    if _fname.endswith(".json"):
                        _zf.write(os.path.join(DATA_FOLDER, _fname), _fname)
        _buf.seek(0)
        st.download_button(
            label="📦 匯出全部資料",
            data=_buf,
            file_name="bricklayer_backup.zip",
            mime="application/zip",
            use_container_width=True,
            key="home_export_btn",
        )

    with imp_col:
        _uploaded = st.file_uploader(
            "匯入備份 ZIP", type=["zip"],
            key="home_import_zip",
            label_visibility="collapsed",
        )
        if _uploaded is not None:
            if st.button("✅ 確認還原資料", use_container_width=True, key="home_import_confirm"):
                try:
                    ensure_data_folder()
                    restored = []
                    restored_contents = {}  # name → content str，供後續 GitHub 同步用
                    with zipfile.ZipFile(io.BytesIO(_uploaded.read())) as _zf:
                        for _name in _zf.namelist():
                            # 只還原頂層 JSON，防止路徑穿越
                            if _name.endswith(".json") and "/" not in _name and "\\" not in _name:
                                _dest = os.path.join(DATA_FOLDER, _name)
                                _content_bytes = _zf.read(_name)
                                with open(_dest, "wb") as _dst:
                                    _dst.write(_content_bytes)
                                restored.append(_name)
                                restored_contents[_name] = _content_bytes.decode("utf-8")
                    if restored:
                        st.success(f"已還原 {len(restored)} 個檔案：{', '.join(restored)}  ⟶ 請重新整理頁面（F5）使資料生效。")
                        # ── 同步到 GitHub，確保 reboot 後資料不消失 ──
                        _token, _ = _github_config()
                        if _token:
                            _gh_ok_list = []
                            _gh_fail_list = []
                            for _name, _content_str in restored_contents.items():
                                _ok, _err = _github_write(
                                    f"data/{_name}", _content_str, None,
                                    f"Restore {_name} from backup"
                                )
                                if _ok:
                                    _gh_ok_list.append(_name)
                                else:
                                    _gh_fail_list.append(f"{_name}（{_err}）")
                            if _gh_fail_list:
                                st.warning(f"⚠️ 本機還原成功，但以下檔案 GitHub 同步失敗（reboot 後可能消失）：\n" + "\n".join(_gh_fail_list))
                            else:
                                st.info(f"☁️ 已同步 {len(_gh_ok_list)} 個檔案至 GitHub，重新啟動後資料不會消失。")
                        else:
                            st.warning("⚠️ 未設定 GITHUB_TOKEN，資料僅存於本機，reboot 後可能消失。")
                    else:
                        st.warning("ZIP 裡沒有找到 JSON 資料檔案。")
                except Exception as _e:
                    st.error(f"還原失敗：{_e}")

    # ── 設定 ────────────────────────────────────────────────
    st.divider()
    if st.button("⚙️ AI 設定", use_container_width=False, key="home_settings"):
        go_settings(); st.rerun()

    # ── 新增語言 ────────────────────────────────────────────
    st.divider()
    st.subheader("Add New Language")
    with st.form("add_language_form"):
        new_label = st.text_input("Language Name", placeholder="e.g. Korean")
        new_reading_label = st.text_input("Reading Label", value="Reading")
        new_supports_reading = st.checkbox("This language uses a reading column", value=False)
        submitted = st.form_submit_button("Add Language", use_container_width=True)
        if submitted:
            try:
                new_key = add_language(
                    label=new_label,
                    reading_label=new_reading_label,
                    supports_reading=new_supports_reading
                )
                st.success(f"Language added: {new_label}")
                reset_vocab_state()
                reset_study_state()
                go_language(new_key)
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ══════════════════════════════════════════════════════════
# 語言首頁
# ══════════════════════════════════════════════════════════
def language_home():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config = get_language_config(language)
    display_name = lang_config["label"] if lang_config else language.capitalize()
    flag = LANGUAGE_FLAGS.get(language, "🌐")

    st.title(f"{flag} {display_name}")

    # ── 詞彙學習區 ──────────────────────────────────────────
    st.markdown("##### 📚 字彙")
    if st.button("📖 字彙學習", use_container_width=True, key="lh_study"):
        go_study(); st.rerun()
    if st.button("🔄 字彙複習", use_container_width=True, key="lh_review"):
        go_review(); st.rerun()
    if st.button("✏️ 自訂字彙", use_container_width=True, key="lh_vocab"):
        go_custom_vocab(); st.rerun()

    # ── 句型練習區 ──────────────────────────────────────────
    st.markdown("##### 🗣️ 句型")
    if st.button("🗣️ 句型學習", use_container_width=True, key="lh_ps"):
        go_pattern_study(); st.rerun()
    if st.button("📝 句型複習", use_container_width=True, key="lh_pr"):
        go_pattern_review(); st.rerun()
    if st.button("📚 自訂句型", use_container_width=True, key="lh_pv"):
        go_pattern_vocab(); st.rerun()

    # ── 返回 ────────────────────────────────────────────────
    st.divider()
    if st.button("← 返回首頁", use_container_width=True, key="lh_back"):
        go_home(); st.rerun()


# ══════════════════════════════════════════════════════════
# 詞庫編輯頁
# ══════════════════════════════════════════════════════════
def custom_vocab_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    refresh_vocab_from_disk_if_task_finished(language)

    lang_config = get_language_config(language)
    display_name = lang_config["label"] if lang_config else language.capitalize()
    st.title(f"✏️ {display_name} — Custom Vocabulary")
    st.caption(
        "直接新增或編輯單字，離開欄位後即自動儲存到 GitHub。"
        "「#」欄位可以留空，系統會自動編號。「#」數字決定學習順序，"
        "也決定例句能使用哪些單字（例句只能用到當前編號以下的詞彙）。"
    )

    render_autocomplete_task_status(language)

    df = get_current_vocab_df(language)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"vocab_editor_{language}",
        column_config={
            "code":    st.column_config.TextColumn("#",       width="small",
                           help="學習順序編號。留空會自動填入。"),
            "term":    st.column_config.TextColumn("單字",    width="medium"),
            "reading": st.column_config.TextColumn("讀音",    width="medium"),
            "meaning": st.column_config.TextColumn("意思",    width="medium"),
            "pos":     st.column_config.TextColumn("詞性",    width="small"),
            "note":    st.column_config.TextColumn("備註",    width="large"),
        }
    )
    st.session_state.vocab_df = edited_df

    # ── 自動儲存（偵測內容變化即同步到 GitHub）─────────────
    _autosave_key = f"vocab_autosave_hash_{language}"
    _current_hash = str(pd.util.hash_pandas_object(edited_df, index=True).sum())
    _last_hash    = st.session_state.get(_autosave_key)

    if _last_hash is None:
        # 第一次載入：記錄基準 hash，不儲存
        st.session_state[_autosave_key] = _current_hash
    elif _current_hash != _last_hash:
        # 內容有變動 → 自動儲存
        try:
            save_vocab(language, edited_df)
            st.session_state[_autosave_key] = _current_hash
            # 更新 session state 中的詞庫（讓學習系統立即看到最新資料）
            st.session_state.vocab_df = edited_df
            st.toast("✅ 已自動儲存", icon="✅")
        except Exception as e:
            st.toast(f"❌ 自動儲存失敗：{e}", icon="❌")

    if st.button("AI 補全翻譯", use_container_width=True):
        try:
            save_vocab(language, st.session_state.vocab_df)
            task_id = start_autocomplete_task(language)
            st.session_state.autocomplete_task_id = task_id
            st.session_state.autocomplete_task_seen_done = False
            st.success("AI 補全已在背景開始執行。你可以先去別頁，完成後會自動存入詞庫。")
            st.rerun()
        except Exception as e:
            st.error(f"AI 補全啟動失敗：{e}")
    if st.button("💾 儲存詞庫", use_container_width=True):
        try:
            save_vocab(language, st.session_state.vocab_df)
            # 清除 editor delta，讓下次渲染從乾淨的存檔重新開始
            st.session_state.pop(f"vocab_editor_{language}", None)
            st.session_state.vocab_df = None
            st.session_state.vocab_loaded_language = None
            st.session_state.pop(_autosave_key, None)
            load_vocab_into_state(language)
            st.success("詞庫已儲存。")
            st.rerun()
        except Exception as e:
            st.error(f"儲存失敗：{e}")
    if st.button("🔄 重新載入詞庫", use_container_width=True):
        try:
            # 清除 editor delta，避免舊的編輯紀錄疊加到重新載入的資料上
            st.session_state.pop(f"vocab_editor_{language}", None)
            st.session_state.vocab_df = None
            st.session_state.vocab_loaded_language = None
            st.session_state.pop(_autosave_key, None)
            load_vocab_into_state(language)
            st.success("已重新載入最新詞庫。")
            st.rerun()
        except Exception as e:
            st.error(f"重新載入失敗：{e}")

    # ── 匯出 / 匯入 ────────────────────────────────────────
    st.divider()
    st.subheader("📦 備份與還原")
    st.caption("可以把詞庫匯出成 JSON 保存在本地，或從備份 JSON 匯入。")

    # 匯出：把目前詞庫轉成 JSON 下載
    export_df = get_current_vocab_df(language)
    export_df_clean = export_df[export_df["term"].astype(str).str.strip() != ""]
    import json as _json
    export_json = _json.dumps(export_df_clean.to_dict(orient="records"), ensure_ascii=False, indent=2)
    st.download_button(
        label=f"⬇️ 匯出詞庫 ({len(export_df_clean)} 筆)",
        data=export_json.encode("utf-8"),
        file_name=f"{language}_vocab_backup.json",
        mime="application/json",
        use_container_width=True
    )

    uploaded = st.file_uploader("⬆️ 匯入備份 JSON", type=["json"], key=f"import_{language}")
    if uploaded is not None:
        try:
            import json as _json2
            imported_data = _json2.loads(uploaded.read().decode("utf-8"))
            imported_df = pd.DataFrame(imported_data)
            save_vocab(language, imported_df)
            load_vocab_into_state(language)
            st.success(f"成功匯入 {len(imported_df)} 筆詞彙！")
            st.rerun()
        except Exception as e:
            st.error(f"匯入失敗：{e}")

    if st.button("← Back", use_container_width=True):
        st.session_state.page = "language_home"; st.rerun()


# ══════════════════════════════════════════════════════════
def _legacy_language_translation_practice_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config = get_language_config(language)
    display_name = lang_config["label"] if lang_config else language.capitalize()
    supports_reading = bool(lang_config.get("supports_reading")) if lang_config else False
    reading_label = lang_config.get("reading_label", "Reading") if lang_config else "Reading"

    st.title(f"中文翻成 {display_name}")
    st.caption("輸入中文句子後，AI 會翻成你目前選擇的學習語言；翻譯結果可以手動播放語音。")

    with st.form("translation_practice_form"):
        source = st.text_area(
            "中文句子",
            value=st.session_state.get("translation_input", ""),
            height=110,
            placeholder="例如：我今天想去咖啡廳讀書。",
        )
        submitted = st.form_submit_button("翻譯", use_container_width=True)

    if submitted:
        source = source.strip()
        st.session_state.translation_input = source
        if not source:
            st.warning("請先輸入中文句子。")
        else:
            try:
                with st.spinner("AI 正在翻譯..."):
                    result = translate_chinese_sentence(language, display_name, source)
                st.session_state.translation_result = result
                st.session_state.translation_source = source
                st.session_state.translation_language = language
                st.session_state.translation_tts_audio = None
                st.session_state.translation_tts_for = ""
            except Exception as e:
                st.error(f"翻譯失敗：{e}")

    result = st.session_state.get("translation_result", {})
    sentence = str(result.get("sentence", "") or "").strip()
    if sentence and st.session_state.get("translation_language") == language:
        st.divider()
        st.markdown("#### 翻譯結果")
        st.markdown(
            f'<div class="study-card"><div class="study-value-lg">{sentence}</div></div>',
            unsafe_allow_html=True,
        )

        reading = str(result.get("reading", "") or "").strip()
        if supports_reading and reading:
            st.markdown(f"**{reading_label}：** {reading}")

        note = str(result.get("note", "") or "").strip()
        if note:
            st.info(note)

        play_key = f"{language}::{sentence}"
        cached_audio = get_cached_tts(language, "translation", "sent", sentence)
        if cached_audio:
            st.session_state.translation_tts_audio = cached_audio
            st.session_state.translation_tts_for = play_key

        if st.button("播放翻譯語音", use_container_width=True, key="translation_play"):
            try:
                audio_bytes = cached_audio or generate_tts_audio(sentence, language)
                set_cached_tts(language, "translation", "sent", audio_bytes, sentence)
                st.session_state.translation_tts_audio = audio_bytes
                st.session_state.translation_tts_for = play_key
            except Exception as e:
                st.error(f"語音產生失敗：{e}")

        audio_bytes = st.session_state.get("translation_tts_audio")
        if audio_bytes and st.session_state.get("translation_tts_for") == play_key:
            components.html(audio_player_pausable(audio_bytes), height=70)

    st.divider()
    if st.button("← 回到語言首頁", use_container_width=True, key="translation_back"):
        st.session_state.page = "language_home"; st.rerun()


# 學習頁面（左：詞彙  右：例句＋文法）
# ══════════════════════════════════════════════════════════
def study_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config = get_language_config(language)
    display_name = lang_config["label"] if lang_config else language.capitalize()
    reading_label = lang_config["reading_label"] if lang_config else "Reading"
    supports_reading = lang_config["supports_reading"] if lang_config else False

    raw_df = get_current_vocab_df(language)
    study_df = prepare_study_df(raw_df)

    if study_df.empty:
        st.warning("目前詞庫是空的。請先到 Custom Vocabulary 新增詞彙並儲存。")
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "language_home"; st.rerun()
        return

    saved_index = get_language_progress(language)
    if st.session_state.study_index != saved_index and st.session_state.study_index == 0:
        st.session_state.study_index = min(saved_index, len(study_df) - 1)
    if st.session_state.study_index >= len(study_df):
        st.session_state.study_index = len(study_df) - 1
    # 自動保存目前位置（避免 Reboot 後回到舊進度）
    if st.session_state.study_index > saved_index:
        update_language_progress(language, st.session_state.study_index)

    current = get_current_row(study_df, st.session_state.study_index)
    current_code = int(current["code_num"])
    current_term = current["term"]
    st.session_state.study_current_code = str(current["code"])

    full_vocab_list = _df_to_allowed_vocab(study_df)
    allowed_df = get_allowed_vocab(study_df, current_code)
    allowed_terms = allowed_df["term"].astype(str).tolist()

    # ── 批次預生成例句（每 10 個單字一批，進入新批次時一次生成完畢）──
    _current_batch = st.session_state.get("study_batch_generated", -1)
    if _current_batch != st.session_state.get("study_batch_generated", -1):
        _batch_start = _current_batch * 10
        _batch_end   = min(_batch_start + 10, len(study_df))
        _missing = [
            i for i in range(_batch_start, _batch_end)
            if not get_cached_sentence(language, str(get_current_row(study_df, i)["code"])).get("sentence")
        ]
        if _missing:
            _batch_results = {}
            with st.spinner(f"AI 正在預先生成第 {_batch_start + 1}–{_batch_end} 個例句（共 {len(_missing)} 個）…"):
                for _bi in _missing:
                    _row   = get_current_row(study_df, _bi)
                    _bcode = str(_row["code"])
                    _bcode_num = int(_row["code_num"])
                    _b_allowed_df = get_allowed_vocab(study_df, _bcode_num)
                    try:
                        _bresult = generate_example_sentence(
                            language=language,
                            current_term=str(_row["term"]),
                            allowed_terms=_b_allowed_df["term"].astype(str).tolist(),
                            term_meaning=str(_row.get("meaning", "")),
                            term_reading=str(_row.get("reading", "")),
                            term_pos=str(_row.get("pos", "")),
                            current_code=_bcode_num,
                            allowed_vocab=_df_to_allowed_vocab(_b_allowed_df),
                            full_vocab=full_vocab_list,
                        )
                        _batch_results[_bcode] = _bresult
                    except Exception:
                        pass
            # 一次性存入快取（1 次 GitHub write，而非 N 次）
            set_cached_sentences_bulk(language, _batch_results)
        st.session_state.study_batch_generated = _current_batch

    cached_sentence = get_cached_sentence(language, str(current["code"]))
    needs_switch = st.session_state.study_sentence_term != current_term

    # 決定是否需要 AI 生成：有快取直接套用；無快取則延遲到右欄渲染後再生成
    _gen_needed = False
    if needs_switch:
        if cached_sentence.get("sentence") and _is_cache_sentence_valid(
            sentence_data=cached_sentence,
            vocab_df=study_df,
            current_code=current_code,
            language=language,
        ):
            # 有快取 → 立即套用，本次渲染就能顯示
            st.session_state.study_sentence = cached_sentence
            st.session_state.study_sentence_term = current_term
            st.session_state.study_current_code = str(current["code"])
        else:
            # 無快取 → 先清空句子、標記待生成，讓詞彙卡先渲染出來
            st.session_state.study_sentence = {}
            st.session_state.study_sentence_term = current_term
            st.session_state.study_current_code = str(current["code"])
            _gen_needed = True
    elif st.session_state.study_sentence.get("sentence") and not _is_cache_sentence_valid(
        sentence_data=st.session_state.study_sentence,
        vocab_df=study_df,
        current_code=current_code,
        language=language,
    ):
        st.session_state.study_sentence = {}
        _gen_needed = True

    # _code_str 在整個頁面（TTS 按鈕）都會用到，保留在此
    _code_str = str(current["code"])

    # ── 頂部導航列（第一個 UI 元素，點按後立即渲染）──────────────────
    _pct = st.session_state.study_index / max(len(study_df) - 1, 1)
    _tnl, _tnm, _tnr = st.columns([2, 5, 2])
    with _tnl:
        if st.button("⬅ 上一個", key="study_prev_top", use_container_width=True):
            save_current_sentence_before_leaving()
            st.session_state.study_index = get_prev_index(study_df, st.session_state.study_index)
            update_language_progress(language, st.session_state.study_index)
            st.session_state.study_sentence_term = ""
            st.rerun()
    with _tnm:
        st.progress(_pct)
        st.caption(f"📚 {display_name}　{st.session_state.study_index + 1} / {len(study_df)}　｜　Available: {len(allowed_terms)}")
        st.caption(f"Build: {_build_hash()}")
        _sync_status = get_progress_sync_status()
        if _sync_status.get("ok") is False:
            st.caption(f"⚠️ 進度雲端同步失敗：{_sync_status.get('message', '')}")
        elif _sync_status.get("ok") is True:
            st.caption("✅ 進度已同步到 GitHub")
    with _tnr:
        if st.button("下一個 ➡", key="study_next_top", use_container_width=True):
            save_current_sentence_before_leaving()
            st.session_state.study_index = get_next_index(study_df, st.session_state.study_index)
            update_language_progress(language, st.session_state.study_index)
            st.session_state.study_sentence_term = ""
            st.rerun()

    # ══ 左右分欄（電腦左右、手機上下）══════════════════════
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── 左欄：詞彙資訊 ────────────────────────────────────
    with col_left:
        st.markdown('<div class="study-card">', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Code</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-md">{current["code"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Term</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-lg">{current["term"]}</div>', unsafe_allow_html=True)

        if supports_reading:
            st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{current.get("reading", "")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Meaning</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-md">{current.get("meaning", "")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Part of Speech</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-md">{current.get("pos", "")}</div>', unsafe_allow_html=True)

        if current.get("note", "").strip():
            st.markdown('<div class="study-label">Note</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{current.get("note", "")}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # TTS 發音
        if st.session_state.tts_term_for != current_term:
            st.session_state.tts_term_audio = None
            st.session_state.tts_term_for = ""

        if st.button("🔊 播放發音", key="tts_term_btn", use_container_width=True):
            try:
                audio_bytes = (
                    (st.session_state.tts_term_audio if st.session_state.tts_term_for == current_term else None)
                    or get_cached_tts(language, _code_str, "term")
                )
                if not audio_bytes:
                    with st.spinner("生成發音中..."):
                        audio_bytes = generate_tts_audio(current_term, language)
                        set_cached_tts(language, _code_str, "term", audio_bytes)
                st.session_state.tts_term_audio = audio_bytes
                st.session_state.tts_term_for = current_term
                components.html(audio_player(audio_bytes), height=64)
            except Exception as e:
                st.error(f"TTS 失敗：{e}")

        # ── 熟悉度標記 ────────────────────────────────────
        _fam = get_familiarity(language, current_code)
        _fam_labels = {FAMILIAR: "✅ 熟悉", UNFAMILIAR: "❗ 陌生", None: ""}
        if _fam:
            st.caption(f"目前標記：{_fam_labels[_fam]}")
        fam_c1, fam_c2 = st.columns(2)
        with fam_c1:
            _btn_fam = "✅ 熟悉" if _fam != FAMILIAR else "✅ 熟悉（已標）"
            if st.button(_btn_fam, key="study_fam_familiar", use_container_width=True):
                set_familiarity(language, current_code, None if _fam == FAMILIAR else FAMILIAR)
                st.rerun()
        with fam_c2:
            _btn_unfam = "❗ 陌生" if _fam != UNFAMILIAR else "❗ 陌生（已標）"
            if st.button(_btn_unfam, key="study_fam_unfamiliar", use_container_width=True):
                set_familiarity(language, current_code, None if _fam == UNFAMILIAR else UNFAMILIAR)
                st.rerun()

    # ── 右欄：例句＋文法 ──────────────────────────────────
    with col_right:
        sentence_data = st.session_state.study_sentence

        st.markdown('<div class="study-card">', unsafe_allow_html=True)
        st.markdown('<div class="study-label">Example Sentence</div>', unsafe_allow_html=True)

        if _gen_needed:
            # 詞彙卡已渲染，在右欄生成 AI 例句（用戶可看到左欄詞彙）
            st.markdown('<div class="study-value-md" style="color:#aaa;">正在生成例句…</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            _generated_ok = False
            with st.spinner("AI 正在生成例句…"):
                try:
                    result = generate_example_sentence(
                        language=language,
                        current_term=current_term,
                        allowed_terms=allowed_terms,
                        term_meaning=str(current.get("meaning", "")),
                        term_reading=str(current.get("reading", "")),
                        term_pos=str(current.get("pos", "")),
                        current_code=current_code,
                        allowed_vocab=_df_to_allowed_vocab(allowed_df),
                        full_vocab=full_vocab_list,
                    )
                    st.session_state.study_sentence = result
                    set_cached_sentence(language, str(current["code"]), result)
                    _generated_ok = True
                except Exception as e:
                    log_error(f"[study_page] auto generate failed lang={language} code={current.get('code','')}: {e}")
                    st.error(f"自動生成例句失敗：{e}")
            if _generated_ok:
                st.rerun()
        elif sentence_data.get("sentence"):
            st.markdown(f'<div class="study-value-md">{sentence_data["sentence"]}</div>', unsafe_allow_html=True)

            if supports_reading and sentence_data.get("reading"):
                st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{sentence_data["reading"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="study-label">Translation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{sentence_data.get("translation", "")}</div>', unsafe_allow_html=True)

            if sentence_data.get("grammar"):
                st.markdown('<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                _render_grammar_box(sentence_data["grammar"])

            render_used_vocab(sentence_data["sentence"], study_df, current_code, vocab_codes=sentence_data.get("vocab_codes"))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="study-value-md" style="color:#aaa;">正在生成例句…</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 新例句 + TTS 例句
        current_sentence = sentence_data.get("sentence", "")
        if st.session_state.tts_sentence_for != current_sentence:
            st.session_state.tts_sentence_audio = None
            st.session_state.tts_sentence_for = ""

        if st.button("🔀 新例句", key="new_sentence_btn", use_container_width=True):
            try:
                with st.spinner("AI 正在生成新例句..."):
                    result = generate_example_sentence(
                        language=language,
                        current_term=current_term,
                        allowed_terms=allowed_terms,
                        term_meaning=str(current.get("meaning", "")),
                        term_reading=str(current.get("reading", "")),
                        term_pos=str(current.get("pos", "")),
                        current_code=current_code,
                        allowed_vocab=_df_to_allowed_vocab(allowed_df),
                        full_vocab=full_vocab_list,
                    )
                    st.session_state.study_sentence = result
                    st.session_state.study_sentence_term = current_term
                    st.session_state.study_current_code = str(current["code"])
                    set_cached_sentence(language, str(current["code"]), result)
                st.rerun()
            except Exception as e:
                log_error(f"[study_page] new sentence failed lang={language} code={current.get('code','')}: {e}")
                st.error(f"生成例句失敗：{e}")

        if st.button("🔊 播放例句", key="tts_sentence_btn", use_container_width=True):
            try:
                if st.session_state.tts_sentence_audio and st.session_state.tts_sentence_for == current_sentence:
                    audio_bytes = st.session_state.tts_sentence_audio
                else:
                    _tts_s = _sent_tts_text(sentence_data, language)
                    with st.spinner("生成例句發音中..."):
                        audio_bytes = generate_tts_audio(_tts_s, language)
                        st.session_state.tts_sentence_audio = audio_bytes
                        st.session_state.tts_sentence_for = current_sentence
                components.html(audio_player(audio_bytes), height=64)
            except Exception as e:
                st.error(f"TTS 失敗：{e}")

    # ── 底部導航 ───────────────────────────────────────────
    st.divider()
    _bnl, _bnm, _bnr = st.columns([2, 3, 2])
    with _bnl:
        if st.button("⬅ 上一個", key="study_prev_bot", use_container_width=True):
            save_current_sentence_before_leaving()
            st.session_state.study_index = get_prev_index(study_df, st.session_state.study_index)
            update_language_progress(language, st.session_state.study_index)
            st.session_state.study_sentence_term = ""
            st.rerun()
    with _bnm:
        if st.button("↩ 回到語言首頁", key="study_home_bot", use_container_width=True):
            save_current_sentence_before_leaving()
            update_language_progress(language, st.session_state.study_index)
            st.session_state.page = "language_home"
            st.rerun()
    with _bnr:
        if st.button("下一個 ➡", key="study_next_bot", use_container_width=True):
            save_current_sentence_before_leaving()
            st.session_state.study_index = get_next_index(study_df, st.session_state.study_index)
            update_language_progress(language, st.session_state.study_index)
            st.session_state.study_sentence_term = ""
            st.rerun()

    # ── 跳號（移到底部）──────────────────────────────────────────
    jc1, jc2 = st.columns([4, 1])
    with jc1:
        jump_val = st.number_input("跳到編號", min_value=0, value=0, step=1,
                                   key="study_jump_input", label_visibility="collapsed")
    with jc2:
        if st.button("跳至", key="study_jump_btn", use_container_width=True):
            if jump_val > 0:
                matches = study_df[study_df["code_num"] == int(jump_val)]
                if not matches.empty:
                    st.session_state.study_index = int(matches.index[0])
                    update_language_progress(language, st.session_state.study_index)
                    st.session_state.study_sentence_term = ""
                    st.rerun()
                else:
                    st.warning(f"找不到編號 {int(jump_val)}")

    # ── 自動播放（放在頁面最後，確保詞彙卡和例句已先渲染完畢）──────
    # 頁面元素全部送出後才執行 TTS，用戶已能看到內容，不會空白等待
    _sentence_data_auto = st.session_state.study_sentence
    _cur_sent_auto = _sentence_data_auto.get("sentence", "")
    if _cur_sent_auto and not _gen_needed:
        _next_index = get_next_index(study_df, st.session_state.study_index)
        if _next_index != st.session_state.study_index:
            _next_row = get_current_row(study_df, _next_index)
            _next_code = str(_next_row["code"])
            if not get_cached_sentence(language, _next_code).get("sentence"):
                _next_code_num = int(_next_row["code_num"])
                _next_allowed_df = get_allowed_vocab(study_df, _next_code_num)
                start_sentence_prefetch_task(
                    language=language,
                    code=_next_code,
                    current_term=str(_next_row["term"]),
                    allowed_terms=_next_allowed_df["term"].astype(str).tolist(),
                    term_meaning=str(_next_row.get("meaning", "")),
                    term_reading=str(_next_row.get("reading", "")),
                    term_pos=str(_next_row.get("pos", "")),
                    current_code=_next_code_num,
                    allowed_vocab=_df_to_allowed_vocab(_next_allowed_df),
                    full_vocab=full_vocab_list,
                    ai_provider=st.session_state.get("ai_provider", "openai"),
                    ai_model=st.session_state.get("ai_model", ""),
                )
    if _cur_sent_auto and not _gen_needed and st.session_state.auto_played_for != _cur_sent_auto:
        try:
            # 自動播放只使用既有快取，避免頁面為了 TTS 再等待 API
            _term_audio = (
                get_cached_tts(language, _code_str, "term")
                or (st.session_state.tts_term_for == current_term and st.session_state.tts_term_audio)
            )
            _sent_tts_val = _sent_tts_text(_sentence_data_auto, language)
            _sent_audio = (
                get_cached_tts(language, _code_str, "sent", _sent_tts_val)
                or (st.session_state.tts_sentence_for == _cur_sent_auto and st.session_state.tts_sentence_audio)
            )
            if _term_audio and _sent_audio:
                st.session_state.tts_term_audio = _term_audio
                st.session_state.tts_term_for = current_term
                st.session_state.tts_sentence_audio = _sent_audio
                st.session_state.tts_sentence_for = _cur_sent_auto
                components.html(audio_player_dual(_term_audio, _sent_audio), height=0)
                st.session_state.auto_played_for = _cur_sent_auto
        except Exception as e:
            st.warning(f"自動播放失敗：{e}")


# ══════════════════════════════════════════════════════════
# 複習頁面（單字複習 ＋ 重組練習 兩個模式）
# ══════════════════════════════════════════════════════════
def review_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config   = get_language_config(language)
    display_name  = lang_config["label"]        if lang_config else language.capitalize()
    reading_label = lang_config["reading_label"] if lang_config else "Reading"
    supports_reading = lang_config["supports_reading"] if lang_config else False
    flag = LANGUAGE_FLAGS.get(language, "🌐")

    st.title(f"{flag} {display_name} — 複習")

    raw_df   = get_current_vocab_df(language)
    study_df = prepare_study_df(raw_df)
    review_full_vocab_list = _df_to_allowed_vocab(study_df)

    if study_df.empty:
        st.warning("詞庫是空的，請先新增詞彙。")
        if st.button("← Back"): st.session_state.page = "language_home"; st.rerun()
        return

    progress_idx = get_language_progress(language)
    learned_df   = study_df.iloc[:progress_idx + 1]

    if learned_df.empty:
        st.info("還沒有學習進度，請先到 Study Vocabulary 學習至少一個單字。")
        if st.button("← Back"): st.session_state.page = "language_home"; st.rerun()
        return

    # ── 練習範圍設定 ──────────────────────────────────────
    all_min = int(learned_df["code_num"].min())
    all_max = int(learned_df["code_num"].max())

    # 初始化：首次進入或超出已學範圍時重設
    if st.session_state.review_code_min is None or st.session_state.review_code_min < all_min:
        st.session_state.review_code_min = all_min
    if st.session_state.review_code_max is None or st.session_state.review_code_max > all_max:
        st.session_state.review_code_max = all_max

    with st.expander("🎯 練習範圍設定（預設：全部已學）", expanded=False):
        st.caption(f"已學範圍：{all_min} ～ {all_max}　共 {len(learned_df)} 個單字")
        rc1, rc2 = st.columns(2)
        with rc1:
            new_min = st.number_input(
                "從編號", min_value=all_min, max_value=all_max,
                value=st.session_state.review_code_min,
                step=1, key="review_range_min"
            )
        with rc2:
            new_max = st.number_input(
                "到編號", min_value=all_min, max_value=all_max,
                value=st.session_state.review_code_max,
                step=1, key="review_range_max"
            )
        if new_min > new_max:
            st.warning("「從編號」不能大於「到編號」，已自動對調。")
            new_min, new_max = new_max, new_min

        # 範圍改變時清除當前題目，觸發重抽
        if new_min != st.session_state.review_code_min or new_max != st.session_state.review_code_max:
            st.session_state.review_code_min = new_min
            st.session_state.review_code_max = new_max
            st.session_state.review_term = ""
            st.session_state.combo_words = []
            st.rerun()

    range_min = st.session_state.review_code_min
    range_max = st.session_state.review_code_max

    # 過濾 learned_df 到指定範圍
    learned_df = learned_df[
        (learned_df["code_num"] >= range_min) & (learned_df["code_num"] <= range_max)
    ]
    if learned_df.empty:
        st.warning(f"編號 {range_min}～{range_max} 範圍內沒有已學的單字，請調整範圍。")
        return

    # 若當前題目已不在範圍內，清除讓系統重抽
    if st.session_state.review_term:
        cur_code = st.session_state.review_term_code
        if cur_code < range_min or cur_code > range_max:
            st.session_state.review_term = ""
            st.session_state.combo_words = []

    # ── 模式切換 ──────────────────────────────────────────
    mode = st.radio(
        "複習模式",
        ["📖 單字複習", "🔀 重組練習", "📝 短文練習"],
        horizontal=True,
        key="review_mode_radio"
    )
    st.divider()

    # ════════════════════════════════════════════════════
    # 模式 1：單字複習 — FSI Substitution + Transformation
    # ════════════════════════════════════════════════════
    if mode == "📖 單字複習":
        st.caption("隨機抽一個已學單字，透過 FSI 練習法觀察搭配詞（Substitution）與句型變化（Transformation）。")

        # ── 抽題 ──────────────────────────────────────
        if st.button("🎲 抽新題目", use_container_width=True, key="word_draw"):
            _weights = get_sample_weights(language, learned_df["code_num"].tolist())
            picked       = learned_df.sample(1, weights=_weights).iloc[0]
            pick_code    = int(picked["code_num"])
            pick_term    = str(picked["term"])
            pick_meaning = str(picked.get("meaning", ""))
            pick_reading = str(picked.get("reading", ""))
            pick_pos     = str(picked.get("pos", ""))
            allowed_df   = get_allowed_vocab(study_df, pick_code)
            allowed_vocab_list = _df_to_allowed_vocab(allowed_df)

            fsi_kwargs = dict(
                language=language,
                current_term=pick_term,
                term_meaning=pick_meaning,
                term_reading=pick_reading,
                term_pos=pick_pos,
                current_code=pick_code,
                allowed_vocab=allowed_vocab_list,
                full_vocab=review_full_vocab_list,
            )
            try:
                with st.spinner("AI 正在生成搭配練習句..."):
                    sub_data = generate_fsi_sentence(drill_type="substitution", **fsi_kwargs)
            except Exception as e:
                st.error(f"Substitution 生成失敗：{e}")
                sub_data = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}

            try:
                with st.spinner("AI 正在生成句型變化句..."):
                    trans_data = generate_fsi_sentence(drill_type="transformation", **fsi_kwargs)
            except Exception as e:
                st.error(f"Transformation 生成失敗：{e}")
                trans_data = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}

            st.session_state.review_term          = pick_term
            st.session_state.review_meaning       = pick_meaning
            st.session_state.review_term_code     = pick_code
            st.session_state.review_term_reading  = pick_reading
            st.session_state.review_term_pos      = pick_pos
            st.session_state.review_sub_sentence  = sub_data
            st.session_state.review_trans_sentence= trans_data
            st.session_state.review_sub_prev_notes  = [sub_data["drill_note"]] if sub_data.get("drill_note") else []
            st.session_state.review_trans_prev_notes= [trans_data["drill_note"]] if trans_data.get("drill_note") else []
            st.rerun()

        if not st.session_state.review_term:
            st.info("按「抽新題目」後才會產生 Substitution 和 Transformation 練習，避免一進頁面就等待 AI 生成。")
            return

        # ── 左右分欄 ──────────────────────────────────
        rev_left, rev_right = st.columns([1, 1], gap="large")

        # ── 左欄：單字資訊卡 ──────────────────────────
        with rev_left:
            pick_code = st.session_state.review_term_code
            pick_term = st.session_state.review_term

            st.markdown('<div class="study-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="study-label">Code</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{pick_code}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-label">單字</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-lg">{pick_term}</div>', unsafe_allow_html=True)
            if supports_reading and st.session_state.review_term_reading:
                st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{st.session_state.review_term_reading}</div>', unsafe_allow_html=True)
            if st.session_state.review_meaning:
                st.markdown(f'<div class="study-label">意思</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md" style="color:#4F8BF9;">{st.session_state.review_meaning}</div>', unsafe_allow_html=True)
            if st.session_state.review_term_pos:
                st.markdown(f'<div class="study-label">詞性</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{st.session_state.review_term_pos}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔊 播放發音", key="review_term_tts", use_container_width=True):
                try:
                    with st.spinner("生成發音中..."):
                        audio_bytes = generate_tts_audio(pick_term, language)
                    components.html(audio_player(audio_bytes), height=64)
                except Exception as e:
                    st.error(f"TTS 失敗：{e}")

            # ── 熟悉度標記 ────────────────────────────
            _fam_r = get_familiarity(language, pick_code)
            _fam_labels_r = {FAMILIAR: "✅ 熟悉", UNFAMILIAR: "❗ 陌生"}
            if _fam_r:
                st.caption(f"目前標記：{_fam_labels_r[_fam_r]}")
            rfam_c1, rfam_c2 = st.columns(2)
            with rfam_c1:
                _rb_fam = "✅ 熟悉（已標）" if _fam_r == FAMILIAR else "✅ 熟悉"
                if st.button(_rb_fam, key="rev_fam_familiar", use_container_width=True):
                    set_familiarity(language, pick_code, None if _fam_r == FAMILIAR else FAMILIAR)
                    st.rerun()
            with rfam_c2:
                _rb_unfam = "❗ 陌生（已標）" if _fam_r == UNFAMILIAR else "❗ 陌生"
                if st.button(_rb_unfam, key="rev_fam_unfamiliar", use_container_width=True):
                    set_familiarity(language, pick_code, None if _fam_r == UNFAMILIAR else UNFAMILIAR)
                    st.rerun()

        # ── 右欄：FSI 練習區 ──────────────────────────
        with rev_right:
            allowed_df = get_allowed_vocab(study_df, st.session_state.review_term_code)
            allowed_vocab_list = _df_to_allowed_vocab(allowed_df)
            fsi_base = dict(
                language=language,
                current_term=st.session_state.review_term,
                term_meaning=st.session_state.review_meaning,
                term_reading=st.session_state.review_term_reading,
                term_pos=st.session_state.review_term_pos,
                current_code=st.session_state.review_term_code,
                allowed_vocab=allowed_vocab_list,
                full_vocab=review_full_vocab_list,
            )

            # ===== Substitution 搭配練習 =====
            st.markdown("#### 🔄 Substitution — 搭配練習")
            st.caption("觀察目標單字與不同搭配詞或情境的組合")

            sub_data = st.session_state.review_sub_sentence
            if sub_data.get("sentence"):
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                if sub_data.get("drill_note"):
                    st.markdown(f'<div class="study-label">搭配說明</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md" style="color:#e65100;font-weight:600;">{sub_data["drill_note"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{sub_data["sentence"]}</div>', unsafe_allow_html=True)
                if supports_reading and sub_data.get("reading"):
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md">{sub_data["reading"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{sub_data.get("translation","")}</div>', unsafe_allow_html=True)
                if sub_data.get("grammar"):
                    st.markdown(f'<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                    _render_grammar_box(sub_data["grammar"])
                render_used_vocab(sub_data["sentence"], study_df, st.session_state.review_term_code, vocab_codes=sub_data.get("vocab_codes"))
                st.markdown('</div>', unsafe_allow_html=True)

            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                if st.button("🔀 下一個", key="sub_next_btn", use_container_width=True):
                    prev = st.session_state.get("review_sub_prev_notes", [])
                    try:
                        with st.spinner("生成下一個搭配..."):
                            new_sub = generate_fsi_sentence(drill_type="substitution", prev_drill_notes=prev, **fsi_base)
                        st.session_state.review_sub_sentence = new_sub
                        if new_sub.get("drill_note"):
                            st.session_state.review_sub_prev_notes = (prev + [new_sub["drill_note"]])[-6:]
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成失敗：{e}")
            with sub_c2:
                if st.button("🔊 發音", key="sub_tts_btn", use_container_width=True):
                    if sub_data.get("sentence"):
                        try:
                            with st.spinner("生成發音中..."):
                                audio_bytes = generate_tts_audio(_sent_tts_text(sub_data, language), language)
                            components.html(audio_player(audio_bytes), height=64)
                        except Exception as e:
                            st.error(f"TTS 失敗：{e}")

            st.divider()

            # ===== Transformation 句型變化 =====
            st.markdown("#### ⚙️ Transformation — 句型變化")
            st.caption("觀察目標單字在不同文法形態下的變化")

            trans_data = st.session_state.review_trans_sentence
            if trans_data.get("sentence"):
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                if trans_data.get("drill_note"):
                    st.markdown(f'<div class="study-label">文法形態</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md" style="color:#1565c0;font-weight:600;">{trans_data["drill_note"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{trans_data["sentence"]}</div>', unsafe_allow_html=True)
                if supports_reading and trans_data.get("reading"):
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md">{trans_data["reading"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{trans_data.get("translation","")}</div>', unsafe_allow_html=True)
                if trans_data.get("grammar"):
                    st.markdown(f'<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                    _render_grammar_box(trans_data["grammar"])
                render_used_vocab(trans_data["sentence"], study_df, st.session_state.review_term_code, vocab_codes=trans_data.get("vocab_codes"))
                st.markdown('</div>', unsafe_allow_html=True)

            trans_c1, trans_c2 = st.columns(2)
            with trans_c1:
                if st.button("🔀 下一個", key="trans_next_btn", use_container_width=True):
                    prev = st.session_state.get("review_trans_prev_notes", [])
                    try:
                        with st.spinner("生成下一個變化..."):
                            new_trans = generate_fsi_sentence(drill_type="transformation", prev_drill_notes=prev, **fsi_base)
                        st.session_state.review_trans_sentence = new_trans
                        if new_trans.get("drill_note"):
                            st.session_state.review_trans_prev_notes = (prev + [new_trans["drill_note"]])[-6:]
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成失敗：{e}")
            with trans_c2:
                if st.button("🔊 發音", key="trans_tts_btn", use_container_width=True):
                    if trans_data.get("sentence"):
                        try:
                            with st.spinner("生成發音中..."):
                                audio_bytes = generate_tts_audio(_sent_tts_text(trans_data, language), language)
                            components.html(audio_player(audio_bytes), height=64)
                        except Exception as e:
                            st.error(f"TTS 失敗：{e}")

    # ════════════════════════════════════════════════════
    # 模式 2：重組練習（新功能）
    # ════════════════════════════════════════════════════
    elif mode == "🔀 重組練習":
        n_learned = len(learned_df)
        n_pick = 2  # 固定抽 2 個詞

        if n_learned < 2:
            st.info("需要至少學 2 個單字才能使用重組練習。")
        else:
            st.caption(f"隨機抽 2 個已學過的單字，AI 會把它們組成一句短句。按「看答案」可查看文法分析與使用單字。")

            if st.button("🎲 抽新組合", use_container_width=True, key="combo_draw") or not st.session_state.combo_words:
                _combo_weights = get_sample_weights(language, learned_df["code_num"].tolist())
                picked_rows = learned_df.sample(n=n_pick, weights=_combo_weights)
                all_allowed = learned_df["term"].astype(str).tolist()

                target_words = [
                    {
                        "term":    str(row["term"]),
                        "meaning": str(row.get("meaning", "")),
                        "reading": str(row.get("reading", ""))
                    }
                    for _, row in picked_rows.iterrows()
                ]

                # 隨機挑一個文法句型
                patterns = GRAMMAR_PATTERNS.get(language, GRAMMAR_PATTERNS["default"])
                pattern  = random.choice(patterns)

                try:
                    with st.spinner("AI 正在重組新句子..."):
                        combo_sentence = generate_recombination_sentence(
                            language=language,
                            target_words=target_words,
                            all_allowed_terms=all_allowed,
                            grammar_instruction=pattern["instruction"],
                            all_allowed_vocab=_df_to_allowed_vocab(learned_df),
                        )
                except Exception as e:
                    st.error(f"生成例句失敗：{e}")
                    combo_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}

                st.session_state.combo_words       = target_words
                st.session_state.combo_sentence    = combo_sentence
                st.session_state.combo_show_answer = False
                st.session_state.combo_pattern     = pattern
                st.rerun()

            # 顯示本題文法句型標籤
            pattern_label = st.session_state.combo_pattern.get("label", "")
            if pattern_label:
                st.markdown(
                    f'<div style="display:inline-block;background:#eef4fb;border:1.5px solid #4F8BF9;'
                    f'border-radius:20px;padding:0.25rem 0.9rem;font-size:0.95rem;'
                    f'color:#2c5db3;margin-bottom:0.8rem;">🎯 本題文法：{pattern_label}</div>',
                    unsafe_allow_html=True
                )

            combo_data = st.session_state.combo_sentence
            if combo_data.get("sentence"):
                # 自動播放例句（每題只播一次）
                if st.session_state.combo_auto_played_for != combo_data["sentence"]:
                    try:
                        with st.spinner("自動播放中..."):
                            audio_bytes = generate_tts_audio(_sent_tts_text(combo_data, language), language)
                        components.html(audio_player(audio_bytes), height=64)
                        st.session_state.combo_auto_played_for = combo_data["sentence"]
                    except Exception as e:
                        st.warning(f"自動播放失敗：{e}")

                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                st.markdown('<div class="study-label">重組例句</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{combo_data["sentence"]}</div>', unsafe_allow_html=True)

                if supports_reading and combo_data.get("reading"):
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md">{combo_data["reading"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{combo_data.get("translation", "")}</div>', unsafe_allow_html=True)

                if combo_data.get("grammar"):
                    st.markdown('<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                    _render_grammar_box(combo_data["grammar"])

                # 顯示本題使用的所有目標單字
                st.markdown('<div class="study-label">本題使用的單字</div>', unsafe_allow_html=True)
                words_html = ""
                for w in st.session_state.combo_words:
                    reading_part = f"（{w['reading']}）" if w.get("reading") else ""
                    words_html += (
                        f'<div style="margin-bottom:0.5rem;">'
                        f'<span style="font-size:1.2rem;font-weight:700;">{w["term"]}</span>'
                        f'<span style="font-size:0.95rem;color:#666;margin-left:0.4rem;">{reading_part}</span>'
                        f'<span style="font-size:1rem;color:#4F8BF9;margin-left:0.6rem;">{w.get("meaning","")}</span>'
                        f'</div>'
                    )
                st.markdown(words_html, unsafe_allow_html=True)

                max_learned_code = int(learned_df["code_num"].max()) if not learned_df.empty else 0
                render_used_vocab(combo_data["sentence"], study_df, max_learned_code, vocab_codes=combo_data.get("vocab_codes"))

                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("🔊 播放例句", key="combo_tts", use_container_width=True):
                    try:
                        with st.spinner("生成發音中..."):
                            audio_bytes = generate_tts_audio(_sent_tts_text(combo_data, language), language)
                        components.html(audio_player(audio_bytes), height=64)
                    except Exception as e:
                        st.error(f"TTS 失敗：{e}")

    # ════════════════════════════════════════════════════
    # 模式 3：短文練習（短文 / 故事 / 對話）
    # ════════════════════════════════════════════════════
    if mode == "📝 短文練習":
        st.caption("AI 根據你目前學過的詞彙，生成一段短文、故事或對話。最多 50 字，自然流暢。")

        if len(learned_df) < 3:
            st.info("需要至少學 3 個單字才能使用短文練習。")
        else:
            # ── 類型 & 字數設定 ────────────────────────────
            pc1, pc2 = st.columns([2, 1])
            with pc1:
                ptype = st.radio(
                    "選擇類型",
                    ["短文", "故事", "對話"],
                    horizontal=True,
                    key="passage_type_radio",
                    index=["短文", "故事", "對話"].index(st.session_state.passage_type),
                )
            with pc2:
                max_chars = st.slider(
                    "最多字數",
                    min_value=20, max_value=150, value=50, step=10,
                    key="passage_max_chars",
                    help="CJK 語言以字元計，西班牙文以單詞計"
                )
            if ptype != st.session_state.passage_type:
                st.session_state.passage_type = ptype
                st.session_state.passage_data = {"passage": "", "reading": "", "translation": "", "vocab_notes": [], "grammar_notes": "", "char_count": 0, "unit_label": "字元"}
                st.rerun()

            # ── 生成按鈕 ──────────────────────────────────
            if st.button("✨ 生成新短文", use_container_width=True, key="passage_gen") or not st.session_state.passage_data.get("passage"):
                try:
                    with st.spinner(f"AI 正在生成{ptype}..."):
                        passage_result = generate_passage(
                            language=language,
                            passage_type=ptype,
                            all_allowed_vocab=_df_to_allowed_vocab(learned_df),
                            max_length=max_chars,
                        )
                    st.session_state.passage_data = passage_result
                    st.rerun()
                except Exception as e:
                    st.error(f"生成失敗：{e}")

            pdata = st.session_state.passage_data
            if pdata.get("passage"):
                # ── 字數統計 ──────────────────────────────
                char_count  = pdata.get("char_count", 0)
                unit_label  = pdata.get("unit_label", "字元")
                over_limit  = char_count > max_chars
                count_color = "#e03030" if over_limit else "#4caf50"
                count_badge = (
                    f'<span style="font-size:0.85rem;color:{count_color};'
                    f'font-weight:600;margin-left:0.6rem;">'
                    f'📊 {char_count} / {max_chars} {unit_label}'
                    f'{"　⚠️ 超過上限" if over_limit else ""}</span>'
                )

                # ── 短文卡片 ──────────────────────────────
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="study-label">📄 {ptype}{count_badge}</div>',
                    unsafe_allow_html=True
                )

                # 對話格式換行顯示
                passage_html = pdata["passage"].replace("\n", "<br>")
                st.markdown(f'<div class="study-value-md" style="font-size:1.25rem;line-height:1.8;">{passage_html}</div>', unsafe_allow_html=True)

                if supports_reading and pdata.get("reading"):
                    reading_html = pdata["reading"].replace("\n", "<br>")
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md" style="color:#555;">{reading_html}</div>', unsafe_allow_html=True)

                st.markdown('<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                translation_html = pdata["translation"].replace("\n", "<br>")
                st.markdown(f'<div class="study-value-md">{translation_html}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── TTS：可暫停／繼續 ──────────────────────
                tts_text = pdata["passage"]
                if st.button("🔊 生成朗讀", key="passage_tts", use_container_width=True):
                    try:
                        with st.spinner("生成朗讀中..."):
                            audio_bytes = generate_tts_audio(tts_text, language)
                        components.html(audio_player_pausable(audio_bytes), height=64)
                    except Exception as e:
                        st.error(f"TTS 失敗：{e}")

                st.divider()

                # ── 單字解釋 ──────────────────────────────
                vocab_notes = pdata.get("vocab_notes", [])
                if vocab_notes:
                    st.markdown("**📚 本文單字**")
                    for note in vocab_notes:
                        word    = note.get("word", "")
                        reading = note.get("reading", "")
                        meaning = note.get("meaning", "")
                        pos     = note.get("pos", "")
                        reading_part = f"（{reading}）" if reading else ""
                        pos_part     = f" [{pos}]" if pos else ""
                        st.markdown(
                            f'<div style="margin-bottom:0.4rem;">'
                            f'<span style="font-size:1.1rem;font-weight:700;">{word}</span>'
                            f'<span style="color:#888;font-size:0.95rem;">{reading_part}</span>'
                            f'<span style="color:#4F8BF9;font-size:0.9rem;">{pos_part}</span>'
                            f'<span style="color:#333;margin-left:0.5rem;">＝ {meaning}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # ── 文法說明 ──────────────────────────────
                if pdata.get("grammar_notes"):
                    st.markdown("**📝 文法說明**")
                    _render_grammar_box(pdata["grammar_notes"])

    st.divider()
    if st.button("↩ 回到語言首頁", use_container_width=True):
        st.session_state.page = "language_home"; st.rerun()


# ══════════════════════════════════════════════════════════
# 句型詞庫編輯頁
# ══════════════════════════════════════════════════════════
def pattern_vocab_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config = get_language_config(language)
    display_name = lang_config["label"] if lang_config else language.capitalize()
    flag = LANGUAGE_FLAGS.get(language, "🌐")

    st.title(f"{flag} {display_name} — 句型詞庫")
    st.caption(
        "專門用於句型練習的詞彙表，和主詞彙學習完全獨立。"
        "「#」欄位可留空，系統會自動編號。離開欄位後即自動儲存。"
    )

    df = get_current_pattern_vocab_df(language)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"pattern_editor_{language}",
        column_config={
            "code":    st.column_config.TextColumn("#",    width="small"),
            "term":    st.column_config.TextColumn("單字", width="medium"),
            "reading": st.column_config.TextColumn("讀音", width="medium"),
            "meaning": st.column_config.TextColumn("意思", width="medium"),
            "pos":     st.column_config.TextColumn("詞性", width="small"),
            "note":    st.column_config.TextColumn("備註", width="large"),
        }
    )
    st.session_state.pattern_vocab_df = edited_df

    # 自動儲存
    _autosave_key = f"pattern_autosave_hash_{language}"
    _current_hash = str(pd.util.hash_pandas_object(edited_df, index=True).sum())
    _last_hash    = st.session_state.get(_autosave_key)
    if _last_hash is None:
        st.session_state[_autosave_key] = _current_hash
    elif _current_hash != _last_hash:
        try:
            save_pattern_vocab(language, edited_df)
            st.session_state[_autosave_key] = _current_hash
            st.session_state.pattern_vocab_df = edited_df
            st.toast("✅ 已自動儲存", icon="✅")
        except Exception as e:
            st.toast(f"❌ 自動儲存失敗：{e}", icon="❌")

    if st.button("💾 儲存句型詞庫", use_container_width=True):
        try:
            save_pattern_vocab(language, edited_df)
            st.session_state.pop(f"pattern_editor_{language}", None)
            st.session_state.pattern_vocab_df = None
            st.session_state.pattern_vocab_loaded_language = None
            st.session_state.pop(_autosave_key, None)
            load_pattern_vocab_into_state(language)
            st.success("句型詞庫已儲存。")
            st.rerun()
        except Exception as e:
            st.error(f"儲存失敗：{e}")
    if st.button("🔄 重新載入", use_container_width=True):
            try:
                st.session_state.pop(f"pattern_editor_{language}", None)
                st.session_state.pattern_vocab_df = None
                st.session_state.pattern_vocab_loaded_language = None
                st.session_state.pop(_autosave_key, None)
                load_pattern_vocab_into_state(language)
                st.success("已重新載入句型詞庫。")
                st.rerun()
            except Exception as e:
                st.error(f"重新載入失敗：{e}")

    st.divider()
    if st.button("← Back", use_container_width=True):
        st.session_state.page = "language_home"; st.rerun()


# ══════════════════════════════════════════════════════════
# 句型練習頁（用句型詞庫生成練習句）
# ══════════════════════════════════════════════════════════
def pattern_study_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config = get_language_config(language)
    display_name     = lang_config["label"] if lang_config else language.capitalize()
    reading_label    = lang_config["reading_label"] if lang_config else "Reading"
    supports_reading = lang_config["supports_reading"] if lang_config else False

    raw_df   = get_current_pattern_vocab_df(language)
    study_df = prepare_study_df(raw_df)
    pattern_full_vocab_list = _df_to_allowed_vocab(study_df)

    if study_df.empty:
        st.warning("句型詞庫是空的。請先到「自訂句型」頁面新增詞彙。")
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "language_home"; st.rerun()
        return

    # ── 索引邊界保護 ───────────────────────────────────────
    if st.session_state.pattern_study_index >= len(study_df):
        st.session_state.pattern_study_index = len(study_df) - 1

    current      = get_current_row(study_df, st.session_state.pattern_study_index)
    current_code = int(current["code_num"])
    current_term = current["term"]
    st.session_state.pattern_study_current_code = str(current["code"])

    allowed_df    = get_allowed_vocab(study_df, current_code)
    allowed_terms = allowed_df["term"].astype(str).tolist()

    # ── 批次預生成例句（每 10 個句型一批，進入新批次時一次生成完畢）──
    _pattern_lang_key = f"{language}_pattern"
    _current_batch = st.session_state.pattern_study_index // 10
    if _current_batch != st.session_state.get("pattern_study_batch_generated", -1):
        _batch_start = _current_batch * 10
        _batch_end   = min(_batch_start + 10, len(study_df))
        _missing = [
            i for i in range(_batch_start, _batch_end)
            if not get_cached_sentence(_pattern_lang_key, str(get_current_row(study_df, i)["code"])).get("sentence")
        ]
        if _missing:
            _batch_results = {}
            with st.spinner(f"AI 正在預先生成第 {_batch_start + 1}–{_batch_end} 個例句（共 {len(_missing)} 個）…"):
                for _bi in _missing:
                    _row   = get_current_row(study_df, _bi)
                    _bcode = str(_row["code"])
                    _bcode_num = int(_row["code_num"])
                    _b_allowed_df = get_allowed_vocab(study_df, _bcode_num)
                    try:
                        _bresult = generate_example_sentence(
                            language=language,
                            current_term=str(_row["term"]),
                            allowed_terms=_b_allowed_df["term"].astype(str).tolist(),
                            term_meaning=str(_row.get("meaning", "")),
                            term_reading=str(_row.get("reading", "")),
                            term_pos=str(_row.get("pos", "")),
                            current_code=_bcode_num,
                            allowed_vocab=_df_to_allowed_vocab(_b_allowed_df),
                            full_vocab=pattern_full_vocab_list,
                        )
                        _batch_results[_bcode] = _bresult
                    except Exception:
                        pass
            # 一次性存入快取（1 次 GitHub write，而非 N 次）
            set_cached_sentences_bulk(_pattern_lang_key, _batch_results)
        st.session_state.pattern_study_batch_generated = _current_batch

    cached_sentence = get_cached_sentence(f"{language}_pattern", str(current["code"]))
    needs_switch    = st.session_state.pattern_study_sentence_term != current_term

    # 決定是否需要 AI 生成：有快取直接套用；無快取則延遲到右欄渲染後再生成
    _gen_needed = False
    if needs_switch:
        if cached_sentence.get("sentence") and _is_cache_sentence_valid(
            sentence_data=cached_sentence,
            vocab_df=study_df,
            current_code=current_code,
            language=language,
        ):
            # 有快取 → 立即套用，本次渲染就能顯示
            st.session_state.pattern_study_sentence = cached_sentence
            st.session_state.pattern_study_sentence_term = current_term
            st.session_state.pattern_study_current_code = str(current["code"])
        else:
            # 無快取 → 先清空句子、標記待生成，讓詞彙卡先渲染出來
            st.session_state.pattern_study_sentence = {}
            st.session_state.pattern_study_sentence_term = current_term
            st.session_state.pattern_study_current_code = str(current["code"])
            _gen_needed = True
    elif st.session_state.pattern_study_sentence.get("sentence") and not _is_cache_sentence_valid(
        sentence_data=st.session_state.pattern_study_sentence,
        vocab_df=study_df,
        current_code=current_code,
        language=language,
    ):
        st.session_state.pattern_study_sentence = {}
        _gen_needed = True

    # _code_str 在整個頁面（TTS 按鈕）都會用到，保留在此
    _code_str = str(current["code"])

    # ── 頂部導航列（第一個 UI 元素，點按後立即渲染）──────────────────
    _pct = st.session_state.pattern_study_index / max(len(study_df) - 1, 1)
    _tnl, _tnm, _tnr = st.columns([2, 5, 2])
    with _tnl:
        if st.button("⬅ 上一個", key="pat_study_prev_top", use_container_width=True):
            st.session_state.pattern_study_index = get_prev_index(study_df, st.session_state.pattern_study_index)
            st.session_state.pattern_study_sentence_term = ""
            st.rerun()
    with _tnm:
        st.progress(_pct)
        st.caption(f"🗣️ {display_name} 句型　{st.session_state.pattern_study_index + 1} / {len(study_df)}　｜　Available: {len(allowed_terms)}")
        st.caption(f"Build: {_build_hash()}")
    with _tnr:
        if st.button("下一個 ➡", key="pat_study_next_top", use_container_width=True):
            st.session_state.pattern_study_index = get_next_index(study_df, st.session_state.pattern_study_index)
            st.session_state.pattern_study_sentence_term = ""
            st.rerun()

    # ══ 左右分欄（電腦左右、手機上下）══════════════════════
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── 左欄：詞彙資訊 ────────────────────────────────────
    with col_left:
        st.markdown('<div class="study-card">', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Code</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-md">{current["code"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Term</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-lg">{current["term"]}</div>', unsafe_allow_html=True)

        if supports_reading:
            st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{current.get("reading", "")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Meaning</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-md">{current.get("meaning", "")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="study-label">Part of Speech</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="study-value-md">{current.get("pos", "")}</div>', unsafe_allow_html=True)

        if current.get("note", "").strip():
            st.markdown('<div class="study-label">Note</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{current.get("note", "")}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # TTS 發音
        if st.session_state.pattern_tts_term_for != current_term:
            st.session_state.pattern_tts_term_audio = None
            st.session_state.pattern_tts_term_for = ""

        if st.button("🔊 播放發音", key="pat_study_tts_term_btn", use_container_width=True):
            try:
                audio_bytes = (
                    (st.session_state.pattern_tts_term_audio if st.session_state.pattern_tts_term_for == current_term else None)
                    or get_cached_tts(language, _code_str, "term")
                )
                if not audio_bytes:
                    with st.spinner("生成發音中..."):
                        audio_bytes = generate_tts_audio(current_term, language)
                        set_cached_tts(language, _code_str, "term", audio_bytes)
                st.session_state.pattern_tts_term_audio = audio_bytes
                st.session_state.pattern_tts_term_for = current_term
                components.html(audio_player(audio_bytes), height=64)
            except Exception as e:
                st.error(f"TTS 失敗：{e}")

        # ── 熟悉度標記 ────────────────────────────────────
        _fam = get_familiarity(language, current_code)
        _fam_labels = {FAMILIAR: "✅ 熟悉", UNFAMILIAR: "❗ 陌生", None: ""}
        if _fam:
            st.caption(f"目前標記：{_fam_labels[_fam]}")
        fam_c1, fam_c2 = st.columns(2)
        with fam_c1:
            _btn_fam = "✅ 熟悉（已標）" if _fam == FAMILIAR else "✅ 熟悉"
            if st.button(_btn_fam, key="pat_study_fam_familiar", use_container_width=True):
                set_familiarity(language, current_code, None if _fam == FAMILIAR else FAMILIAR)
                st.rerun()
        with fam_c2:
            _btn_unfam = "❗ 陌生（已標）" if _fam == UNFAMILIAR else "❗ 陌生"
            if st.button(_btn_unfam, key="pat_study_fam_unfamiliar", use_container_width=True):
                set_familiarity(language, current_code, None if _fam == UNFAMILIAR else UNFAMILIAR)
                st.rerun()

    # ── 右欄：例句＋文法 ──────────────────────────────────
    with col_right:
        sentence_data = st.session_state.pattern_study_sentence

        st.markdown('<div class="study-card">', unsafe_allow_html=True)
        st.markdown('<div class="study-label">Example Sentence</div>', unsafe_allow_html=True)

        if _gen_needed:
            # 詞彙卡已渲染，在右欄生成 AI 例句（用戶可看到左欄詞彙）
            st.markdown('<div class="study-value-md" style="color:#aaa;">正在生成例句…</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            _generated_ok = False
            with st.spinner("AI 正在生成例句…"):
                try:
                    result = generate_example_sentence(
                        language=language,
                        current_term=current_term,
                        allowed_terms=allowed_terms,
                        term_meaning=str(current.get("meaning", "")),
                        term_reading=str(current.get("reading", "")),
                        term_pos=str(current.get("pos", "")),
                        current_code=current_code,
                        allowed_vocab=_df_to_allowed_vocab(allowed_df),
                        full_vocab=pattern_full_vocab_list,
                    )
                    st.session_state.pattern_study_sentence = result
                    set_cached_sentence(f"{language}_pattern", str(current["code"]), result)
                    _generated_ok = True
                except Exception as e:
                    log_error(f"[pattern_study_page] auto generate failed lang={language} code={current.get('code','')}: {e}")
                    st.error(f"自動生成例句失敗：{e}")
            if _generated_ok:
                st.rerun()
        elif sentence_data.get("sentence"):
            st.markdown(f'<div class="study-value-md">{sentence_data["sentence"]}</div>', unsafe_allow_html=True)

            if supports_reading and sentence_data.get("reading"):
                st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{sentence_data["reading"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="study-label">Translation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{sentence_data.get("translation", "")}</div>', unsafe_allow_html=True)

            if sentence_data.get("grammar"):
                st.markdown('<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                _render_grammar_box(sentence_data["grammar"])

            render_used_vocab(sentence_data["sentence"], study_df, current_code, vocab_codes=sentence_data.get("vocab_codes"))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="study-value-md" style="color:#aaa;">正在生成例句…</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 新例句 + TTS 例句
        current_sentence = sentence_data.get("sentence", "")
        if st.session_state.pattern_tts_sentence_for != current_sentence:
            st.session_state.pattern_tts_sentence_audio = None
            st.session_state.pattern_tts_sentence_for = ""

        if st.button("🔀 新例句", key="pat_study_new_sentence_btn", use_container_width=True):
            try:
                with st.spinner("AI 正在生成新例句..."):
                    result = generate_example_sentence(
                        language=language,
                        current_term=current_term,
                        allowed_terms=allowed_terms,
                        term_meaning=str(current.get("meaning", "")),
                        term_reading=str(current.get("reading", "")),
                        term_pos=str(current.get("pos", "")),
                        current_code=current_code,
                        allowed_vocab=_df_to_allowed_vocab(allowed_df),
                        full_vocab=pattern_full_vocab_list,
                    )
                    st.session_state.pattern_study_sentence = result
                    st.session_state.pattern_study_sentence_term = current_term
                    st.session_state.pattern_study_current_code = str(current["code"])
                    set_cached_sentence(f"{language}_pattern", str(current["code"]), result)
                st.rerun()
            except Exception as e:
                log_error(f"[pattern_study_page] new sentence failed lang={language} code={current.get('code','')}: {e}")
                st.error(f"生成例句失敗：{e}")

        if st.button("🔊 播放例句", key="pat_study_tts_sentence_btn", use_container_width=True):
            try:
                if st.session_state.pattern_tts_sentence_audio and st.session_state.pattern_tts_sentence_for == current_sentence:
                    audio_bytes = st.session_state.pattern_tts_sentence_audio
                else:
                    _tts_s = _sent_tts_text(sentence_data, language)
                    with st.spinner("生成例句發音中..."):
                        audio_bytes = generate_tts_audio(_tts_s, language)
                        st.session_state.pattern_tts_sentence_audio = audio_bytes
                        st.session_state.pattern_tts_sentence_for = current_sentence
                components.html(audio_player(audio_bytes), height=64)
            except Exception as e:
                st.error(f"TTS 失敗：{e}")

    # ── 底部導航 ───────────────────────────────────────────
    st.divider()
    _bnl, _bnm, _bnr = st.columns([2, 3, 2])
    with _bnl:
        if st.button("⬅ 上一個", key="pat_study_prev_bot", use_container_width=True):
            st.session_state.pattern_study_index = get_prev_index(study_df, st.session_state.pattern_study_index)
            st.session_state.pattern_study_sentence_term = ""
            st.rerun()
    with _bnm:
        if st.button("↩ 回到語言首頁", key="pat_study_back", use_container_width=True):
            st.session_state.page = "language_home"
            st.rerun()
    with _bnr:
        if st.button("下一個 ➡", key="pat_study_next_bot", use_container_width=True):
            st.session_state.pattern_study_index = get_next_index(study_df, st.session_state.pattern_study_index)
            st.session_state.pattern_study_sentence_term = ""
            st.rerun()

    # ── 跳號（移到底部）──────────────────────────────────────────
    jc1, jc2 = st.columns([4, 1])
    with jc1:
        jump_val = st.number_input("跳到編號", min_value=0, value=0, step=1,
                                   key="pattern_study_jump_input", label_visibility="collapsed")
    with jc2:
        if st.button("跳至", key="pattern_study_jump_btn", use_container_width=True):
            if jump_val > 0:
                matches = study_df[study_df["code_num"] == int(jump_val)]
                if not matches.empty:
                    st.session_state.pattern_study_index = int(matches.index[0])
                    st.session_state.pattern_study_sentence_term = ""
                    st.rerun()
                else:
                    st.warning(f"找不到編號 {int(jump_val)}")

    # ── 自動播放（放在頁面最後，確保詞彙卡和例句已先渲染完畢）──────
    _sentence_data_auto = st.session_state.pattern_study_sentence
    _cur_sent_auto = _sentence_data_auto.get("sentence", "")
    if _cur_sent_auto and not _gen_needed and st.session_state.pattern_study_auto_played_for != _cur_sent_auto:
        try:
            _term_audio = (
                get_cached_tts(language, _code_str, "term")
                or (st.session_state.pattern_tts_term_for == current_term and st.session_state.pattern_tts_term_audio)
            )
            if not _term_audio:
                with st.spinner("生成發音中..."):
                    _term_audio = generate_tts_audio(current_term, language)
                    set_cached_tts(language, _code_str, "term", _term_audio)
            st.session_state.pattern_tts_term_audio = _term_audio
            st.session_state.pattern_tts_term_for = current_term
            _sent_tts_val = _sent_tts_text(_sentence_data_auto, language)
            _sent_audio = (
                get_cached_tts(f"{language}_pattern", _code_str, "sent", _sent_tts_val)
                or (st.session_state.pattern_tts_sentence_for == _cur_sent_auto and st.session_state.pattern_tts_sentence_audio)
            )
            if not _sent_audio:
                with st.spinner("生成例句發音中..."):
                    _sent_audio = generate_tts_audio(_sent_tts_val, language)
                    set_cached_tts(f"{language}_pattern", _code_str, "sent", _sent_audio, _sent_tts_val)
            st.session_state.pattern_tts_sentence_audio = _sent_audio
            st.session_state.pattern_tts_sentence_for = _cur_sent_auto
            components.html(audio_player_dual(_term_audio, _sent_audio), height=0)
            st.session_state.pattern_study_auto_played_for = _cur_sent_auto
        except Exception as e:
            st.warning(f"自動播放失敗：{e}")


# ══════════════════════════════════════════════════════════
# 句型複習頁（從句型詞庫隨機抽字，用句型詞庫例句出題）
# ══════════════════════════════════════════════════════════
def pattern_review_page():
    language = st.session_state.language
    if not language:
        go_home(); st.rerun()

    lang_config = get_language_config(language)
    display_name = lang_config["label"] if lang_config else language.capitalize()
    reading_label = lang_config["reading_label"] if lang_config else "Reading"
    supports_reading = lang_config["supports_reading"] if lang_config else False
    flag = LANGUAGE_FLAGS.get(language, "🌐")

    st.title(f"{flag} {display_name} — 句型複習")

    raw_df = get_current_pattern_vocab_df(language)
    study_df = prepare_study_df(raw_df)
    pattern_review_full_vocab_list = _df_to_allowed_vocab(study_df)

    if study_df.empty:
        st.warning("句型詞庫是空的。請先到「句型詞庫」頁面新增詞彙。")
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "language_home"; st.rerun()
        return

    # ── 練習範圍設定 ──────────────────────────────────────
    all_min = int(study_df["code_num"].min())
    all_max = int(study_df["code_num"].max())

    # 初始化：首次進入或超出範圍時重設
    if st.session_state.pattern_review_code_min is None or st.session_state.pattern_review_code_min < all_min:
        st.session_state.pattern_review_code_min = all_min
    if st.session_state.pattern_review_code_max is None or st.session_state.pattern_review_code_max > all_max:
        st.session_state.pattern_review_code_max = all_max

    with st.expander("🎯 練習範圍設定（預設：全部）", expanded=False):
        st.caption(f"全部句型範圍：{all_min} ～ {all_max}　共 {len(study_df)} 個句型")
        rc1, rc2 = st.columns(2)
        with rc1:
            new_min = st.number_input(
                "從編號", min_value=all_min, max_value=all_max,
                value=st.session_state.pattern_review_code_min,
                step=1, key="pattern_review_range_min"
            )
        with rc2:
            new_max = st.number_input(
                "到編號", min_value=all_min, max_value=all_max,
                value=st.session_state.pattern_review_code_max,
                step=1, key="pattern_review_range_max"
            )
        if new_min > new_max:
            st.warning("「從編號」不能大於「到編號」，已自動對調。")
            new_min, new_max = new_max, new_min

        # 範圍改變時清除當前題目，觸發重抽
        if new_min != st.session_state.pattern_review_code_min or new_max != st.session_state.pattern_review_code_max:
            st.session_state.pattern_review_code_min = new_min
            st.session_state.pattern_review_code_max = new_max
            st.session_state.pattern_review_term = ""
            st.session_state.pattern_combo_words = []
            st.rerun()

    range_min = st.session_state.pattern_review_code_min
    range_max = st.session_state.pattern_review_code_max

    # 過濾 study_df 到指定範圍
    filtered_df = study_df[
        (study_df["code_num"] >= range_min) & (study_df["code_num"] <= range_max)
    ]
    if filtered_df.empty:
        st.warning(f"編號 {range_min}～{range_max} 範圍內沒有句型，請調整範圍。")
        return

    # 若當前題目已不在範圍內，清除讓系統重抽
    if st.session_state.pattern_review_term:
        cur_code = st.session_state.pattern_review_term_code
        if cur_code < range_min or cur_code > range_max:
            st.session_state.pattern_review_term = ""
            st.session_state.pattern_combo_words = []

    # ── 模式切換 ──────────────────────────────────────────
    mode = st.radio(
        "複習模式",
        ["📖 單字複習", "🔀 重組練習", "📝 短文練習"],
        horizontal=True,
        key="pattern_review_mode_radio"
    )
    st.divider()

    # ════════════════════════════════════════════════════
    # 模式 1：單字複習 — FSI Substitution + Transformation
    # ════════════════════════════════════════════════════
    if mode == "📖 單字複習":
        st.caption("隨機抽一個句型詞彙，透過 FSI 練習法觀察搭配詞（Substitution）與句型變化（Transformation）。")

        # ── 抽題 ──────────────────────────────────────
        if st.button("🎲 抽新題目", use_container_width=True, key="pat_review_word_draw"):
            picked       = filtered_df.sample(1).iloc[0]
            pick_code    = int(picked["code_num"])
            pick_term    = str(picked["term"])
            pick_meaning = str(picked.get("meaning", ""))
            pick_reading = str(picked.get("reading", ""))
            pick_pos     = str(picked.get("pos", ""))
            allowed_df   = get_allowed_vocab(filtered_df, pick_code)
            allowed_vocab_list = _df_to_allowed_vocab(allowed_df)

            fsi_kwargs = dict(
                language=language,
                current_term=pick_term,
                term_meaning=pick_meaning,
                term_reading=pick_reading,
                term_pos=pick_pos,
                current_code=pick_code,
                allowed_vocab=allowed_vocab_list,
                full_vocab=pattern_review_full_vocab_list,
            )
            try:
                with st.spinner("AI 正在生成搭配練習句..."):
                    sub_data = generate_fsi_sentence(drill_type="substitution", **fsi_kwargs)
            except Exception as e:
                st.error(f"Substitution 生成失敗：{e}")
                sub_data = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}

            try:
                with st.spinner("AI 正在生成句型變化句..."):
                    trans_data = generate_fsi_sentence(drill_type="transformation", **fsi_kwargs)
            except Exception as e:
                st.error(f"Transformation 生成失敗：{e}")
                trans_data = {"sentence": "", "reading": "", "translation": "", "grammar": "", "drill_note": ""}

            st.session_state.pattern_review_term          = pick_term
            st.session_state.pattern_review_meaning       = pick_meaning
            st.session_state.pattern_review_term_code     = pick_code
            st.session_state.pattern_review_term_reading  = pick_reading
            st.session_state.pattern_review_term_pos      = pick_pos
            st.session_state.pattern_review_sub_sentence  = sub_data
            st.session_state.pattern_review_trans_sentence= trans_data
            st.session_state.pattern_review_sub_prev_notes  = [sub_data["drill_note"]] if sub_data.get("drill_note") else []
            st.session_state.pattern_review_trans_prev_notes= [trans_data["drill_note"]] if trans_data.get("drill_note") else []
            st.rerun()

        if not st.session_state.pattern_review_term:
            st.info("按「抽新題目」後才會產生 Substitution 和 Transformation 練習，避免一進頁面就等待 AI 生成。")
            return

        # ── 左右分欄 ──────────────────────────────
        rev_left, rev_right = st.columns([1, 1], gap="large")

        # ── 左欄：單字資訊卡 ──────────────────────────
        with rev_left:
            pick_code = st.session_state.pattern_review_term_code
            pick_term = st.session_state.pattern_review_term

            st.markdown('<div class="study-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="study-label">Code</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-md">{pick_code}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-label">單字</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="study-value-lg">{pick_term}</div>', unsafe_allow_html=True)
            if supports_reading and st.session_state.pattern_review_term_reading:
                st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{st.session_state.pattern_review_term_reading}</div>', unsafe_allow_html=True)
            if st.session_state.pattern_review_meaning:
                st.markdown(f'<div class="study-label">意思</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md" style="color:#4F8BF9;">{st.session_state.pattern_review_meaning}</div>', unsafe_allow_html=True)
            if st.session_state.pattern_review_term_pos:
                st.markdown(f'<div class="study-label">詞性</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{st.session_state.pattern_review_term_pos}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔊 播放發音", key="pat_review_term_tts", use_container_width=True):
                try:
                    with st.spinner("生成發音中..."):
                        audio_bytes = generate_tts_audio(pick_term, language)
                    components.html(audio_player(audio_bytes), height=64)
                except Exception as e:
                    st.error(f"TTS 失敗：{e}")

            # ── 熟悉度標記 ────────────────────────────
            _fam_r = get_familiarity(language, pick_code)
            _fam_labels_r = {FAMILIAR: "✅ 熟悉", UNFAMILIAR: "❗ 陌生"}
            if _fam_r:
                st.caption(f"目前標記：{_fam_labels_r[_fam_r]}")
            rfam_c1, rfam_c2 = st.columns(2)
            with rfam_c1:
                _rb_fam = "✅ 熟悉（已標）" if _fam_r == FAMILIAR else "✅ 熟悉"
                if st.button(_rb_fam, key="pat_rev_fam_familiar", use_container_width=True):
                    set_familiarity(language, pick_code, None if _fam_r == FAMILIAR else FAMILIAR)
                    st.rerun()
            with rfam_c2:
                _rb_unfam = "❗ 陌生（已標）" if _fam_r == UNFAMILIAR else "❗ 陌生"
                if st.button(_rb_unfam, key="pat_rev_fam_unfamiliar", use_container_width=True):
                    set_familiarity(language, pick_code, None if _fam_r == UNFAMILIAR else UNFAMILIAR)
                    st.rerun()

        # ── 右欄：FSI 練習區 ──────────────────────────
        with rev_right:
            allowed_df = get_allowed_vocab(filtered_df, st.session_state.pattern_review_term_code)
            allowed_vocab_list = _df_to_allowed_vocab(allowed_df)
            fsi_base = dict(
                language=language,
                current_term=st.session_state.pattern_review_term,
                term_meaning=st.session_state.pattern_review_meaning,
                term_reading=st.session_state.pattern_review_term_reading,
                term_pos=st.session_state.pattern_review_term_pos,
                current_code=st.session_state.pattern_review_term_code,
                allowed_vocab=allowed_vocab_list,
                full_vocab=pattern_review_full_vocab_list,
            )

            # ===== Substitution 搭配練習 =====
            st.markdown("#### 🔄 Substitution — 搭配練習")
            st.caption("觀察目標單字與不同搭配詞或情境的組合")

            sub_data = st.session_state.pattern_review_sub_sentence
            if sub_data.get("sentence"):
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                if sub_data.get("drill_note"):
                    st.markdown(f'<div class="study-label">搭配說明</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md" style="color:#e65100;font-weight:600;">{sub_data["drill_note"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{sub_data["sentence"]}</div>', unsafe_allow_html=True)
                if supports_reading and sub_data.get("reading"):
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md">{sub_data["reading"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{sub_data.get("translation","")}</div>', unsafe_allow_html=True)
                if sub_data.get("grammar"):
                    st.markdown(f'<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                    _render_grammar_box(sub_data["grammar"])
                render_used_vocab(sub_data["sentence"], study_df, st.session_state.pattern_review_term_code, vocab_codes=sub_data.get("vocab_codes"))
                st.markdown('</div>', unsafe_allow_html=True)

            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                if st.button("🔀 下一個", key="pat_sub_next_btn", use_container_width=True):
                    prev = st.session_state.get("pattern_review_sub_prev_notes", [])
                    try:
                        with st.spinner("生成下一個搭配..."):
                            new_sub = generate_fsi_sentence(drill_type="substitution", prev_drill_notes=prev, **fsi_base)
                        st.session_state.pattern_review_sub_sentence = new_sub
                        if new_sub.get("drill_note"):
                            st.session_state.pattern_review_sub_prev_notes = (prev + [new_sub["drill_note"]])[-6:]
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成失敗：{e}")
            with sub_c2:
                if st.button("🔊 發音", key="pat_sub_tts_btn", use_container_width=True):
                    if sub_data.get("sentence"):
                        try:
                            with st.spinner("生成發音中..."):
                                audio_bytes = generate_tts_audio(_sent_tts_text(sub_data, language), language)
                            components.html(audio_player(audio_bytes), height=64)
                        except Exception as e:
                            st.error(f"TTS 失敗：{e}")

            st.divider()

            # ===== Transformation 句型變化 =====
            st.markdown("#### ⚙️ Transformation — 句型變化")
            st.caption("觀察目標單字在不同文法形態下的變化")

            trans_data = st.session_state.pattern_review_trans_sentence
            if trans_data.get("sentence"):
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                if trans_data.get("drill_note"):
                    st.markdown(f'<div class="study-label">文法形態</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md" style="color:#1565c0;font-weight:600;">{trans_data["drill_note"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{trans_data["sentence"]}</div>', unsafe_allow_html=True)
                if supports_reading and trans_data.get("reading"):
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md">{trans_data["reading"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{trans_data.get("translation","")}</div>', unsafe_allow_html=True)
                if trans_data.get("grammar"):
                    st.markdown(f'<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                    _render_grammar_box(trans_data["grammar"])
                render_used_vocab(trans_data["sentence"], study_df, st.session_state.pattern_review_term_code, vocab_codes=trans_data.get("vocab_codes"))
                st.markdown('</div>', unsafe_allow_html=True)

            trans_c1, trans_c2 = st.columns(2)
            with trans_c1:
                if st.button("🔀 下一個", key="pat_trans_next_btn", use_container_width=True):
                    prev = st.session_state.get("pattern_review_trans_prev_notes", [])
                    try:
                        with st.spinner("生成下一個變化..."):
                            new_trans = generate_fsi_sentence(drill_type="transformation", prev_drill_notes=prev, **fsi_base)
                        st.session_state.pattern_review_trans_sentence = new_trans
                        if new_trans.get("drill_note"):
                            st.session_state.pattern_review_trans_prev_notes = (prev + [new_trans["drill_note"]])[-6:]
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成失敗：{e}")
            with trans_c2:
                if st.button("🔊 發音", key="pat_trans_tts_btn", use_container_width=True):
                    if trans_data.get("sentence"):
                        try:
                            with st.spinner("生成發音中..."):
                                audio_bytes = generate_tts_audio(_sent_tts_text(trans_data, language), language)
                            components.html(audio_player(audio_bytes), height=64)
                        except Exception as e:
                            st.error(f"TTS 失敗：{e}")

    # ════════════════════════════════════════════════════
    # 模式 2：重組練習（新功能）
    # ════════════════════════════════════════════════════
    elif mode == "🔀 重組練習":
        n_learned = len(filtered_df)
        n_pick = 2  # 固定抽 2 個詞

        if n_learned < 2:
            st.info("需要至少 2 個句型才能使用重組練習。")
        else:
            st.caption(f"隨機抽 2 個句型詞彙，AI 會把它們組成一句短句。按「看答案」可查看文法分析與使用單字。")

            if st.button("🎲 抽新組合", use_container_width=True, key="pat_combo_draw") or not st.session_state.pattern_combo_words:
                picked_rows = filtered_df.sample(n=n_pick)
                all_allowed = filtered_df["term"].astype(str).tolist()

                target_words = [
                    {
                        "term":    str(row["term"]),
                        "meaning": str(row.get("meaning", "")),
                        "reading": str(row.get("reading", ""))
                    }
                    for _, row in picked_rows.iterrows()
                ]

                # 隨機挑一個文法句型
                patterns = GRAMMAR_PATTERNS.get(language, GRAMMAR_PATTERNS["default"])
                pattern  = random.choice(patterns)

                try:
                    with st.spinner("AI 正在重組新句子..."):
                        combo_sentence = generate_recombination_sentence(
                            language=language,
                            target_words=target_words,
                            all_allowed_terms=all_allowed,
                            grammar_instruction=pattern["instruction"],
                            all_allowed_vocab=_df_to_allowed_vocab(filtered_df),
                        )
                except Exception as e:
                    st.error(f"生成例句失敗：{e}")
                    combo_sentence = {"sentence": "", "reading": "", "translation": "", "grammar": ""}

                st.session_state.pattern_combo_words       = target_words
                st.session_state.pattern_combo_sentence    = combo_sentence
                st.session_state.pattern_combo_show_answer = False
                st.session_state.pattern_combo_pattern     = pattern
                st.rerun()

            # 顯示本題文法句型標籤
            pattern_label = st.session_state.pattern_combo_pattern.get("label", "")
            if pattern_label:
                st.markdown(
                    f'<div style="display:inline-block;background:#eef4fb;border:1.5px solid #4F8BF9;'
                    f'border-radius:20px;padding:0.25rem 0.9rem;font-size:0.95rem;'
                    f'color:#2c5db3;margin-bottom:0.8rem;">🎯 本題文法：{pattern_label}</div>',
                    unsafe_allow_html=True
                )

            combo_data = st.session_state.pattern_combo_sentence
            if combo_data.get("sentence"):
                # 自動播放例句（每題只播一次）
                if st.session_state.pattern_combo_auto_played_for != combo_data["sentence"]:
                    try:
                        with st.spinner("自動播放中..."):
                            audio_bytes = generate_tts_audio(_sent_tts_text(combo_data, language), language)
                        components.html(audio_player(audio_bytes), height=64)
                        st.session_state.pattern_combo_auto_played_for = combo_data["sentence"]
                    except Exception as e:
                        st.warning(f"自動播放失敗：{e}")

                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                st.markdown('<div class="study-label">重組例句</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{combo_data["sentence"]}</div>', unsafe_allow_html=True)

                if supports_reading and combo_data.get("reading"):
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md">{combo_data["reading"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="study-value-md">{combo_data.get("translation", "")}</div>', unsafe_allow_html=True)

                if combo_data.get("grammar"):
                    st.markdown('<div class="study-label">文法分析</div>', unsafe_allow_html=True)
                    _render_grammar_box(combo_data["grammar"])

                # 顯示本題使用的所有目標單字
                st.markdown('<div class="study-label">本題使用的單字</div>', unsafe_allow_html=True)
                words_html = ""
                for w in st.session_state.pattern_combo_words:
                    reading_part = f"（{w['reading']}）" if w.get("reading") else ""
                    words_html += (
                        f'<div style="margin-bottom:0.5rem;">'
                        f'<span style="font-size:1.2rem;font-weight:700;">{w["term"]}</span>'
                        f'<span style="font-size:0.95rem;color:#666;margin-left:0.4rem;">{reading_part}</span>'
                        f'<span style="font-size:1rem;color:#4F8BF9;margin-left:0.6rem;">{w.get("meaning","")}</span>'
                        f'</div>'
                    )
                st.markdown(words_html, unsafe_allow_html=True)

                max_code = int(filtered_df["code_num"].max()) if not filtered_df.empty else 0
                render_used_vocab(combo_data["sentence"], study_df, max_code, vocab_codes=combo_data.get("vocab_codes"))

                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("🔊 播放例句", key="pat_combo_tts", use_container_width=True):
                    try:
                        with st.spinner("生成發音中..."):
                            audio_bytes = generate_tts_audio(_sent_tts_text(combo_data, language), language)
                        components.html(audio_player(audio_bytes), height=64)
                    except Exception as e:
                        st.error(f"TTS 失敗：{e}")

    # ════════════════════════════════════════════════════
    # 模式 3：短文練習（短文 / 故事 / 對話）
    # ════════════════════════════════════════════════════
    if mode == "📝 短文練習":
        st.caption("AI 根據你選定的句型範圍，生成一段短文、故事或對話。最多 50 字，自然流暢。")

        if len(filtered_df) < 3:
            st.info("需要至少 3 個句型才能使用短文練習。")
        else:
            # ── 類型 & 字數設定 ────────────────────────────
            pc1, pc2 = st.columns([2, 1])
            with pc1:
                ptype = st.radio(
                    "選擇類型",
                    ["短文", "故事", "對話"],
                    horizontal=True,
                    key="passage_type_radio",
                    index=["短文", "故事", "對話"].index(st.session_state.passage_type),
                )
            with pc2:
                max_chars = st.slider(
                    "最多字數",
                    min_value=20, max_value=150, value=50, step=10,
                    key="passage_max_chars",
                    help="CJK 語言以字元計，西班牙文以單詞計"
                )
            if ptype != st.session_state.passage_type:
                st.session_state.passage_type = ptype
                st.session_state.passage_data = {"passage": "", "reading": "", "translation": "", "vocab_notes": [], "grammar_notes": "", "char_count": 0, "unit_label": "字元"}
                st.rerun()

            # ── 生成按鈕 ──────────────────────────────────
            if st.button("✨ 生成新短文", use_container_width=True, key="pat_passage_gen") or not st.session_state.passage_data.get("passage"):
                try:
                    with st.spinner(f"AI 正在生成{ptype}..."):
                        passage_result = generate_passage(
                            language=language,
                            passage_type=ptype,
                            all_allowed_vocab=_df_to_allowed_vocab(filtered_df),
                            max_length=max_chars,
                        )
                    st.session_state.passage_data = passage_result
                    st.rerun()
                except Exception as e:
                    st.error(f"生成失敗：{e}")

            pdata = st.session_state.passage_data
            if pdata.get("passage"):
                # ── 字數統計 ──────────────────────────────
                char_count  = pdata.get("char_count", 0)
                unit_label  = pdata.get("unit_label", "字元")
                over_limit  = char_count > max_chars
                count_color = "#e03030" if over_limit else "#4caf50"
                count_badge = (
                    f'<span style="font-size:0.85rem;color:{count_color};'
                    f'font-weight:600;margin-left:0.6rem;">'
                    f'📊 {char_count} / {max_chars} {unit_label}'
                    f'{"　⚠️ 超過上限" if over_limit else ""}</span>'
                )

                # ── 短文卡片 ──────────────────────────────
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="study-label">📄 {ptype}{count_badge}</div>',
                    unsafe_allow_html=True
                )

                # 對話格式換行顯示
                passage_html = pdata["passage"].replace("\n", "<br>")
                st.markdown(f'<div class="study-value-md" style="font-size:1.25rem;line-height:1.8;">{passage_html}</div>', unsafe_allow_html=True)

                if supports_reading and pdata.get("reading"):
                    reading_html = pdata["reading"].replace("\n", "<br>")
                    st.markdown(f'<div class="study-label">{reading_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="study-value-md" style="color:#555;">{reading_html}</div>', unsafe_allow_html=True)

                st.markdown('<div class="study-label">翻譯</div>', unsafe_allow_html=True)
                translation_html = pdata["translation"].replace("\n", "<br>")
                st.markdown(f'<div class="study-value-md">{translation_html}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── TTS：可暫停／繼續 ──────────────────────
                tts_text = pdata["passage"]
                if st.button("🔊 生成朗讀", key="pat_passage_tts", use_container_width=True):
                    try:
                        with st.spinner("生成朗讀中..."):
                            audio_bytes = generate_tts_audio(tts_text, language)
                        components.html(audio_player_pausable(audio_bytes), height=64)
                    except Exception as e:
                        st.error(f"TTS 失敗：{e}")

                st.divider()

                # ── 單字解釋 ──────────────────────────────
                vocab_notes = pdata.get("vocab_notes", [])
                if vocab_notes:
                    st.markdown("**📚 本文單字**")
                    for note in vocab_notes:
                        word    = note.get("word", "")
                        reading = note.get("reading", "")
                        meaning = note.get("meaning", "")
                        pos     = note.get("pos", "")
                        reading_part = f"（{reading}）" if reading else ""
                        pos_part     = f" [{pos}]" if pos else ""
                        st.markdown(
                            f'<div style="margin-bottom:0.4rem;">'
                            f'<span style="font-size:1.1rem;font-weight:700;">{word}</span>'
                            f'<span style="color:#888;font-size:0.95rem;">{reading_part}</span>'
                            f'<span style="color:#4F8BF9;font-size:0.9rem;">{pos_part}</span>'
                            f'<span style="color:#333;margin-left:0.5rem;">＝ {meaning}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # ── 文法說明 ──────────────────────────────
                if pdata.get("grammar_notes"):
                    st.markdown("**📝 文法說明**")
                    _render_grammar_box(pdata["grammar_notes"])

    st.divider()
    if st.button("↩ 回到語言首頁", use_container_width=True, key="pat_review_back"):
        st.session_state.page = "language_home"; st.rerun()


# ══════════════════════════════════════════════════════════
# AI 設定頁
# ══════════════════════════════════════════════════════════
def settings_page():
    st.title("⚙️ AI 設定")

    PROVIDER_OPTIONS = {
        "openai": "OpenAI（GPT）",
        "gemini": "Google Gemini",
        "claude": "Anthropic Claude",
    }
    MODELS = {
        "openai": ["gpt-4o-mini", "gpt-4o"],
        "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "claude": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
    }

    current_provider = st.session_state.get("ai_provider", "openai")
    provider_keys    = list(PROVIDER_OPTIONS.keys())
    provider_idx     = provider_keys.index(current_provider) if current_provider in provider_keys else 0

    selected_provider = st.selectbox(
        "AI 引擎",
        options=provider_keys,
        format_func=lambda k: PROVIDER_OPTIONS[k],
        index=provider_idx,
        key="settings_provider_select",
    )

    model_list    = MODELS.get(selected_provider, [])
    current_model = st.session_state.get("ai_model", "") or model_list[0]
    model_idx     = model_list.index(current_model) if current_model in model_list else 0

    selected_model = st.selectbox(
        "模型",
        options=model_list,
        index=model_idx,
        key="settings_model_select",
    )

    st.info(
        "API Key 請在 `.streamlit/secrets.toml` 或 Streamlit Cloud Secrets 中設定：\n\n"
        "- OpenAI → `OPENAI_API_KEY`\n"
        "- Gemini → `GEMINI_API_KEY`\n"
        "- Claude → `ANTHROPIC_API_KEY`"
    )
    st.checkbox(
        "顯示詞彙過濾偵錯資訊",
        key="debug_vocab_filter",
        help="開啟後，會在「例句使用的詞彙」下方顯示被忽略的高編號詞（功能詞或疑似斷詞）。",
    )

    if st.button("💾 儲存設定", use_container_width=True, key="settings_save"):
        st.session_state.ai_provider = selected_provider
        st.session_state.ai_model    = selected_model
        st.success(f"已儲存：{PROVIDER_OPTIONS[selected_provider]} / {selected_model}")

    st.divider()
    if st.button("← 返回首頁", use_container_width=True, key="settings_back"):
        go_home(); st.rerun()


# ══════════════════════════════════════════════════════════
def _render_translation_audio(language: str, sentence: str, key: str):
    sentence = str(sentence or "").strip()
    if not sentence:
        return
    play_key = f"{language}::{sentence}"
    cached_audio = get_cached_tts(language, "translation", "sent", sentence)
    if cached_audio:
        st.session_state.translation_tts_audio = cached_audio
        st.session_state.translation_tts_for = play_key
    if st.button("播放語音", use_container_width=True, key=key):
        try:
            audio_bytes = cached_audio or generate_tts_audio(sentence, language)
            set_cached_tts(language, "translation", "sent", audio_bytes, sentence)
            st.session_state.translation_tts_audio = audio_bytes
            st.session_state.translation_tts_for = play_key
        except Exception as e:
            st.error(f"語音產生失敗：{e}")
    audio_bytes = st.session_state.get("translation_tts_audio")
    if audio_bytes and st.session_state.get("translation_tts_for") == play_key:
        components.html(audio_player_pausable(audio_bytes), height=70)


def _render_translation_grammar(entry: dict, lang: dict, result: dict, key_prefix: str):
    lang_key = lang["key"]
    sentence = str(result.get("sentence", "") or "").strip()
    if not sentence:
        return

    grammar = str(result.get("grammar", "") or "").strip()
    button_key = f"{key_prefix}_{entry.get('id')}_{lang_key}_grammar"
    if st.button("文法解釋", use_container_width=True, key=button_key):
        if not grammar:
            try:
                with st.spinner("AI 正在分析文法..."):
                    grammar = explain_translated_sentence_grammar(
                        lang_key,
                        lang.get("label", lang_key.capitalize()),
                        str(entry.get("source", "") or ""),
                        sentence,
                    )
                update_translation_grammar(str(entry.get("id", "")), lang_key, grammar)
                result["grammar"] = grammar
            except Exception as e:
                st.error(f"文法解釋產生失敗：{e}")
                return

    if grammar:
        _render_grammar_box(grammar)


def _render_translation_entry(entry: dict, languages: list, key_prefix: str):
    st.markdown(f"**中文：** {entry.get('source', '')}")
    translations = entry.get("translations", {})
    for lang in languages:
        lang_key = lang["key"]
        result = translations.get(lang_key, {})
        sentence = str(result.get("sentence", "") or "").strip()
        if not sentence:
            continue
        label = lang.get("label", lang_key.capitalize())
        flag = LANGUAGE_FLAGS.get(lang_key, "🌐")
        reading = str(result.get("reading", "") or "").strip()
        note = str(result.get("note", "") or "").strip()
        st.markdown(f"**{flag} {label}：** {sentence}")
        if reading:
            st.caption(reading)
        if note:
            st.caption(note)
        _render_translation_audio(lang_key, sentence, f"{key_prefix}_{entry.get('id')}_{lang_key}")
        _render_translation_grammar(entry, lang, result, key_prefix)


def translation_practice_page():
    languages = load_languages()
    sentences = load_translation_sentences()

    st.title("中文句子翻譯練習")
    st.caption("在這裡建立共用中文句子庫；每一句會翻成目前所有學習語言，並可在新增列表或複習時播放語音。")

    tab_add, tab_review = st.tabs(["新增句子", "複習句子"])

    with tab_add:
        with st.form("translation_practice_form"):
            source = st.text_area(
                "中文句子",
                value=st.session_state.get("translation_input", ""),
                height=110,
                placeholder="例如：我今天想去咖啡廳讀書。",
            )
            submitted = st.form_submit_button("加入並翻譯", use_container_width=True)

        if submitted:
            source = source.strip()
            st.session_state.translation_input = source
            if not source:
                st.warning("請先輸入中文句子。")
            elif not languages:
                st.warning("目前還沒有學習語言，請先新增語言。")
            else:
                translations = {}
                with st.spinner("AI 正在翻譯成目前所有學習語言..."):
                    for lang in languages:
                        lang_key = lang["key"]
                        label = lang.get("label", lang_key.capitalize())
                        try:
                            translations[lang_key] = translate_chinese_sentence(lang_key, label, source)
                        except Exception as e:
                            translations[lang_key] = {
                                "sentence": "",
                                "reading": "",
                                "note": f"翻譯失敗：{e}",
                            }
                sentences = add_translation_sentence(source, translations)
                st.session_state.translation_input = ""
                st.success("已加入句子列表。")

        st.divider()
        st.markdown("#### 句子列表")
        if not sentences:
            st.info("目前還沒有句子。先在上方輸入一句中文吧。")
        else:
            for idx, entry in enumerate(reversed(sentences), start=1):
                with st.expander(f"{idx}. {entry.get('source', '')}", expanded=idx == 1):
                    _render_translation_entry(entry, languages, "add_list")

    with tab_review:
        if not sentences:
            st.info("目前還沒有可複習的句子。請先到「新增句子」加入。")
        else:
            mode = st.radio(
                "複習順序",
                ["依序", "隨機"],
                horizontal=True,
                key="translation_review_order",
            )
            if st.session_state.translation_review_index >= len(sentences):
                st.session_state.translation_review_index = 0

            nav_prev, nav_next = st.columns(2)
            with nav_prev:
                if st.button("上一句", use_container_width=True, disabled=(mode == "隨機"), key="translation_prev"):
                    st.session_state.translation_review_index = (
                        st.session_state.translation_review_index - 1
                    ) % len(sentences)
                    st.rerun()
            with nav_next:
                next_label = "隨機下一句" if mode == "隨機" else "下一句"
                if st.button(next_label, use_container_width=True, key="translation_next"):
                    if mode == "隨機":
                        st.session_state.translation_review_index = random.randrange(len(sentences))
                    else:
                        st.session_state.translation_review_index = (
                            st.session_state.translation_review_index + 1
                        ) % len(sentences)
                    st.rerun()

            current = sentences[st.session_state.translation_review_index]
            st.divider()
            _render_translation_entry(current, languages, "review")

    st.divider()
    if st.button("← 回到首頁", use_container_width=True, key="translation_back"):
        go_home(); st.rerun()


# 路由
# ══════════════════════════════════════════════════════════
page = st.session_state.page

if page == "home":
    home_page()
elif page == "language_home":
    language_home()
elif page == "custom_vocab":
    custom_vocab_page()
elif page == "study":
    study_page()
elif page == "review":
    review_page()
elif page == "pattern_vocab":
    pattern_vocab_page()
elif page == "pattern_study":
    pattern_study_page()
elif page == "pattern_review":
    pattern_review_page()
elif page == "translation_practice":
    translation_practice_page()
elif page == "settings":
    settings_page()
else:
    go_home()
    st.rerun()
