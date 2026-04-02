import os
import json
import re
import pandas as pd
import streamlit as st

# ── AI 引擎路由 ────────────────────────────────────────────

def _ai_provider() -> str:
    """從 session_state 讀取 provider；預設 openai。"""
    return st.session_state.get("ai_provider", "openai")

def _ai_model() -> str:
    """從 session_state 讀取 model；依 provider 給預設值。"""
    provider = _ai_provider()
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "claude": "claude-haiku-4-5-20251001",
    }
    return st.session_state.get("ai_model") or defaults.get(provider, "gpt-4o-mini")


def _call_ai(system_message: str, user_prompt: str, ai_provider: str = "", ai_model: str = "") -> str:
    """統一 AI 呼叫介面，回傳模型的純文字回應。"""
    provider = ai_provider or _ai_provider()
    model    = ai_model or _ai_model()

    if provider == "openai":
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("尚未設定 OPENAI_API_KEY。")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_prompt},
            ]
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("尚未設定 GEMINI_API_KEY。")
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_message,
        )
        resp = gm.generate_content(user_prompt)
        return resp.text

    elif provider == "claude":
        import anthropic
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("尚未設定 ANTHROPIC_API_KEY。")
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_message,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    else:
        raise RuntimeError(f"不支援的 AI Provider：{provider}")


# ── 詞彙索引輔助 ──────────────────────────────────────────

def _safe_int(value):
    try:
        return int(str(value).strip())
    except:
        return None


def prepare_study_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "term", "reading", "meaning", "pos", "note"])
    df = df.copy().fillna("")
    needed = ["code", "term", "reading", "meaning", "pos", "note"]
    for col in needed:
        if col not in df.columns:
            df[col] = ""
    df = df[needed]
    df = df[df["term"].astype(str).str.strip() != ""]
    df["code_num"] = df["code"].apply(_safe_int)
    df = df[df["code_num"].notna()]
    df["code_num"] = df["code_num"].astype(int)
    df = df.sort_values("code_num").reset_index(drop=True)
    return df


def get_allowed_vocab(df: pd.DataFrame, current_code: int):
    # 若 df 已含 code_num（已 prepare 過），直接過濾，避免重複 copy/sort
    if "code_num" not in df.columns:
        df = prepare_study_df(df)
    return df[df["code_num"] <= current_code]


def get_current_row(df: pd.DataFrame, index: int):
    # 若 df 已含 code_num（已 prepare 過），直接取列，避免重複 copy/sort
    if "code_num" not in df.columns:
        df = prepare_study_df(df)
    if df.empty:
        return None
    if index >= len(df):
        index = len(df) - 1
    return df.iloc[index].to_dict()


def get_next_index(df: pd.DataFrame, index: int):
    df = prepare_study_df(df)
    if index + 1 >= len(df):
        return len(df) - 1
    return index + 1


def get_prev_index(df: pd.DataFrame, index: int):
    if index - 1 < 0:
        return 0
    return index - 1


def _extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}


def _grammar_has_focus_bullets(grammar: str) -> bool:
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


def _parse_grammar_parts(grammar: str) -> list[dict]:
    parts = []
    raw = str(grammar or "")
    for p in [x.strip() for x in raw.split("+") if x.strip()]:
        token = p.split("(", 1)[0].split("[", 1)[0].strip()
        pos = ""
        m = re.search(r"\[([^\]:]+)", p)
        if m:
            pos = m.group(1).strip()
        if token:
            parts.append({"token": token, "pos": pos, "raw": p})
    return parts


def _korean_particle_explanation(token: str) -> str:
    m = {
        "은": "主題/對比助詞，表示「就……而言」",
        "는": "主題/對比助詞，表示「就……而言」",
        "이": "主格助詞，標示動作或狀態的主語",
        "가": "主格助詞，標示動作或狀態的主語",
        "을": "受格助詞，標示動作直接作用的對象",
        "를": "受格助詞，標示動作直接作用的對象",
        "에": "方向/時間/靜態位置標記，依語境表示「到、在、於」",
        "에서": "動作發生位置，表示「在……做」",
        "로": "方向/手段標記，表示「往、用、作為」",
        "으로": "方向/手段標記，表示「往、用、作為」",
        "와": "並列助詞，表示「和」",
        "과": "並列助詞，表示「和」",
        "하고": "並列助詞，表示「和」",
        "랑": "並列助詞，表示「和」",
        "도": "添加助詞，表示「也」",
        "만": "限定助詞，表示「只」",
        "의": "屬格助詞，表示「的」",
        "에게": "與格助詞，表示「給、對」",
        "한테": "與格助詞，表示「給、對」",
        "부터": "起點助詞，表示「從」",
        "까지": "終點助詞，表示「到」",
    }
    return m.get(token, "此助詞用來標示句中語法關係")


def _build_specific_rule_notes(grammar: str) -> list[str]:
    parts = _parse_grammar_parts(grammar)
    if not parts:
        return ["• 目標詞：本句以最基本可理解形式呈現，並依語境保持自然。"]

    verb = next((p for p in parts if "動詞" in p["pos"] or "形容詞" in p["pos"]), None)
    particles = [p for p in parts if "助詞" in p["pos"] or "介詞" in p["pos"] or "連接詞" in p["pos"] or "冠詞" in p["pos"]]
    nouns = [p for p in parts if "名詞" in p["pos"]]

    rules = []
    if verb:
        t = verb["token"]
        if re.search(r"(다|る|ir|ar|er)$", t):
            rules.append(f"• {t}：這裡使用原形/辭書形，是因為本句採簡潔敘述句式，未加入過去、敬語或否定標記。")
        else:
            rules.append(f"• {t}：這裡不是原形，而是依本句語氣與時態需求做的詞形變化。")

    if particles:
        for p in particles[:2]:
            token = p["token"]
            explanation = _korean_particle_explanation(token)
            rules.append(f"• {token}：{explanation}；接在前一個詞後，讓該詞在本句取得對應語法角色。")

    # 若句中有格變化線索，補充一條明確說明（跨語言）
    case_markers = [p for p in parts if any(k in p["pos"] for k in ("主格", "受格", "與格", "屬格", "奪格", "工具格"))]
    if case_markers:
        cm = case_markers[0]
        rules.append(f"• {cm['token']}：本句採此格標記是為了表達該名詞在句中的格功能，而不是詞彙本體改義。")

    # 若仍不足，補名詞連接理由
    if len(rules) < 3 and nouns:
        n = nouns[0]["token"]
        rules.append(f"• {n}：名詞本身通常不做形態變化，主要透過後接標記來表達語法功能。")

    return rules[:4]


def _ensure_grammar_focus_bullets(grammar: str) -> str:
    g = _strip_focus_section(str(grammar or "")).strip()
    if not g:
        g = "（未提供逐詞拆解）"

    # 從拆解中抓出詞形資訊，補成「有答案」的具體解說
    parsed = _parse_grammar_parts(g)
    verb_or_adj = next((p["raw"] for p in parsed if "動詞" in p["pos"] or "形容詞" in p["pos"]), "")
    particle_like = next((p["raw"] for p in parsed if "助詞" in p["pos"] or "介詞" in p["pos"] or "連接詞" in p["pos"] or "冠詞" in p["pos"]), "")

    # 若沒有變化規則區塊，直接補逐詞規則（不再補「文法重點」）
    if "變化規則" not in g:
        g = g.rstrip() + "\n變化規則："

    if "變化規則" not in g:
        specific_rules = _build_specific_rule_notes(g)
        if "變化規則：" in g:
            g = g.rstrip() + "\n" + "\n".join(specific_rules)
        else:
            g = f"{g}\n變化規則：\n" + "\n".join(specific_rules)
    return g


def _extract_codes_from_reason(reason: str) -> list[int]:
    text = str(reason or "")
    m = re.search(r"\[(.*?)\]", text)
    if not m:
        return []
    items = re.findall(r"-?\d+", m.group(1))
    return [int(x) for x in items]


def _reason_mentions_higher_code(reason: str) -> bool:
    text = str(reason or "")
    keys = (
        "detected higher-code vocab",
        "reported disallowed vocab codes",
        "too many higher-code vocab codes",
        "too many higher-code vocab",
        "higher-code",
    )
    return any(k in text for k in keys)


def _terms_by_codes(codes: list[int], vocab_items: list | None) -> list[str]:
    if not vocab_items or not codes:
        return []
    code_set = set(codes)
    terms = []
    for v in vocab_items:
        c = str(v.get("code", "")).strip()
        if c.lstrip("-").isdigit() and int(c) in code_set:
            t = str(v.get("term", "")).strip()
            if t:
                terms.append(t)
    return sorted(set(terms))


def _infer_target_code(current_term: str, allowed_vocab: list | None) -> int | None:
    if not allowed_vocab:
        return None
    term = str(current_term or "").strip()
    for v in allowed_vocab:
        if str(v.get("term", "")).strip() == term:
            c = str(v.get("code", "")).strip()
            if c.lstrip("-").isdigit():
                return int(c)
    return None


def _fallback_sentence_payload(
    language: str,
    current_term: str,
    term_reading: str = "",
    term_meaning: str = "",
    allowed_vocab: list | None = None,
) -> dict:
    # 保留函式簽名以避免其他呼叫點炸掉；但不再輸出不自然保底句。
    # 由上層改為拋錯，讓 UI 顯示「生成失敗」而非假句。
    target_code = _infer_target_code(current_term, allowed_vocab)
    return {
        "sentence": "",
        "reading": str(term_reading or "").strip(),
        "translation": "",
        "grammar": f"{current_term}[目標詞彙]",
        "vocab_codes": [target_code] if target_code is not None else [],
    }


def _generate_grammar_explanation(
    language: str,
    sentence: str,
    reading: str,
    translation: str,
    ai_provider: str = "",
    ai_model: str = "",
) -> str:
    if not str(sentence or "").strip():
        return _ensure_grammar_focus_bullets("")

    system_message = (
        "You are a precise grammar explainer for language learners. "
        "Write grammar in Traditional Chinese only. "
        "Use actual words from the sentence; no placeholders. "
        "Output JSON only."
    )

    if language == "korean":
        prompt = f"""請針對下面韓文句子生成文法解說，只輸出 JSON：
{{"grammar":"..."}}

句子：{sentence}
羅馬拼音：{reading}
翻譯：{translation}

格式要求：
1) 先做逐詞拆解：實際詞(讀音)[詞性:意思] + ...
2) 接著寫「變化規則：」並用 1-3 個條列。
- 每條必須以句中實際詞開頭。
- 優先解釋：動詞/形容詞為什麼是這個詞形（或為何不變）、助詞是什麼意思與為何能這樣接。
- 若有格標記，也要說明為何使用該格。
3) grammar 欄位必須是多行純文字，不要輸出 Python/JSON 字典格式。"""
    elif language == "japanese":
        prompt = f"""請針對下面日文句子生成文法解說，只輸出 JSON：
{{"grammar":"..."}}

句子：{sentence}
讀音：{reading}
翻譯：{translation}

格式要求：
1) 先做逐詞拆解：実際の単語(よみ)[品詞: 意思] + ...
2) 接著寫「變化規則：」並用 1-3 個條列。
- 每條必須以句中實際詞開頭。
- 優先解釋：動詞/形容詞為何採此形、助詞/接續為何這樣接。
3) grammar 欄位必須是多行純文字，不要輸出 Python/JSON 字典格式。"""
    else:
        prompt = f"""請針對下面句子生成文法解說，只輸出 JSON：
{{"grammar":"..."}}

句子：{sentence}
翻譯：{translation}

格式要求：
1) 先做逐詞拆解：實際詞[詞性:意思] + ...
2) 接著寫「變化規則：」並用 1-3 個條列。
- 每條必須以句中實際詞開頭。
- 解釋該詞在本句中的變化或語法功能。
3) grammar 欄位必須是多行純文字，不要輸出 Python/JSON 字典格式。"""

    for _ in range(2):
        try:
            content = _call_ai(system_message, prompt, ai_provider=ai_provider, ai_model=ai_model)
            data = _extract_json(content)
            g = str(data.get("grammar", "")).strip()
            if g:
                g = _normalize_grammar_text(g)
                return _ensure_grammar_focus_bullets(g)
        except Exception:
            pass

    # 保底：至少給出規則區塊，避免空白
    return _ensure_grammar_focus_bullets(f"{sentence}")


def _normalize_grammar_text(grammar: str) -> str:
    """
    清理模型偶爾輸出的「字典樣式」字串，轉為可讀段落。
    """
    g = str(grammar or "").strip()
    if not g:
        return g
    # 把 {'逐詞拆解': '...', '變化規則': '...'} 這類輸出轉為純文字
    if g.startswith("{") and g.endswith("}"):
        text = g.strip("{}").replace("'", "")
        text = text.replace('"', "")
        text = text.replace("逐詞拆解:", "逐詞拆解：")
        text = text.replace("變化規則:", "\n變化規則：")
        text = text.replace(", 變化規則", "\n變化規則")
        text = text.replace(", ", "\n")
        return text.strip()
    return g


def _rule_level_by_code(current_code: int, review_mode: bool = False) -> str:
    """
    逐步收緊：
    - relax0: <=300 幾乎不限制
    - relax1: 301~700 允許最多 3 個高編號詞
    - relax2: 701~1000 允許最多 2 個高編號詞
    - strict1: 1001~2000 允許最多 1 個高編號詞
    - strict2: >2000 嚴格 0 個
    """
    if review_mode:
        return "strict2"
    c = int(current_code or 0)
    if c <= 300:
        return "relax0"
    if c <= 700:
        return "relax1"
    if c <= 1000:
        return "relax2"
    if c <= 2000:
        return "strict1"
    return "strict2"


def _sentence_length_limit_by_code(current_code: int, review_mode: bool = False) -> int:
    """
    編號越小，句子越短；逐步放寬。
    CJK 用字元數，其他語言用單詞數。
    """
    if review_mode:
        return 7
    c = int(current_code or 0)
    if c <= 120:
        return 6
    if c <= 300:
        return 8
    if c <= 700:
        return 10
    if c <= 1000:
        return 12
    if c <= 2000:
        return 14
    return 16


def _count_sentence_units(sentence: str, language: str) -> int:
    text = str(sentence or "").strip()
    if not text:
        return 0
    if language in ("japanese", "korean", "chinese"):
        return len(re.sub(r"\s", "", text))
    return len(text.split())


def _normalize_text_for_match(text: str, language: str) -> str:
    if language in ("japanese", "korean", "chinese"):
        return text
    return text.lower()


def _find_vocab_hits(sentence: str, vocab_items: list, language: str) -> list:
    if not sentence or not vocab_items:
        return []
    haystack = _normalize_text_for_match(sentence, language)
    hits = []
    for item in sorted(vocab_items, key=lambda x: len(str(x.get("term", ""))), reverse=True):
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        needle = _normalize_text_for_match(term, language)
        if needle and needle in haystack:
            hits.append(item)
    return hits


def _is_function_word(item: dict) -> bool:
    pos = str(item.get("pos", "")).strip().lower()
    if not pos:
        return False
    function_pos_keywords = (
        "助詞", "助动词", "助動詞", "連接詞", "接続詞", "冠詞", "介詞", "前置詞", "後置詞",
        "代名詞", "語尾", "詞尾", "접속", "조사", "어미", "보조", "관형사", "감탄사", "대명사",
        "particle", "auxiliary", "article", "preposition", "conjunction", "pronoun",
        "determiner", "postposition", "suffix",
    )
    return any(k in pos for k in function_pos_keywords)


def _is_ignorable_disallowed_item(item: dict, language: str) -> bool:
    if _is_function_word(item):
        return True
    term = str(item.get("term", "")).strip()
    # CJK 單字元常見於斷詞/詞素誤判，視為可忽略
    if language in ("japanese", "korean", "chinese") and len(term) <= 1:
        return True
    return False


def _validate_vocab_usage(
    sentence: str,
    vocab_codes: list,
    language: str,
    current_code: int,
    allowed_vocab: list = None,
    full_vocab: list = None,
    review_mode: bool = False,
) -> tuple[bool, str]:
    if not sentence.strip():
        return False, "empty sentence"

    allowed_codes = {
        int(v["code"]) for v in (allowed_vocab or [])
        if str(v.get("code", "")).strip().lstrip("-").isdigit()
    }
    level = _rule_level_by_code(current_code, review_mode=review_mode)
    max_higher_code_words = {
        "relax0": 99,   # 等同幾乎不限制（仍保留非空句檢查）
        "relax1": 3,
        "relax2": 2,
        "strict1": 1,
        "strict2": 0,
    }.get(level, 0)
    reported_disallowed = sorted({c for c in vocab_codes if c not in allowed_codes})
    if reported_disallowed:
        if len(reported_disallowed) > max_higher_code_words:
            return False, f"too many higher-code vocab codes: {reported_disallowed}"

    # 在較鬆等級時，優先採信模型結構化回傳，避免 CJK 子字串誤判。
    if vocab_codes and level in ("relax0", "relax1", "relax2"):
        return True, ""

    if not full_vocab:
        return True, ""

    disallowed_items = [
        v for v in full_vocab
        if (
            str(v.get("code", "")).strip().lstrip("-").isdigit()
            and int(v["code"]) > current_code
            and not _is_ignorable_disallowed_item(v, language)
        )
    ]
    matched_disallowed = _find_vocab_hits(sentence, disallowed_items, language)
    matched_codes = sorted({int(v["code"]) for v in matched_disallowed})

    if matched_codes:
        if len(matched_codes) > max_higher_code_words:
            return False, f"detected higher-code vocab in sentence: {matched_codes}"

    return True, ""


# ── 例句生成 ──────────────────────────────────────────────

def _generate_example_sentence_staged(
    language: str,
    current_term: str,
    term_meaning: str = "",
    term_reading: str = "",
    term_pos: str = "",
    current_code: int = 0,
    review_mode: bool = False,
    allowed_terms: list | None = None,
    allowed_vocab: list | None = None,
    full_vocab: list | None = None,
    ai_provider: str = "",
    ai_model: str = "",
) -> dict:
    """分級收緊模式：先鬆後嚴，避免一開始就卡死。"""
    level = _rule_level_by_code(current_code, review_mode=review_mode)
    max_len = _sentence_length_limit_by_code(current_code, review_mode=review_mode)
    unit_label = "字元（不含空白）" if language in ("japanese", "korean", "chinese") else "單詞"
    level_hint_map = {
        "relax0": "規則等級 L0：先求自然可用，不做詞彙限制。",
        "relax1": "規則等級 L1：儘量用已學詞，最多可用 3 個高編號內容詞。",
        "relax2": "規則等級 L2：儘量用已學詞，最多可用 2 個高編號內容詞。",
        "strict1": "規則等級 L3：最多可用 1 個高編號內容詞。",
        "strict2": "規則等級 L4：只用已學內容詞。",
    }
    level_hint = level_hint_map.get(level, "規則等級：一般")

    vocab_list = ""
    if allowed_vocab:
        vocab_list = "\n".join(f"- [{v['code']}] {v['term']}" for v in allowed_vocab[:500])
    elif allowed_terms:
        vocab_list = "\n".join(f"- {t}" for t in allowed_terms[:500])

    system_message = (
        "You are a language learning assistant. "
        "Write one short, natural, real-usage sentence. "
        "Output JSON only."
    )

    if language == "korean":
        prompt = f"""請用韓文造一句自然、真實可用的短句，且必須包含目標詞。
目標詞：{current_term}
目標詞資訊：詞性={term_pos}，意思={term_meaning}，讀音={term_reading}
{level_hint}
句長限制：最多 {max_len} {unit_label}。
已學詞表（可優先使用）：
{vocab_list}

只輸出 JSON：
{{"sentence":"...","reading":"全句 Revised Romanization","translation":"繁體中文","vocab_codes":[整數代碼]}}"""
    elif language == "japanese":
        prompt = f"""請用日文造一句自然、真實可用的短句，且必須包含目標詞。
目標詞：{current_term}
目標詞資訊：詞性={term_pos}，意思={term_meaning}，讀音={term_reading}
{level_hint}
句長限制：最多 {max_len} {unit_label}。
已學詞表（可優先使用）：
{vocab_list}

只輸出 JSON：
{{"sentence":"...","reading":"ひらがな","translation":"繁體中文","vocab_codes":[整數代碼]}}"""
    else:
        prompt = f"""請造一句自然、真實可用的短句，且必須包含目標詞。
目標詞：{current_term}
目標詞資訊：詞性={term_pos}，意思={term_meaning}
{level_hint}
句長限制：最多 {max_len} {unit_label}。
已學詞表（可優先使用）：
{vocab_list}

只輸出 JSON：
{{"sentence":"...","reading":"","translation":"繁體中文","vocab_codes":[整數代碼]}}"""

    last_reason = "unknown"
    for _ in range(6):
        try:
            content = _call_ai(system_message, prompt, ai_provider=ai_provider, ai_model=ai_model)
            data = _extract_json(content)
            sentence = str(data.get("sentence", "")).strip()
            if not sentence:
                last_reason = "empty sentence"
                continue
            if _count_sentence_units(sentence, language) > max_len:
                last_reason = f"sentence too long: limit={max_len}{unit_label}"
                continue
            reading = str(data.get("reading", "")).strip()
            translation = str(data.get("translation", "")).strip()
            raw_codes = data.get("vocab_codes", [])
            vocab_codes = [
                int(c) for c in (raw_codes if isinstance(raw_codes, list) else [])
                if str(c).strip().lstrip("-").isdigit()
            ]
            ok, reason = _validate_vocab_usage(
                sentence=sentence,
                vocab_codes=vocab_codes,
                language=language,
                current_code=current_code,
                allowed_vocab=allowed_vocab,
                full_vocab=full_vocab,
                review_mode=review_mode,
            )
            if not ok:
                last_reason = reason
                continue
            grammar = _generate_grammar_explanation(
                language=language,
                sentence=sentence,
                reading=reading,
                translation=translation,
                ai_provider=ai_provider,
                ai_model=ai_model,
            )
            return {
                "sentence": sentence,
                "reading": reading,
                "translation": translation,
                "grammar": grammar,
                "vocab_codes": vocab_codes,
            }
        except Exception as e:
            last_reason = str(e)
            continue

    raise RuntimeError(f"例句生成失敗（{level}）：{last_reason}")

def generate_example_sentence(
    language: str,
    current_term: str,
    allowed_terms: list,
    term_meaning: str = "",
    term_reading: str = "",
    term_pos: str = "",
    current_code: int = 0,
    review_mode: bool = False,
    allowed_vocab: list = None,   # [{"code": int, "term": str}, ...] — 有值時詞表附帶編號
    full_vocab: list = None,      # [{"code": int, "term": str}] — 用於生成後驗證
    ai_provider: str = "",
    ai_model: str = "",
) -> dict:
    """
    生成例句。
    - term_meaning / term_reading / term_pos：消除同音異義
    - current_code <= 1000：允許最多 1 個編號較大的內容詞
    - current_code > 1000：嚴格只能用已學內容詞
    - review_mode=True：最多 2 個已學詞彙、句子不超過 7 個字
    - allowed_vocab：帶編號的詞表，AI 將在 vocab_codes 欄位回傳使用到的編號
    """
    # 分級收緊：從最鬆逐步變嚴
    return _generate_example_sentence_staged(
        language=language,
        current_term=current_term,
        term_meaning=term_meaning,
        term_reading=term_reading,
        term_pos=term_pos,
        current_code=current_code,
        review_mode=review_mode,
        allowed_terms=allowed_terms,
        allowed_vocab=allowed_vocab,
        full_vocab=full_vocab,
        ai_provider=ai_provider,
        ai_model=ai_model,
    )

    if allowed_vocab:
        vocab_list = "\n".join(f"- [{v['code']}] {v['term']}" for v in allowed_vocab)
    else:
        vocab_list = "\n".join(f"- {t}" for t in allowed_terms)

    # 同音異義消歧提示
    meaning_hint = ""
    if term_meaning or term_reading or term_pos:
        parts = []
        if term_reading:
            parts.append(f"reading: {term_reading}")
        if term_pos:
            parts.append(f"part of speech: {term_pos}")
        if term_meaning:
            parts.append(f"meaning: {term_meaning}")
        meaning_hint = f" ({', '.join(parts)})"

    # 詞彙使用規則
    if review_mode:
        vocab_rule = (
            "REVIEW MODE CONSTRAINTS — these are HARD rules:\n"
            "1. Use AT MOST 2 content words (nouns, verbs, adjectives, adverbs) from the ALLOWED VOCABULARY list.\n"
            "2. The sentence MUST NOT exceed 7 characters/words "
            "(for Japanese/Korean: count actual kanji/kana/hangul characters, max 7; "
            "for Spanish/other: count words, max 7).\n"
            "3. Grammatical particles, conjugations, articles, pronouns do NOT count toward the 2-word limit.\n"
            "4. If the 7-character limit and naturalness conflict, prioritise the limit — use a shorter form.\n"
        )
    else:
        vocab_rule = (
            "VOCABULARY RULE (GUIDED): "
            "Primary rule: content words (nouns, verbs, adjectives, adverbs) should come from the ALLOWED VOCABULARY list. "
            "Inflected/conjugated forms of learned lemmas are allowed and still count as learned words. "
            "If current code <= 1000 and list-only wording is too unnatural, allow at most ONE higher-code content word. "
            "If current code > 1000, do not use higher-code content words.\n"
        )

    system_message = (
        "You are a language learning assistant that writes natural, idiomatic example sentences. "
        "NATURALNESS IS THE TOP PRIORITY — the sentence must sound like something a native speaker would actually say. "
        "The output sentence must be genuinely usable in real conversation or writing, not a placeholder fragment. "
        f"{vocab_rule}"
        "Grammatical elements required by the language — particles, articles, conjunctions, auxiliary verbs, "
        "verb conjugations, pronouns — are always permitted even if not in the list. "
        "CRITICAL: The TARGET WORD may be a homonym or polysemous word. "
        "You MUST use the word in the EXACT sense given by its Chinese meaning and part of speech. "
        "Do NOT use any other reading, sense, or grammatical category of the word. "
        "For Japanese: if the target word has multiple possible readings (e.g. 生：なま/せい/しょう), "
        "you MUST use ONLY the reading specified. The sentence context must MATCH that specific reading's meaning. "
        "CONTEXTUAL DEMONSTRATION: The sentence should make the word's Chinese meaning naturally inferable "
        "from context — a learner who reads it should understand what the word means. "
        "Do NOT force multiple allowed words into the sentence just because they exist; "
        "use only as many as needed to form a natural, meaningful sentence. "
        "CRITICAL for the 'sentence' field: write ONLY the plain sentence text with NO furigana, "
        "NO parentheses, NO reading annotations of any kind. "
        "For example, write 机の上に物がある。NOT 机（つくえ）の上（うえ）に物（もの）がある。 "
        "CRITICAL for Korean: the 'reading' field MUST contain the full Revised Romanization (RR) "
        "of the entire sentence — every word romanized. Never leave it empty for Korean. "
        "IMPORTANT: The grammar analysis field must be written ENTIRELY in Traditional Chinese (繁體中文). "
        "Use Chinese grammatical terms such as: 名詞、動詞、形容詞、副詞、助詞、主格助詞、受格助詞、"
        "屬格助詞、連接詞、助動詞、代名詞、否定詞 etc. "
        "Do NOT use Korean, Japanese, or English labels in the grammar field. "
        "CRITICAL for the 'grammar' field: use the ACTUAL words from the sentence — never write '語' or "
        "any placeholder. For example if sentence is 春が来る then grammar is: "
        "春(はる)[名詞: 春天] + が[主格助詞] + 来る(くる)[動詞: 來] "
        "After the word-by-word breakdown, append a section titled「文法重點」in Traditional Chinese with EXACTLY 3 bullet points. "
        "Each bullet MUST start with '• ' and MUST quote ACTUAL words from the sentence (not abstract terms). "
        "Bullet 1 must be『活用/時態』: explain why that concrete verb/adjective form is used "
        "(tense, polarity, aspect, politeness, conjugation). "
        "Bullet 2 must be『助詞/連接』: explain why those concrete particles/conjunctions/prepositions/articles are used. "
        "Bullet 3 must be『語氣/情境』: explain sentence mood (statement/question/inference/request) and context effect. "
        "Then append「變化規則」with 1-3 concise bullets in Traditional Chinese. "
        "Each bullet MUST start with an ACTUAL word from this sentence and explain that word's change/function in THIS sentence. "
        "PRIORITY: include at least 1 verb/adjective-form bullet and at least 1 particle/preposition/conjunction bullet when such words exist. "
        "For verbs/adjectives, explicitly explain why this sentence uses this form (or why it stays base form). "
        "For particles/prepositions/conjunctions, explicitly explain meaning and why this connection is valid in this sentence. "
        "If noun case change exists, explicitly explain why that case is chosen. "
        "Do NOT write generic textbook rules. Each rule bullet MUST be an answer statement, not a question. "
        + (
        "CRITICAL for the 'vocab_codes' field: list the integer code numbers "
        "(the [N] prefix in the ALLOWED VOCABULARY list) of every content word you used from that list, "
        "even if you used it in a conjugated or inflected form. "
        "Example: if [5] 가다 appears as 가요, include 5. "
        "Include the target word's code too. Output only codes that actually appear in the sentence. "
        if allowed_vocab else ""
        ) +
        "Respond only with a JSON object — no explanation, no markdown."
    )

    # 句長限制提示（複習模式）
    length_rule_ja = "\n- 【複習モード】句子不超過7個字。" if review_mode else ""
    length_rule_ko = "\n- 【복습 모드】문장은 7자 이내로 작성할 것." if review_mode else ""
    length_rule_es = "\n- 【Review mode】La oración NO debe superar 7 palabras." if review_mode else ""
    content_word_rule_ja = "\n- 【複習モード】リストから使う内容語は最大2語。" if review_mode else ""
    content_word_rule_ko = "\n- 【복습 모드】목록에서 사용하는 내용어는 최대 2개." if review_mode else ""
    content_word_rule_es = "\n- 【Review mode】Usa como máximo 2 palabras de contenido de la lista." if review_mode else ""

    if language == "japanese":
        _reading_constraint = f"読みは必ず「{term_reading}」のみ — 他の読みは絶対に使わないこと。" if term_reading else ""
        _meaning_constraint = f"意味「{term_meaning}」を文脈で自然に示すこと。" if term_meaning else ""
        meaning_line = (
            f"\nTARGET WORDの品詞・意味・読み：{term_pos}「{term_meaning}」（読み：{term_reading}）。"
            f"{_reading_constraint}{_meaning_constraint}"
        ) if (term_meaning or term_reading or term_pos) else ""
        prompt = f"""以下は学習者がこれまでに学んだ語彙リストです。

ALLOWED VOCABULARY（内容語はこのリストから選ぶ）:
{vocab_list}

TARGET WORD（必ずこの単語を含める）: {current_term}{meaning_hint}{meaning_line}

【条件】
- sentenceフィールドは漢字とかなのみ（ふりがな・括弧・読み仮名は絶対に含めない）。
  良い例：机の上に物がある。　悪い例：机（つくえ）の上（うえ）に物（もの）がある。
- TARGET WORDは同音異義語に注意し、指定された品詞・意味で使うこと。
- 文は自然な日本語であること。不自然な語順や不要な詰め込みは禁止。
- 助詞・助動詞・活用語尾・接続詞などの文法要素は自由に使ってよい。
- リストの語を無理に全部使う必要はない。自然な短文を一つだけ作ること。{content_word_rule_ja}{length_rule_ja}
- 【厳守】内容語は必ずリスト内の語のみ。リスト外の内容語は使用禁止。
- grammarフィールドは文中の実際の単語をそのまま使うこと。「語」という文字は絶対に使わない。
- grammarフィールドでは、単語分解のあとに「文法重點：」を付け、繁體中文の條列（各行「• 」開始）で2〜4点説明すること。
  必ず「なぜその活用形なのか」「なぜその助詞・接続詞を使うのか」を含めること。
  文が「春が来る。」なら → 春(はる)[名詞: 春天] + が[主格助詞] + 来る(くる)[動詞: 來]

JSON形式で出力：
{{"sentence":"ふりがななしの日本語文","reading":"ひらがな","translation":"繁體中文翻譯","grammar":"実際の単語(よみ)[品詞: 意思] + ...","vocab_codes":[使用したリストの整数コード]}}"""

    elif language == "korean":
        meaning_line = f"\n대상 단어 품사·의미: {term_pos}「{term_meaning}」（발음: {term_reading}）. 반드시 이 품사·의미로 사용하세요." if (term_meaning or term_pos) else ""
        prompt = f"""아래는 학습자가 지금까지 배운 어휘 목록입니다.

ALLOWED VOCABULARY（내용어는 이 목록에서 선택）:
{vocab_list}

TARGET WORD（반드시 포함）: {current_term}{meaning_hint}{meaning_line}

【조건】
- sentence 필드는 순수한 한국어 텍스트만 (괄호·로마자·발음 표기 절대 포함 금지).
- TARGET WORD는 동음이의어에 주의하여 지정된 품사·의미로 사용할 것.
- 문장은 자연스러운 한국어여야 합니다.
- 내용어（명사·동사·형용사·부사）는 목록에 있는 것을 사용.
- 조사·어미·접속사·보조동사 등 문법 요소는 자유롭게 사용 가능.{content_word_rule_ko}{length_rule_ko}
- 【엄수】내용어는 반드시 목록 안의 단어만 사용. 목록 밖 내용어는 사용 금지.
- grammar 필드는 문장의 실제 단어만 사용（「단어」라는 글자 절대 금지）.
- grammar 필드에서는 단어 분석 뒤에 「文法重點:」 섹션을 추가하고, 각 줄을 '• '로 시작하는繁體中文 불릿 2~4개로 설명할 것.
  반드시 왜 그 활용형을 썼는지, 왜 그 조사·접속 표현을 썼는지를 포함할 것.
  예：나(na)[代名詞: 我] + 는[主格助詞] + 학교(hakgyo)[名詞: 學校] + 에[方向助詞] + 가요(gayo)[動詞: 去]

JSON 출력：
{{"sentence":"한국어 문장（괄호 없음）","reading":"全句羅馬字（Revised Romanization）","translation":"繁體中文翻譯","grammar":"실제단어(romaja)[品詞: 意思] + ...","vocab_codes":[사용한 목록의 정수 코드]}}"""

    else:
        meaning_line = f"\nSignificado específico de la palabra objetivo: {term_pos} '{term_meaning}'. Usa la palabra en ESTE sentido exacto, no en ningún otro." if (term_meaning or term_pos) else ""
        prompt = f"""El siguiente es el vocabulario que el aprendiz ha estudiado hasta ahora.

ALLOWED VOCABULARY（las palabras de contenido deben venir de esta lista）:
{vocab_list}

TARGET WORD（debe aparecer en la oración）: {current_term}{meaning_hint}{meaning_line}

【Condiciones】
- La oración debe sonar natural. No fuerces palabras innecesarias.
- Usa la palabra objetivo en el sentido exacto indicado; no uses homónimos.
- Las palabras de contenido（sustantivos, verbos, adjetivos, adverbios）deben venir de la lista.
- Los elementos gramaticales（artículos, preposiciones, conjunciones, pronombres, conjugaciones）son libres.
- No es necesario usar todas las palabras. Una oración corta y natural es mejor.{content_word_rule_es}{length_rule_es}
- 【OBLIGATORIO】Todas las palabras de contenido deben salir de la lista. No uses contenido fuera de la lista.
- grammar: usa siempre las palabras reales de la oración（nunca escribas «palabra»）.
- Después del desglose, añade una sección titulada 「文法重點」 con 2-4 viñetas en chino tradicional; cada viñeta debe empezar con '• '.
  Debes explicar por qué se usa esa forma verbal/adjetival y por qué se usan esas preposiciones, conjunciones o artículos.
  Ej：Yo[代名詞: 我] + como(comer)[動詞: 吃] + pan[名詞: 麵包]

Responde solo con JSON：
{{"sentence":"oración en español","reading":"","translation":"繁體中文翻譯","grammar":"palabraReal[品詞: 意思] + ...","vocab_codes":[códigos enteros usados de la lista]}}"""

    last_reason = "unknown validation failure"
    banned_terms = set()
    for _ in range(6):
        dynamic_prompt = prompt
        if banned_terms:
            dynamic_prompt += (
                "\n\n【額外嚴格限制】\n"
                f"- 以下詞彙絕對禁止出現在 sentence：{', '.join(sorted(banned_terms))}\n"
                "- 若會用到上述詞，請改寫成只使用 ALLOWED VOCABULARY 的句子。"
            )

        try:
            content = _call_ai(system_message, dynamic_prompt, ai_provider=ai_provider, ai_model=ai_model)
        except Exception as e:
            last_reason = f"ai_call_failed: {e}"
            continue
        data = _extract_json(content)
        raw_codes = data.get("vocab_codes", [])
        vocab_codes = [int(c) for c in (raw_codes if isinstance(raw_codes, list) else []) if str(c).strip().lstrip("-").isdigit()]
        sentence = data.get("sentence", "")
        ok, reason = _validate_vocab_usage(
            sentence=sentence,
            vocab_codes=vocab_codes,
            language=language,
            current_code=current_code,
            allowed_vocab=allowed_vocab,
            full_vocab=full_vocab,
            review_mode=review_mode,
        )
        if ok:
            normalized_grammar = _generate_grammar_explanation(
                language=language,
                sentence=sentence,
                reading=data.get("reading", ""),
                translation=data.get("translation", ""),
                ai_provider=ai_provider,
                ai_model=ai_model,
            )
            return {
                "sentence":    sentence,
                "reading":     data.get("reading", ""),
                "translation": data.get("translation", ""),
                "grammar":     normalized_grammar,
                "vocab_codes": vocab_codes,
            }
        last_reason = reason
        if _reason_mentions_higher_code(reason):
            bad_codes = _extract_codes_from_reason(reason)
            banned_terms.update(_terms_by_codes(bad_codes, full_vocab))

    raise RuntimeError(f"生成不到自然可用例句：{last_reason}")


# ── FSI 搭配 / 句型變化生成 ──────────────────────────────

def generate_fsi_sentence(
    language: str,
    current_term: str,
    drill_type: str,               # "substitution" | "transformation"
    term_meaning: str = "",
    term_reading: str = "",
    term_pos: str = "",
    current_code: int = 0,
    allowed_vocab: list = None,    # [{"code": int, "term": str}]
    prev_drill_notes: list = None, # previously generated notes (to avoid repetition)
    full_vocab: list = None,       # [{"code": int, "term": str}] — 用於生成後驗證
    ai_provider: str = "",
    ai_model: str = "",
) -> dict:
    """
    FSI drill sentence generator.
    substitution : same target word, different collocations / partner words each time.
    transformation: same target word in a different grammatical form each time.
    Returns: {sentence, reading, translation, grammar, drill_note, vocab_codes}
    """
    if allowed_vocab:
        vocab_list = "\n".join(f"- [{v['code']}] {v['term']}" for v in allowed_vocab)
    else:
        vocab_list = ""

    # Disambiguation hint
    meaning_hint = ""
    if term_meaning or term_reading or term_pos:
        parts = []
        if term_reading: parts.append(f"reading: {term_reading}")
        if term_pos:     parts.append(f"part of speech: {term_pos}")
        if term_meaning: parts.append(f"meaning: {term_meaning}")
        meaning_hint = f" ({', '.join(parts)})"

    # Avoid-repetition clause
    prev_str = ""
    if prev_drill_notes:
        joined = " | ".join(prev_drill_notes[-5:])
        prev_str = f" ALREADY USED — do NOT repeat these: [{joined}]."

    vocab_rule = (
        "Content words (nouns, verbs, adjectives, adverbs) MUST come from the ALLOWED VOCABULARY list only. "
        "Do NOT use content words outside the list. "
        "Grammatical elements (particles, articles, conjunctions, conjugations, pronouns) are always free. "
    ) if allowed_vocab else ""

    vocab_codes_instr = (
        "CRITICAL for 'vocab_codes': list the integer codes of every content word used from the "
        "ALLOWED VOCABULARY list, even if in conjugated/inflected form. "
    ) if allowed_vocab else ""

    if drill_type == "substitution":
        drill_desc = (
            "FSI SUBSTITUTION DRILL: Write a short, natural sentence using the TARGET WORD "
            "combined with a DIFFERENT collocation each time — vary the objects, subjects, adverbs, "
            "or surrounding context that typically appear with this word. "
            f"{prev_str} "
            "In 'drill_note', describe in Traditional Chinese (繁體中文) the collocation/context used, "
            "e.g. '搭配：每天 + 吃' or '場景：餐廳'."
        )
    else:  # transformation
        drill_desc = (
            "FSI TRANSFORMATION DRILL: Write a short, natural sentence using the TARGET WORD "
            "in a DIFFERENT GRAMMATICAL FORM each time. "
            "Vary tense, polarity, aspect, mood, voice, or other grammatical dimensions "
            "(e.g. past, negative, progressive, conditional, question, potential, causative, passive, imperative). "
            f"{prev_str} "
            "In 'drill_note', name the grammatical form used in Traditional Chinese (繁體中文), "
            "e.g. '過去形', '否定形', '疑問句', '可能形', '條件句 (たら)', '使役形'."
        )

    system_message = (
        "You are a language learning assistant specializing in FSI drill methods. "
        f"{drill_desc} "
        f"{vocab_rule}"
        "The sentence must be SHORT (no more than 10 characters/words) and NATURAL. "
        "CRITICAL: The TARGET WORD may be a homonym or polysemous word. "
        "You MUST use it in the EXACT sense given by its Chinese meaning and part of speech. "
        "For Japanese: if the word has multiple possible readings, use ONLY the reading specified — "
        "the sentence context must match that reading's specific meaning. "
        "CRITICAL for 'sentence': plain text only — no furigana, no parentheses, no reading annotations. "
        "CRITICAL for Korean: 'reading' MUST contain full Revised Romanization of the entire sentence. "
        "Grammar analysis ('grammar') must be ENTIRELY in Traditional Chinese (繁體中文) — "
        "use terms like 名詞、動詞、形容詞、副詞、助詞、主格助詞、否定詞 etc. "
        "Use ACTUAL words from the sentence in 'grammar' — never write '語' or any placeholder. "
        "After the breakdown, append「文法重點」with EXACTLY 3 Traditional Chinese bullet points. "
        "Each bullet MUST start with '• ' and MUST cite ACTUAL words from the sentence. "
        "Bullet 1:『活用/時態』for concrete verb/adjective form choice. "
        "Bullet 2:『助詞/連接』for concrete particles/conjunctions/prepositions/articles. "
        "Bullet 3:『語氣/情境』for mood and context effect. "
        "Then append「變化規則」with 1-3 concise Traditional Chinese bullets. "
        "Each bullet MUST start with an ACTUAL word from this sentence and explain that word's change/function in THIS sentence (no generic rules). "
        "PRIORITY: include at least 1 verb/adjective-form bullet and at least 1 particle/preposition/conjunction bullet when available. "
        f"{vocab_codes_instr}"
        "Respond only with a JSON object — no explanation, no markdown."
    )

    vocab_section = f"\nALLOWED VOCABULARY:\n{vocab_list}" if vocab_list else ""

    if language == "japanese":
        _r = f"読みは必ず「{term_reading}」のみ使うこと。" if term_reading else ""
        _m = f"意味「{term_meaning}」が文脈から伝わる文を作ること。" if term_meaning else ""
        fsi_meaning_line = (
            f"\nTARGET WORDの品詞・意味・読み：{term_pos}「{term_meaning}」（読み：{term_reading}）。{_r}{_m}"
        ) if (term_meaning or term_reading or term_pos) else ""
        prompt = f"""TARGET WORD: {current_term}{meaning_hint}{fsi_meaning_line}{vocab_section}

JSON形式で出力：
{{"sentence":"日本語文（ふりがななし）","reading":"ひらがな","translation":"繁體中文翻譯","grammar":"単語(よみ)[品詞: 意思] + ...","drill_note":"繁體中文說明","vocab_codes":[整数コード]}}"""

    elif language == "korean":
        _m_ko = f"\n대상 단어 품사·의미·발음: {term_pos}「{term_meaning}」（발음: {term_reading}）. 반드시 이 의미로 사용하고, 문맥에서 의미 「{term_meaning}」이 자연스럽게 전달되도록 할 것." if (term_meaning or term_reading or term_pos) else ""
        prompt = f"""TARGET WORD: {current_term}{meaning_hint}{_m_ko}{vocab_section}

JSON 출력：
{{"sentence":"한국어 문장（괄호 없음）","reading":"全句 Revised Romanization","translation":"繁體中文翻譯","grammar":"단어(romaja)[品詞: 意思] + ...","drill_note":"繁體中文說明","vocab_codes":[정수 코드]}}"""

    else:
        _m_es = f"\nSignificado exacto: {term_pos} '{term_meaning}'. Usa la palabra solo en este sentido; el contexto debe hacer que el significado '{term_meaning}' sea comprensible." if (term_meaning or term_pos) else ""
        prompt = f"""TARGET WORD: {current_term}{meaning_hint}{_m_es}{vocab_section}

Responde solo con JSON：
{{"sentence":"oración en español","reading":"","translation":"繁體中文翻譯","grammar":"palabraReal[品詞: 意思] + ...","drill_note":"繁體中文說明","vocab_codes":[códigos enteros]}}"""

    last_reason = "unknown validation failure"
    banned_terms = set()
    for _ in range(6):
        dynamic_prompt = prompt
        if banned_terms:
            dynamic_prompt += (
                "\n\n【額外嚴格限制】\n"
                f"- 以下詞彙絕對禁止出現在 sentence：{', '.join(sorted(banned_terms))}\n"
                "- 若會用到上述詞，請改寫成只使用 ALLOWED VOCABULARY 的句子。"
            )

        try:
            content = _call_ai(system_message, dynamic_prompt, ai_provider=ai_provider, ai_model=ai_model)
        except Exception as e:
            last_reason = f"ai_call_failed: {e}"
            continue
        data = _extract_json(content)
        raw_codes = data.get("vocab_codes", [])
        vocab_codes = [int(c) for c in (raw_codes if isinstance(raw_codes, list) else [])
                       if str(c).strip().lstrip("-").isdigit()]
        sentence = data.get("sentence", "")
        ok, reason = _validate_vocab_usage(
            sentence=sentence,
            vocab_codes=vocab_codes,
            language=language,
            current_code=current_code,
            allowed_vocab=allowed_vocab,
            full_vocab=full_vocab,
            review_mode=False,
        )
        if ok:
            normalized_grammar = _generate_grammar_explanation(
                language=language,
                sentence=sentence,
                reading=data.get("reading", ""),
                translation=data.get("translation", ""),
                ai_provider=ai_provider,
                ai_model=ai_model,
            )
            return {
                "sentence":    sentence,
                "reading":     data.get("reading", ""),
                "translation": data.get("translation", ""),
                "grammar":     normalized_grammar,
                "drill_note":  data.get("drill_note", ""),
                "vocab_codes": vocab_codes,
            }
        last_reason = reason
        if _reason_mentions_higher_code(reason):
            bad_codes = _extract_codes_from_reason(reason)
            banned_terms.update(_terms_by_codes(bad_codes, full_vocab))

    raise RuntimeError(f"FSI 句子生成失敗（無自然可用句）：{last_reason}")


# ── 句型重組生成 ──────────────────────────────────────────

def generate_recombination_sentence(
    language: str,
    target_words: list,
    all_allowed_terms: list,
    grammar_instruction: str = "",
    all_allowed_vocab: list = None,  # [{"code": int, "term": str}, ...]
) -> dict:
    """
    Generate ONE natural sentence that weaves ALL target_words together.
    target_words : [{"term": str, "meaning": str, "reading": str}, ...]
    """
    if all_allowed_vocab:
        vocab_list = "\n".join(f"- [{v['code']}] {v['term']}" for v in all_allowed_vocab)
    else:
        vocab_list = "\n".join(f"- {t}" for t in all_allowed_terms)
    targets_fmt = "\n".join(
        f"  • {w['term']} （{w.get('reading', '')}）＝ {w.get('meaning', '')}"
        for w in target_words
    )
    pattern_line = f"\nSENTENCE TYPE: {grammar_instruction}" if grammar_instruction else ""

    system_message = (
        "You are a language learning assistant. "
        "Your task is to compose ONE short, natural sentence that includes ALL of the TARGET WORDS. "
        "NATURALNESS IS THE TOP PRIORITY — the sentence must sound like something a native speaker would say. "
        "CRITICAL: Each TARGET WORD comes with a specific meaning. You MUST use every word in its given sense. "
        "Do NOT use homonyms or other senses of the word. "
        "STRICT LENGTH RULE: The sentence MUST be 7 content+function words or fewer. Keep it as short as possible. "
        "Do NOT pad the sentence. A 2–4 word sentence is ideal. "
        "Use only content words (nouns, verbs, adjectives, adverbs) from the ALLOWED VOCABULARY list "
        "plus the required target words. "
        "Grammatical function words (particles, articles, conjunctions, auxiliary verbs, pronouns, conjugations) "
        "are always permitted. "
        "If a SENTENCE TYPE is specified, the sentence MUST follow that grammatical pattern or mood. "
        "CRITICAL for the 'sentence' field: plain text only — no furigana, no parentheses, no reading annotations. "
        "CRITICAL for Korean: the 'reading' field MUST contain the full Revised Romanization (RR) "
        "of the entire sentence — every word romanized. Never leave it empty for Korean. "
        "CRITICAL for the 'grammar' field: use the ACTUAL words from the sentence, never '語' or any placeholder. "
        "Grammar labels must be ENTIRELY in Traditional Chinese (繁體中文): "
        "名詞、動詞、形容詞、副詞、助詞、主格助詞、受格助詞、否定詞 etc. "
        "After the word-by-word grammar breakdown, append「文法重點」with EXACTLY 3 Traditional Chinese bullet points. "
        "Each bullet MUST start with '• ' and MUST cite ACTUAL words from the sentence. "
        "Bullet 1:『活用/時態』for concrete verb/adjective form choice. "
        "Bullet 2:『助詞/連接』for concrete particles/conjunctions/prepositions/articles. "
        "Bullet 3:『語氣/情境』for mood and context effect. "
        "Then append「變化規則」with 1-3 concise Traditional Chinese bullets. "
        "Each bullet MUST start with an ACTUAL word from this sentence and explain that word's change/function in THIS sentence (no generic rules). "
        "PRIORITY: include at least 1 verb/adjective-form bullet and at least 1 particle/preposition/conjunction bullet when available. "
        + (
        "CRITICAL for the 'vocab_codes' field: list the integer codes "
        "(the [N] prefix in the ALLOWED VOCABULARY list) of every content word used from that list, "
        "including in conjugated/inflected form. "
        if all_allowed_vocab else ""
        ) +
        "Respond only with a JSON object — no explanation, no markdown."
    )

    if language == "japanese":
        prompt = f"""ALLOWED VOCABULARY（内容語はこのリストから選ぶ）:
{vocab_list}

TARGET WORDS（全て必ず文に含める。各単語は指定された意味・品詞で使うこと）:
{targets_fmt}{pattern_line}

【条件】
- 各 TARGET WORD は指定された意味・品詞で使うこと。同音異義語に注意。
- 上の TARGET WORDS を全て自然に含む文を1つ作ること。
- 文は必ず短く（7語以内）。2〜4語が理想。自然であることが最優先。
- SENTENCE TYPE が指定されている場合は、その文法・語気に必ず従うこと（必須）。
- sentenceフィールドは漢字とかなのみ（ふりがな・括弧・読み仮名は含めない）。
- grammarフィールドは文中の実際の単語をそのまま使うこと。「語」は絶対に書かない。
  例：猫(ねこ)[名詞: 貓] + が[主格助詞] + いる[動詞: 在]

JSON形式で出力：
{{"sentence":"ふりがななし","reading":"ひらがな","translation":"繁體中文翻譯","grammar":"実際の単語(よみ)[品詞: 意思] + ...","vocab_codes":[使用したリストの整数コード]}}"""

    elif language == "korean":
        prompt = f"""ALLOWED VOCABULARY（내용어는 이 목록에서 선택）:
{vocab_list}

TARGET WORDS（모두 반드시 포함. 각 단어는 지정된 의미·품사로 사용）:
{targets_fmt}{pattern_line}

【조건】
- 각 TARGET WORD는 지정된 의미·품사로 사용할 것（동음이의어 주의）.
- TARGET WORDS 를 모두 자연스럽게 포함하는 문장 하나 작성.
- 문장은 반드시 짧게（7어 이내, 2〜4어 이상적）. 자연스러움이 최우선.
- SENTENCE TYPE 이 지정되어 있으면 반드시 따를 것.
- sentence 필드는 순수 한국어만（괄호·로마자 없음）.
- grammar 필드는 문장의 실제 단어만 사용.
  예：고양이(goyangi)[名詞: 貓] + 가[主格助詞] + 있어요(isseoyo)[動詞: 在]

JSON 출력：
{{"sentence":"한국어문장","reading":"全句羅馬字（RR）","translation":"繁體中文翻譯","grammar":"실제단어[品詞: 意思] + ...","vocab_codes":[사용한 목록의 정수 코드]}}"""

    else:
        prompt = f"""ALLOWED VOCABULARY（las palabras de contenido deben venir de esta lista）:
{vocab_list}

TARGET WORDS（todas deben aparecer en la oración, en su sentido especificado）:
{targets_fmt}{pattern_line}

【Condiciones】
- Usa cada TARGET WORD en el sentido exacto indicado（no homónimos）.
- Escribe UNA oración natural que incluya TODAS las palabras objetivo.
- La oración DEBE ser corta（7 palabras o menos, idealmente 2–4）.
- Si se especifica un SENTENCE TYPE, la oración DEBE seguir ese patrón（obligatorio）.
- Las palabras de contenido deben venir de la lista permitida.
- grammar: describe cada palabra real con términos en chino tradicional (繁體中文).
  Ej：Yo[代名詞: 我] + como(comer)[動詞: 吃] + pan[名詞: 麵包]

Responde solo con JSON：
{{"sentence":"oración","reading":"","translation":"繁體中文翻譯","grammar":"palabraReal[品詞: 意思] + ...","vocab_codes":[códigos enteros usados de la lista]}}"""

    content = _call_ai(system_message, prompt)
    data = _extract_json(content)
    raw_codes = data.get("vocab_codes", [])
    vocab_codes = [int(c) for c in (raw_codes if isinstance(raw_codes, list) else []) if str(c).strip().lstrip("-").isdigit()]
    normalized_grammar = _generate_grammar_explanation(
        language=language,
        sentence=data.get("sentence", ""),
        reading=data.get("reading", ""),
        translation=data.get("translation", ""),
    )
    return {
        "sentence":    data.get("sentence", ""),
        "reading":     data.get("reading", ""),
        "translation": data.get("translation", ""),
        "grammar":     normalized_grammar,
        "vocab_codes": vocab_codes,
    }


def _count_passage_length(passage: str, language: str) -> int:
    """
    Count passage length in the same unit used for the max_length constraint:
    - CJK languages (japanese / korean): count every non-whitespace character
    - Roman-script languages: count whitespace-separated words
    """
    text = passage.strip()
    if language in ("japanese", "korean", "chinese"):
        # Count all non-whitespace characters (CJK + kana + hangul + punctuation)
        import re
        return len(re.sub(r"\s", "", text))
    else:
        return len(text.split())


def generate_passage(
    language: str,
    passage_type: str,          # "短文" | "故事" | "對話"
    all_allowed_vocab: list,    # [{"code": int, "term": str, "meaning": str, "reading": str}, ...]
    max_length: int = 50,       # max characters (CJK) or words (roman)
) -> dict:
    """
    Generate a short passage (短文/故事/對話) using ONLY the learned vocabulary.
    Strictly enforces max_length; retries once if AI exceeds the limit.
    Returns {
        "passage": str, "reading": str, "translation": str,
        "vocab_notes": [{"word", "reading", "meaning", "pos"}],
        "grammar_notes": str,
        "char_count": int,   # actual length in the counted unit
    }
    """
    vocab_list = "\n".join(
        f"- [{v['code']}] {v['term']}（{v.get('reading','')}: {v.get('meaning','')}）"
        for v in all_allowed_vocab
    )

    type_desc_map = {
        "短文": "a short descriptive paragraph (natural everyday description)",
        "故事": "a mini story with a tiny plot twist and natural characters",
        "對話": "a two-person dialogue formatted as 'A：...\\nB：...'",
    }
    type_desc = type_desc_map.get(passage_type, "a short text")

    is_cjk = language in ("japanese", "korean", "chinese")
    min_length = max(int(max_length * 0.8), 1)   # 目標區間下界（80%）
    if is_cjk:
        length_rule = (
            f"TARGET RANGE: {min_length}–{max_length} non-whitespace characters. "
            f"Count every kana, kanji, hangul, and punctuation mark individually; whitespace does not count. "
            f"AIM TO FILL AS CLOSE TO {max_length} CHARACTERS AS POSSIBLE — do not write a passage that is "
            f"shorter than {min_length} characters. The passage MUST NOT exceed {max_length} characters. "
            f"If your draft exceeds {max_length}, trim sentences; if it is under {min_length}, expand."
        )
        unit_label = "字元（不含空白）"
    else:
        length_rule = (
            f"TARGET RANGE: {min_length}–{max_length} words. "
            f"Count every whitespace-separated token. "
            f"AIM TO FILL AS CLOSE TO {max_length} WORDS AS POSSIBLE — do not write a passage shorter "
            f"than {min_length} words. The passage MUST NOT exceed {max_length} words. "
            f"If your draft exceeds {max_length}, trim; if it is under {min_length}, expand."
        )
        unit_label = "個單詞"

    system_message = (
        "You are a language learning assistant creating immersive reading practice. "
        f"Generate ONE {passage_type} — {type_desc} — in the target language. "
        "STRICT RULES: "
        "(1) Content words (nouns, verbs, adjectives, adverbs) MUST come ONLY from the ALLOWED VOCABULARY list. "
        "Grammatical function words (particles, articles, conjunctions, auxiliaries, pronouns, "
        "common adverbs like very/also/not, numbers) are always permitted. "
        f"(2) LENGTH — THIS IS CRITICAL: {length_rule} "
        "(3) The text must feel completely NATURAL — not a textbook exercise. "
        "(4) For 對話 type: format passage as 'A：...\\nB：...'. "
        "(5) 'passage' field: plain text only, no furigana, no parentheses, no reading annotations. "
        "(6) 'reading' field: full reading of the entire passage. "
        "Japanese → hiragana reading line by line. Korean → full Revised Romanization. "
        "Spanish/French/German/Italian/Portuguese → leave empty string. "
        "(7) 'vocab_notes': list EVERY content word used from the allowed list. "
        "Each entry has: word (surface form), reading, meaning, pos (Traditional Chinese POS label). "
        "(8) 'grammar_notes': in Traditional Chinese (繁體中文), explain 1–3 key grammar patterns used, and explicitly mention "
        "why any verb/adjective changes form the way it does, plus why important conjunctions, particles, or prepositions are used. "
        "(9) Respond ONLY with valid JSON — no markdown, no extra text."
    )

    prompt = f"""ALLOWED VOCABULARY (only content words from this list):
{vocab_list}

PASSAGE TYPE: {passage_type}
LENGTH: aim for {max_length} {'non-whitespace characters' if is_cjk else 'words'} (range {min_length}–{max_length}, must NOT exceed {max_length}).

Output JSON:
{{
  "passage": "原文（無注音，目標接近 {max_length} {'字元' if is_cjk else '個單詞'}，絕不超過）",
  "reading": "完整讀音",
  "translation": "繁體中文翻譯",
  "vocab_notes": [
    {{"word": "單字", "reading": "讀音", "meaning": "意思", "pos": "詞性（繁體中文）"}}
  ],
  "grammar_notes": "文法說明（繁體中文，1-3條，需說明活用原因與連接詞/助詞/介系詞選用原因）"
}}"""

    # ── First attempt ──────────────────────────────────────
    content = _call_ai(system_message, prompt)
    data = _extract_json(content)
    passage_text = data.get("passage", "")
    actual_len = _count_passage_length(passage_text, language)

    # ── Retry if AI exceeded the hard limit by more than 10% ──
    if actual_len > max_length * 1.10 and passage_text:
        trim_prompt = (
            f"The following passage is {actual_len} {'characters' if is_cjk else 'words'} long. "
            f"The hard limit is {max_length} and the target range is {min_length}–{max_length}. "
            f"Rewrite it to be as close to {max_length} {'non-whitespace characters' if is_cjk else 'words'} "
            f"as possible without exceeding {max_length}. Keep it natural and use only the same vocabulary. "
            "Return the SAME JSON structure with all fields updated to match the rewritten passage.\n\n"
            f"Original passage: {passage_text}"
        )
        retry_content = _call_ai(system_message, trim_prompt)
        retry_data = _extract_json(retry_content)
        if retry_data.get("passage"):
            data = retry_data
            passage_text = data.get("passage", "")
            actual_len = _count_passage_length(passage_text, language)

    raw_notes = data.get("vocab_notes", [])
    if not isinstance(raw_notes, list):
        raw_notes = []
    vocab_notes = [
        {
            "word":    str(n.get("word", "")),
            "reading": str(n.get("reading", "")),
            "meaning": str(n.get("meaning", "")),
            "pos":     str(n.get("pos", "")),
        }
        for n in raw_notes if isinstance(n, dict)
    ]

    return {
        "passage":       passage_text,
        "reading":       data.get("reading", ""),
        "translation":   data.get("translation", ""),
        "vocab_notes":   vocab_notes,
        "grammar_notes": data.get("grammar_notes", ""),
        "char_count":    actual_len,
        "unit_label":    unit_label,
    }

# ── 各語言可練習的文法句型 ────────────────────────────────

GRAMMAR_PATTERNS = {
    "japanese": [
        {"label": "肯定句（〜です／〜ます）",    "instruction": "Write an affirmative sentence in polite style (〜です or 〜ます form)."},
        {"label": "否定句（〜ません／〜ない）",   "instruction": "Write a negative sentence (〜ません or 〜ない form)."},
        {"label": "疑問句（〜ですか？）",         "instruction": "Write a yes/no question ending in 〜ですか or 〜ますか."},
        {"label": "過去式（〜ました／〜だった）", "instruction": "Write a sentence in past tense (〜ました or 〜だった)."},
        {"label": "て形（〜ている／〜てください）","instruction": "Write a sentence using て-form, e.g. ongoing action (〜ている) or polite request (〜てください)."},
        {"label": "意向（〜しましょう）",         "instruction": "Write a sentence expressing suggestion or willingness (〜しましょう or 〜しようと思う)."},
        {"label": "条件（〜たら／〜ば）",         "instruction": "Write a conditional sentence using 〜たら or 〜ば."},
    ],
    "korean": [
        {"label": "평서문（〜아요/어요）",        "instruction": "Write an affirmative sentence in polite present form (〜아요/어요)."},
        {"label": "부정문（안 〜／〜지 않아요）", "instruction": "Write a negative sentence using 안 + verb or 〜지 않아요."},
        {"label": "의문문（〜아요?）",            "instruction": "Write a question in polite form ending in 〜아요? or 〜어요?."},
        {"label": "과거형（〜았/었어요）",         "instruction": "Write a sentence in past tense (〜았어요/었어요)."},
        {"label": "부탁（〜세요）",               "instruction": "Write a polite request or instruction using 〜세요."},
        {"label": "희망（〜고 싶어요）",          "instruction": "Write a sentence expressing desire using 〜고 싶어요."},
        {"label": "진행형（〜고 있어요）",        "instruction": "Write a sentence describing an ongoing action using 〜고 있어요."},
    ],
    "default": [
        {"label": "肯定句",  "instruction": "Write an affirmative sentence."},
        {"label": "否定句",  "instruction": "Write a negative sentence."},
        {"label": "疑問句",  "instruction": "Write a question."},
        {"label": "過去式",  "instruction": "Write a sentence in past tense."},
        {"label": "命令句",  "instruction": "Write an imperative or request sentence."},
    ],
}
