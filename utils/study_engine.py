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


def _call_ai(system_message: str, user_prompt: str) -> str:
    """統一 AI 呼叫介面，回傳模型的純文字回應。"""
    provider = _ai_provider()
    model    = _ai_model()

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
    df = prepare_study_df(df)
    return df[df["code_num"] <= current_code]


def get_current_row(df: pd.DataFrame, index: int):
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


# ── 例句生成 ──────────────────────────────────────────────

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
) -> dict:
    """
    生成例句。
    - term_meaning / term_reading / term_pos：消除同音異義
    - current_code <= 100：允許最多 1 個編號較大的詞（寬鬆模式）
    - current_code > 100：嚴格只能用已學詞彙
    - review_mode=True：最多 2 個已學詞彙、句子不超過 7 個字
    - allowed_vocab：帶編號的詞表，AI 將在 vocab_codes 欄位回傳使用到的編號
    """
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
    elif current_code > 0 and current_code <= 100:
        vocab_rule = (
            "VOCABULARY RULE (BEGINNER — HARD LIMIT, NO EXCEPTIONS): "
            "The learner has studied fewer than 100 words. "
            "ALL content words (nouns, verbs, adjectives, adverbs) MUST come from the ALLOWED VOCABULARY list. "
            "You are permitted AT MOST 1 (ONE) content word outside the list — "
            "ONLY when it is literally impossible to form a grammatically natural sentence without it. "
            "Using 2 or more words outside the list is STRICTLY FORBIDDEN regardless of any reason. "
            "If you are tempted to use a second outside word, rephrase the sentence instead. "
            "This limit is ABSOLUTE.\n"
        )
    else:
        vocab_rule = (
            "VOCABULARY RULE (STRICT): The learner has studied 100+ words. "
            "ALL content words (nouns, verbs, adjectives, adverbs) MUST come from the ALLOWED VOCABULARY list. "
            "Do NOT use any content word outside the list.\n"
        )

    system_message = (
        "You are a language learning assistant that writes natural, idiomatic example sentences. "
        "NATURALNESS IS THE TOP PRIORITY — the sentence must sound like something a native speaker would actually say. "
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
    loose_rule_ja = "\n- 【厳守・絶対ルール】リストにない内容語は最大1語のみ許可。2語以上の使用は絶対禁止。リストの語で言い換えられる場合はリストを使うこと。" if (not review_mode and current_code > 0 and current_code <= 100) else ""
    loose_rule_ko = "\n- 【엄수·절대 규칙】목록에 없는 내용어는 최대 1개만 허용. 2개 이상은 절대 금지. 목록 단어로 바꿔 말할 수 있으면 반드시 목록을 사용할 것." if (not review_mode and current_code > 0 and current_code <= 100) else ""
    loose_rule_es = "\n- 【OBLIGATORIO·ABSOLUTO】Solo se permite 1 (UNA) palabra de contenido fuera de la lista. 2 o más palabras fuera está ESTRICTAMENTE PROHIBIDO. Si se puede reformular con palabras de la lista, hazlo." if (not review_mode and current_code > 0 and current_code <= 100) else ""

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
- リストの語を無理に全部使う必要はない。自然な短文を一つだけ作ること。{content_word_rule_ja}{length_rule_ja}{loose_rule_ja}
- grammarフィールドは文中の実際の単語をそのまま使うこと。「語」という文字は絶対に使わない。
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
- 조사·어미·접속사·보조동사 등 문법 요소는 자유롭게 사용 가능.{content_word_rule_ko}{length_rule_ko}{loose_rule_ko}
- grammar 필드는 문장의 실제 단어만 사용（「단어」라는 글자 절대 금지）.
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
- No es necesario usar todas las palabras. Una oración corta y natural es mejor.{content_word_rule_es}{length_rule_es}{loose_rule_es}
- grammar: usa siempre las palabras reales de la oración（nunca escribas «palabra»）.
  Ej：Yo[代名詞: 我] + como(comer)[動詞: 吃] + pan[名詞: 麵包]

Responde solo con JSON：
{{"sentence":"oración en español","reading":"","translation":"繁體中文翻譯","grammar":"palabraReal[品詞: 意思] + ...","vocab_codes":[códigos enteros usados de la lista]}}"""

    content = _call_ai(system_message, prompt)
    data = _extract_json(content)
    raw_codes = data.get("vocab_codes", [])
    vocab_codes = [int(c) for c in (raw_codes if isinstance(raw_codes, list) else []) if str(c).strip().lstrip("-").isdigit()]
    return {
        "sentence":    data.get("sentence", ""),
        "reading":     data.get("reading", ""),
        "translation": data.get("translation", ""),
        "grammar":     data.get("grammar", ""),
        "vocab_codes": vocab_codes,
    }


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
        "Content words (nouns, verbs, adjectives, adverbs) should come from the ALLOWED VOCABULARY list. "
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

    content = _call_ai(system_message, prompt)
    data = _extract_json(content)
    raw_codes = data.get("vocab_codes", [])
    vocab_codes = [int(c) for c in (raw_codes if isinstance(raw_codes, list) else [])
                   if str(c).strip().lstrip("-").isdigit()]
    return {
        "sentence":    data.get("sentence", ""),
        "reading":     data.get("reading", ""),
        "translation": data.get("translation", ""),
        "grammar":     data.get("grammar", ""),
        "drill_note":  data.get("drill_note", ""),
        "vocab_codes": vocab_codes,
    }


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
    return {
        "sentence":    data.get("sentence", ""),
        "reading":     data.get("reading", ""),
        "translation": data.get("translation", ""),
        "grammar":     data.get("grammar", ""),
        "vocab_codes": vocab_codes,
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
