import json
import pandas as pd
import streamlit as st
from openai import OpenAI


def _require_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("尚未設定 OPENAI_API_KEY。請在 .streamlit/secrets.toml 裡加入 API key。")
    return OpenAI(api_key=api_key)


def _extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(text)
    except Exception:
        return {}


def generate_vocab_info(language: str, term: str) -> dict:
    client = _require_openai_client()

    if language == "japanese":
        prompt = f"""
請為以下日文詞彙補全資訊。

詞彙：{term}

請輸出 JSON，格式如下：
{{
  "reading": "平假名",
  "meaning": "繁體中文翻譯",
  "pos": "詞性",
  "note": "簡短說明"
}}

規則：
1. reading 必須是平假名。
2. meaning 使用繁體中文（台灣用語）。
3. pos 請盡量使用：noun / verb / adj / adv / phrase / expression。
4. note 簡短即可。
5. 只輸出 JSON，不要加解釋。
"""
    elif language == "korean":
        prompt = f"""
請為以下韓文詞彙補全資訊。

詞彙：{term}

請輸出 JSON，格式如下：
{{
  "reading": "羅馬拼音",
  "meaning": "繁體中文翻譯",
  "pos": "詞性",
  "note": "簡短說明"
}}

規則：
1. reading 填入該韓文詞的羅馬拼音（Revised Romanization）。
2. meaning 使用繁體中文（台灣用語）。
3. pos 請盡量使用：noun / verb / adj / adv / phrase / expression。
4. note 簡短即可。
5. 只輸出 JSON，不要加解釋。
"""
    else:
        prompt = f"""
請為以下西班牙文詞彙補全資訊。

詞彙：{term}

請輸出 JSON，格式如下：
{{
  "reading": "",
  "meaning": "繁體中文翻譯",
  "pos": "詞性",
  "note": "簡短說明"
}}

規則：
1. 西班牙文 reading 保持空字串。
2. meaning 使用繁體中文（台灣用語）。
3. pos 請盡量使用：noun / verb / adj / adv / phrase / expression。
4. note 簡短即可。
5. 只輸出 JSON，不要加解釋。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise vocabulary annotation assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content or ""
    data = _extract_json(content)

    return {
        "reading": str(data.get("reading", "") or ""),
        "meaning": str(data.get("meaning", "") or ""),
        "pos": str(data.get("pos", "") or ""),
        "note": str(data.get("note", "") or "")
    }


def autocomplete_dataframe(df: pd.DataFrame, language: str) -> pd.DataFrame:
    df = df.copy().fillna("")

    required_cols = ["code", "term", "reading", "meaning", "pos", "note"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    for i, row in df.iterrows():
        term = str(row.get("term", "")).strip()
        if not term:
            continue

        needs_fill = any(
            str(row.get(col, "")).strip() == ""
            for col in ["reading", "meaning", "pos", "note"]
        )

        if not needs_fill:
            continue

        result = generate_vocab_info(language, term)

        for col in ["reading", "meaning", "pos", "note"]:
            current_value = str(row.get(col, "")).strip()
            if current_value == "":
                df.at[i, col] = result.get(col, "")

    return df


def translate_chinese_sentence(language: str, language_label: str, chinese_sentence: str) -> dict:
    client = _require_openai_client()

    prompt = f"""
Translate the following Traditional Chinese sentence into {language_label}.

Target language key: {language}
Chinese sentence: {chinese_sentence}

Return JSON only:
{{
  "sentence": "natural translation in the target language",
  "reading": "pronunciation guide if useful, otherwise empty string",
  "note": "one short Traditional Chinese note about wording, otherwise empty string"
}}

Rules:
1. Translate the meaning naturally, not word by word.
2. Use the target language only in "sentence".
3. For Japanese, put hiragana reading in "reading" when the sentence contains kanji.
4. For Korean, put Revised Romanization in "reading".
5. For languages that do not need a reading guide, leave "reading" empty.
6. Use Traditional Chinese for "note".
7. Output JSON only. Do not add Markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise translation assistant for language learners."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or ""
    data = _extract_json(content)

    return {
        "sentence": str(data.get("sentence", "") or "").strip(),
        "reading": str(data.get("reading", "") or "").strip(),
        "note": str(data.get("note", "") or "").strip(),
    }
