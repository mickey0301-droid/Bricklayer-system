"""
AI briefing generation — supports OpenAI, Google Gemini, and Anthropic Claude.

Model preference is read (in priority order) from:
  1. st.session_state["ai_model"]   (set via the UI model selector)
  2. config/ai_model.json           (persisted between sessions)
  3. Hard-coded default             (gpt-4.1-mini)

Supported model strings:
  OpenAI  : gpt-4o-mini, gpt-4.1-mini
  Gemini  : gemini-2.0-flash, gemini-1.5-pro
  Claude  : claude-haiku-4-5-20251001, claude-sonnet-4-6
"""
import os
import json


# ── Model config helpers ──────────────────────────────────────────────────────

_AI_MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "ai_model.json"
)
_DEFAULT_MODEL = "gpt-4.1-mini"


def _get_preferred_model() -> str:
    """Return the user-selected AI model, falling back to config file then default."""
    # 1. Streamlit session_state (set by UI selector)
    try:
        import streamlit as st
        m = st.session_state.get("ai_model", "")
        if m:
            return m
    except Exception:
        pass
    # 2. Persisted config file
    try:
        if os.path.exists(_AI_MODEL_CONFIG_PATH):
            with open(_AI_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("model", _DEFAULT_MODEL)
    except Exception:
        pass
    return _DEFAULT_MODEL


def save_ai_model(model: str) -> None:
    """Persist the selected AI model to config/ai_model.json."""
    try:
        os.makedirs(os.path.dirname(_AI_MODEL_CONFIG_PATH), exist_ok=True)
        with open(_AI_MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"model": model}, f, ensure_ascii=False)
    except Exception:
        pass


def load_ai_model() -> str:
    """Load the persisted AI model name (or default)."""
    try:
        if os.path.exists(_AI_MODEL_CONFIG_PATH):
            with open(_AI_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("model", _DEFAULT_MODEL)
    except Exception:
        pass
    return _DEFAULT_MODEL


# ── Backward-compat OpenAI client helper (used by report_engine & topic_cluster) ─
def get_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。\n\n請在 Streamlit secrets 或環境變數設定：\nOPENAI_API_KEY=你的key"
        )
    return OpenAI(api_key=api_key)


# ── Provider dispatch ─────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_content: str, model: str | None = None) -> str:
    """
    Call the appropriate LLM provider based on the model name.
    Falls back to _get_preferred_model() if model is None.
    """
    if model is None:
        model = _get_preferred_model()

    # ── Gemini ────────────────────────────────────────────────────────────────
    if model.startswith("gemini"):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "未找到 GOOGLE_API_KEY。請在 Streamlit secrets 或環境變數設定。"
            )
        genai.configure(api_key=api_key)
        gem_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
        )
        response = gem_model.generate_content(
            user_content,
            generation_config=genai.GenerationConfig(temperature=0.3),
        )
        return response.text

    # ── Claude (Anthropic) ────────────────────────────────────────────────────
    if model.startswith("claude"):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "未找到 ANTHROPIC_API_KEY。請在 Streamlit secrets 或環境變數設定。"
            )
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text

    # ── OpenAI (default) ──────────────────────────────────────────────────────
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。請在 Streamlit secrets 或環境變數設定。"
        )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ── Public API ────────────────────────────────────────────────────────────────

def generate_ai_briefing(prompt: str, language: str = "中文") -> str:

    system_prompt = "You are a geopolitical intelligence analyst.\n\nWrite a strategic intelligence briefing.\n\nBe analytical and structured."
    if language == "English":
        system_prompt = "You are a geopolitical intelligence analyst.\n\nWrite a strategic intelligence briefing in English."

    return _call_llm(system_prompt, prompt)


def generate_sub_briefing(prompt: str, group_name: str = "", language: str = "繁體中文") -> str:
    """
    Generate a full eight-chapter sub-report for one source group.
    The source group label indicates the ORIGIN of the media, NOT a topic constraint —
    articles from any source can cover any global topic and must be placed in the
    appropriate chapter based on their CONTENT.
    Citation codes [S1][S2]... MUST be preserved intact in the output.
    """
    system_prompt = f"""You are a senior intelligence analyst preparing a structured sub-report based on articles from the [{group_name}] media source group.

IMPORTANT FRAMING: [{group_name}] refers to the ORIGIN of the media sources, NOT a topic limitation.
These media outlets may cover any global topic — European news, American politics, Taiwan Strait tensions, African conflicts, etc.
Your job is to read each article and place it in the correct chapter based on its CONTENT, regardless of where the media outlet is based.
Every article in the input must be used somewhere in the report.

Requirements:
1. Write in {language}, in formal analytical prose (NOT bullet points).
2. ALL eight chapters are REQUIRED. 五、中國要聞 MUST contain TWO sub-sections: （一）中國對外情勢 and （二）中國內部情勢 — both always present; write「本期無相關新聞。」only if truly no article addresses that sub-topic. For all other chapters or sub-sections, write the content based on what the provided articles actually report. Only write "本期無相關新聞。" if truly no article in the input addresses that topic at all.
3. CRITICAL — Citation preservation: Each input article has a citation code [S1], [S2], etc. When you state a fact from that article, append the code right after the sentence, e.g. "...路透社（Reuters）報導。[S3]". NEVER drop, renumber, merge, or invent [Sx] codes. These are the only link to the source bibliography.
4. Do NOT place raw URLs anywhere in the body text.
5. MANDATORY — People: full official title + Chinese name on first mention. ENGLISH NAME RULES BY NATIONALITY — (A) Taiwan/ROC officials and PRC/China officials: DO NOT add a parenthetical English name. Write the Chinese name only (e.g., 行政院長卓榮泰, NOT 卓榮泰（Cho Jung-tai）; 國家主席習近平, NOT 習近平（Xi Jinping））. (B) Western figures: surname only in parentheses, e.g. 美國總統川普（Donald Trump）. (C) Japanese, Korean, Vietnamese: romanised form in square brackets + English in parentheses, e.g. 日本首相石破茂[Ishiba Shigeru]（Shigeru Ishiba）. (D) Other East Asian (Singapore, etc.): internationally recognised English name in parentheses. CRITICAL — Be 100% certain before writing any English name. Known error to avoid: 黃循財 = Lawrence Wong (Singapore PM since 2024), NOT Heng Swee Keat. If uncertain, OMIT the parenthetical entirely. CRITICAL — Parentheses must contain ONLY the English name, no titles (correct: 川普（Donald Trump）; WRONG: 川普（President Donald Trump）). CRITICAL — Only assign a ministerial title to a person if the news articles explicitly confirm they currently hold it.
5a. MANDATORY — Expert names in 七、專家研析: render expert/analyst names in bold (**Name**) with full title and affiliation on first mention.
6. MANDATORY — Media outlets: never use vague collective terms like "西方媒體"、"外媒". Use the specific outlet name with Chinese and English on first mention, e.g. 路透社（Reuters）、德國之聲（Deutsche Welle）.
6a. MANDATORY — Media country attribution: When citing a non-Chinese / non-English language media outlet, note its country on first mention. Format: 「[媒體名稱]（[國家]）」. E.g. 《朝日新聞》（日本）、《韓聯社》（韓國）、《明鏡週刊》（德國）. The news data includes "來源" with country info in parentheses — use it. Articles with [日文]/[韓文]/[德文] etc. tags in the title come from the corresponding country.
7. MANDATORY — Organizations: Chinese name（English Name）on first mention, e.g. 北大西洋公約組織（NATO）、美國國務院（U.S. Department of State）.
8. Begin directly with 一、摘要.

Output structure (all eight chapters REQUIRED; distribute articles by content, not by source origin):

一、摘要
（本群組本期最重要判斷，一段）

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞
（一）中國對外情勢
（聚焦中國外交、軍事對外、對外貿易與制裁、涉外聲明、對台對美對他國關係；若本期無相關新聞，寫「本期無相關新聞。」）
（二）中國內部情勢
（聚焦中國黨政內鬥、國內經濟、社會民情、人權、新疆西藏香港等內部議題；若本期無相關新聞，寫「本期無相關新聞。」）

六、區域情勢
（依各文章所報導的地理區域分子節，六大區域全部列出；若某區域確實無任何文章涉及，寫「本期無相關新聞。」）
（一）亞太地區
1. 國際要聞研析
2. 台美中要聞研析
（二）亞西地區
1. 國際要聞研析
2. 台美中要聞研析
（三）北美地區
1. 國際要聞研析
2. 台美中要聞研析
（四）拉丁美洲及加勒比海
1. 國際要聞研析
2. 台美中要聞研析
（五）歐洲地區
1. 國際要聞研析
2. 台美中要聞研析
（六）非洲地區
1. 國際要聞研析
2. 台美中要聞研析

七、專家研析
1. 國際要聞研析
2. 台美中要聞研析

八、研析
1. 國際要聞研析
2. 台美中要聞研析"""

    return _call_llm(system_prompt, prompt)


def generate_section_mini_report(
    section_path: str,
    section_label: str,
    news_block: str,
    language: str = "繁體中文",
) -> str:
    """
    Generate a focused 2-3 paragraph mini-report for one specific section.
    Used by the segmented report mode.

    section_path: e.g. "三、台美中要聞" or "六、區域情勢｜（一）亞太地區｜1. 國際要聞研析"
    section_label: short display label, e.g. "三、台美中要聞"
    news_block: formatted news items block (from _format_item_block)
    """
    system_prompt = f"""You are a senior strategic intelligence analyst.

Write a concise mini-report in {language} for the following section of a strategic intelligence briefing:

Section: {section_path}

Requirements:
1. Write 2-3 focused analytical paragraphs (NOT bullet points) covering the most important developments for this specific section.
2. Use ONLY the provided news articles as your source material. If articles are insufficient, write what you can and note gaps.
3. CRITICAL — Citation: preserve [S1][S2]... codes if present in the input. Append the code after the relevant sentence.
4. Do NOT place raw URLs in the body text.
5. MANDATORY — People: full official title + Chinese name on first mention. ENGLISH NAME RULES BY NATIONALITY — (A) Taiwan/ROC officials and PRC/China officials: DO NOT add a parenthetical English name. Write the Chinese name only (e.g., 行政院長卓榮泰, NOT 卓榮泰（Cho Jung-tai）; 國家主席習近平, NOT 習近平（Xi Jinping））. (B) Western figures: surname only in parentheses, e.g. 美國總統川普（Donald Trump）. (C) Japanese, Korean, Vietnamese: romanised form in square brackets + English in parentheses, e.g. 日本首相石破茂[Ishiba Shigeru]（Shigeru Ishiba）. (D) Other East Asian (Singapore, etc.): internationally recognised English name in parentheses. CRITICAL — Be 100% certain before writing any English name. Known error to avoid: 黃循財 = Lawrence Wong (Singapore PM since 2024), NOT Heng Swee Keat. If uncertain, OMIT the parenthetical entirely. CRITICAL — Parentheses must contain ONLY the English name, no titles (correct: 川普（Donald Trump）; WRONG: 川普（President Donald Trump）). CRITICAL — Only assign a ministerial title to a person if the news articles explicitly confirm they currently hold it.
6. MANDATORY — Media outlets: use specific outlet name (Chinese + English) on first mention, e.g. 路透社（Reuters）. Never "外媒" or "西方媒體".
6a. MANDATORY — Media country: for non-Chinese/non-English outlets, note the country on first mention, e.g. 《朝日新聞》（日本）.
7. MANDATORY — Organizations: Chinese name（English Name）on first mention.
8. Begin your response directly with the section content (no extra headers needed since the section title is already provided).
9. If the provided articles genuinely have no relevant content for this section, write: 「本期搜尋結果未見符合本節主旨的相關新聞。」
"""

    user_content = (
        f"Please write the mini-report for: {section_label}\n\n"
        f"News articles:\n{news_block}"
    )

    return _call_llm(system_prompt, user_content)
