import os
from openai import OpenAI


def get_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。\n\n請在系統環境變數設定：\nOPENAI_API_KEY=你的key"
        )

    return OpenAI(api_key=api_key)


def generate_ai_briefing(prompt: str, language: str = "中文"):

    client = get_client()

    system_prompt = """
You are a geopolitical intelligence analyst.

Write a strategic intelligence briefing.

Be analytical and structured.
"""

    if language == "English":
        system_prompt = """
You are a geopolitical intelligence analyst.

Write a strategic intelligence briefing in English.
"""

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],

        temperature=0.3
    )

    return response.choices[0].message.content


def generate_sub_briefing(prompt: str, group_name: str = "", language: str = "繁體中文") -> str:
    """
    Generate a full eight-chapter sub-report for one source group.
    The source group label indicates the ORIGIN of the media, NOT a topic constraint —
    articles from any source can cover any global topic and must be placed in the
    appropriate chapter based on their CONTENT.
    Citation codes [S1][S2]... MUST be preserved intact in the output.
    """
    client = get_client()

    system_prompt = f"""You are a senior intelligence analyst preparing a structured sub-report based on articles from the [{group_name}] media source group.

IMPORTANT FRAMING: [{group_name}] refers to the ORIGIN of the media sources, NOT a topic limitation.
These media outlets may cover any global topic — European news, American politics, Taiwan Strait tensions, African conflicts, etc.
Your job is to read each article and place it in the correct chapter based on its CONTENT, regardless of where the media outlet is based.
Every article in the input must be used somewhere in the report.

Requirements:
1. Write in {language}, in formal analytical prose (NOT bullet points).
2. ALL eight chapters are REQUIRED. For each chapter or sub-section, write the content based on what the provided articles actually report. Only write "本期無相關新聞。" if truly no article in the input addresses that topic at all.
3. CRITICAL — Citation preservation: Each input article has a citation code [S1], [S2], etc. When you state a fact from that article, append the code right after the sentence, e.g. "...路透社（Reuters）報導。[S3]". NEVER drop, renumber, merge, or invent [Sx] codes. These are the only link to the source bibliography.
4. Do NOT place raw URLs anywhere in the body text.
5. MANDATORY — People: full official title + Chinese name on first mention, followed by English/romanised name in parentheses. For Japanese, Korean, and Vietnamese names, add romanised form in square brackets: e.g. 日本首相岸田文雄[Kishida Fumio]（Fumio Kishida）、韓國總統尹錫悅[Yoon Suk-yeol]（Yoon Suk-yeol）. Western figures: surname only, e.g. 美國總統川普（Donald Trump）.
5a. MANDATORY — Expert names in 七、專家研析: render expert/analyst names in bold (**Name**) with full title and affiliation on first mention.
6. MANDATORY — Media outlets: never use vague collective terms like "西方媒體"、"外媒". Use the specific outlet name with Chinese and English on first mention, e.g. 路透社（Reuters）、德國之聲（Deutsche Welle）.
7. MANDATORY — Organizations: Chinese name（English Name）on first mention, e.g. 北大西洋公約組織（NATO）、美國國務院（U.S. Department of State）.
8. Begin directly with 一、摘要.

Output structure (all eight chapters REQUIRED; distribute articles by content, not by source origin):

一、摘要
（本群組本期最重要判斷，一段）

二、國際要聞

三、台美中要聞

四、台灣國安要聞

五、中國要聞

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

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
