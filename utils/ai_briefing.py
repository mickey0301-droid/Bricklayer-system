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