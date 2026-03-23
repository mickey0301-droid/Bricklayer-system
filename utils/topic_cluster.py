from typing import List, Dict
import json
import re

from utils.ai_briefing import get_client


def build_cluster_input(items: List[dict]) -> str:

    blocks = []

    for i, item in enumerate(items):

        title = item.get("title", "")
        source = item.get("source_name") or item.get("source", "")
        link = item.get("link", "")

        block = f"""
[{i}]
title: {title}
source: {source}
link: {link}
"""

        blocks.append(block)

    return "\n".join(blocks)


def extract_json(text: str):

    # 找到 JSON array
    match = re.search(r"\[.*\]", text, re.S)

    if match:
        return match.group(0)

    return None


def cluster_topics(items: List[dict]) -> List[Dict]:

    if not items:
        return []

    client = get_client()

    articles_text = build_cluster_input(items)

    prompt = f"""
你是一個國際政治分析師。

請將以下新聞分群。

要求：

1. 按照議題分群
2. 每個群代表一個 geopolitics topic
3. 每群給一個 topic name
4. 每群列出文章 index

只輸出 JSON。

格式：

[
  {{
    "topic": "topic name",
    "articles": [0,1]
  }}
]

新聞：

{articles_text}
"""

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {"role": "user", "content": prompt}
        ],

        temperature=0.2
    )

    text = response.choices[0].message.content

    print("\n===== AI RESPONSE =====\n")
    print(text)
    print("\n=======================\n")

    json_text = extract_json(text)

    if not json_text:
        return []

    try:
        clusters = json.loads(json_text)
    except Exception:
        clusters = []

    return clusters