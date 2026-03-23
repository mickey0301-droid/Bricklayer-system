from typing import List, Dict

from utils.ai_briefing import generate_ai_briefing


def build_topic_articles(cluster: Dict, items: List[dict]):

    idxs = cluster.get("articles", [])

    blocks = []

    for i in idxs:

        if i < 0 or i >= len(items):
            continue

        item = items[i]

        title = item.get("title", "")
        source = item.get("source_name") or item.get("source", "")
        link = item.get("link", "")

        block = f"""
title: {title}
source: {source}
link: {link}
"""

        blocks.append(block)

    return "\n".join(blocks)


def analyze_topics(
    clusters: List[Dict],
    items: List[dict],
    insights_text: str,
    language: str
):

    results = []

    for cluster in clusters:

        topic = cluster.get("topic", "Unknown Topic")

        articles = build_topic_articles(cluster, items)

        prompt = f"""
你是一個國際戰略分析師。

請分析以下議題：

Topic:
{topic}

相關新聞：

{articles}

分析要求：

- 使用繁體中文
- 使用台灣用語
- 不要只摘要新聞
- 分析 geopolitics
- 指出戰略意涵
- 指出未來觀察重點

Insights:

{insights_text}
"""

        analysis = generate_ai_briefing(
            prompt=prompt,
            language=language
        )

        block = f"""
## {topic}

{analysis}
"""

        results.append(block)

    return "\n".join(results)