import json
import feedparser
from datetime import datetime


def load_experts():

    try:
        with open("config/experts.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def search_expert_news(expert):

    aliases = expert.get("aliases", [])

    items = []

    for name in aliases:

        query = f'"{name}"'

        url = (
            "https://news.google.com/rss/search?q="
            + query +
            "&hl=en-US&gl=US&ceid=US:en"
        )

        feed = feedparser.parse(url)

        for entry in feed.entries:

            items.append({
                "title": entry.title,
                "url": entry.link,
                "source": entry.get("source", {}).get("title",""),
                "published": entry.get("published",""),
                "summary": entry.get("summary",""),
                "expert": expert.get("name"),
                "type": "expert"
            })

    return items


def fetch_expert_items(selected_experts=None):

    experts = load_experts()

    if selected_experts:

        experts = [
            e for e in experts
            if e["name"] in selected_experts
        ]

    all_items = []

    for expert in experts:

        if not expert.get("enabled", True):
            continue

        try:

            items = search_expert_news(expert)

            all_items.extend(items)

        except:

            pass

    return all_items