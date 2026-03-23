import math
from typing import List, Dict, Any

from openai import OpenAI


def get_openai_client() -> OpenAI:
    """
    建立 OpenAI client。
    預設會從環境變數 OPENAI_API_KEY 讀取金鑰。
    """
    return OpenAI()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    計算 cosine similarity。
    若任一向量為空，回傳 0。
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def get_text_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    一次批次取得多段文字的 embeddings。
    空字串會保留位置，回傳空向量。
    """
    if not texts:
        return []

    cleaned_texts = []
    valid_indices = []

    for i, text in enumerate(texts):
        text = (text or "").strip()
        if text:
            cleaned_texts.append(text)
            valid_indices.append(i)

    results = [[] for _ in texts]

    if not cleaned_texts:
        return results

    client = get_openai_client()
    response = client.embeddings.create(
        model=model,
        input=cleaned_texts
    )

    for result_index, embedding_data in enumerate(response.data):
        original_index = valid_indices[result_index]
        results[original_index] = embedding_data.embedding

    return results


def build_embedding_records(
    items: List[Dict[str, Any]],
    model: str = "text-embedding-3-small"
) -> List[Dict[str, Any]]:
    """
    將文章 items 轉成含 embedding 的 records。
    """
    titles = [(item.get("title", "") or "").strip() for item in items]
    embeddings = get_text_embeddings(titles, model=model)

    records = []

    for item, title, embedding in zip(items, titles, embeddings):
        records.append({
            "item": item,
            "title": title,
            "embedding": embedding
        })

    return records


def cluster_by_embedding_similarity(
    items: List[Dict[str, Any]],
    model: str = "text-embedding-3-small",
    similarity_threshold: float = 0.82
) -> List[Dict[str, Any]]:
    """
    用簡單的 greedy clustering 依 embedding 相似度分群。
    """
    records = build_embedding_records(items, model=model)
    clusters = []

    for record in records:
        placed = False

        for cluster in clusters:
            anchor_embedding = cluster["anchor_embedding"]
            sim = cosine_similarity(record["embedding"], anchor_embedding)

            if sim >= similarity_threshold:
                cluster["items"].append(record["item"])
                cluster["titles"].append(record["title"])

                source = (
                    record["item"].get("source_name")
                    or record["item"].get("source")
                    or record["item"].get("feed")
                )

                if source and source not in cluster["sources"]:
                    cluster["sources"].append(source)

                placed = True
                break

        if not placed:
            source = (
                record["item"].get("source_name")
                or record["item"].get("source")
                or record["item"].get("feed")
            )

            clusters.append({
                "cluster_id": f"cluster_{len(clusters) + 1}",
                "anchor_embedding": record["embedding"],
                "article_count": 0,
                "titles": [record["title"]],
                "sources": [source] if source else [],
                "items": [record["item"]]
            })

    for cluster in clusters:
        cluster["article_count"] = len(cluster["items"])
        cluster.pop("anchor_embedding", None)

    clusters.sort(key=lambda x: x["article_count"], reverse=True)

    return clusters