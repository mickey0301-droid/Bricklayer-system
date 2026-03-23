# utils/report_formatter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re


SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}


def to_superscript(num: int) -> str:
    return "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in str(num))


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


@dataclass
class CitationPolicy:
    notes_style: str = "footnote"     # footnote / endnote / none
    links_placement: str = "inline"   # inline / footnote / none

    @classmethod
    def from_format_config(cls, format_config: Optional[Dict[str, Any]]) -> "CitationPolicy":
        format_config = format_config or {}
        notes = format_config.get("notes", {}) or {}
        links = format_config.get("links", {}) or {}

        notes_style = safe_str(notes.get("style", "footnote"), "footnote").lower()
        links_placement = safe_str(links.get("placement", "inline"), "inline").lower()

        if notes_style not in {"footnote", "endnote", "none"}:
            notes_style = "footnote"
        if links_placement not in {"inline", "footnote", "none"}:
            links_placement = "inline"

        return cls(
            notes_style=notes_style,
            links_placement=links_placement,
        )


class CitationManager:
    """
    管理三種引用模式：
    1. inline   -> 內容【連結】
    2. footnote -> 內容¹ + 文末 footnotes
    3. endnote  -> 內容[1] + 文末 sources/endnotes
    """

    def __init__(self, policy: CitationPolicy):
        self.policy = policy
        self._entries: List[Dict[str, Any]] = []
        self._seen_keys: Dict[str, int] = {}

    def _build_key(self, item: Dict[str, Any]) -> str:
        url = safe_str(item.get("url") or item.get("link"))
        title = safe_str(item.get("title"))
        source = safe_str(item.get("source") or item.get("source_name"))
        return f"{url}||{title}||{source}"

    def _get_or_create_index(self, item: Dict[str, Any]) -> int:
        key = self._build_key(item)
        if key in self._seen_keys:
            return self._seen_keys[key]

        index = len(self._entries) + 1
        entry = {
            "index": index,
            "title": safe_str(item.get("title"), "Untitled"),
            "url": safe_str(item.get("url") or item.get("link")),
            "source": safe_str(item.get("source") or item.get("source_name")),
            "published_at": safe_str(
                item.get("published_at")
                or item.get("published")
                or item.get("pub_date")
                or item.get("date")
            ),
        }
        self._entries.append(entry)
        self._seen_keys[key] = index
        return index

    def build_marker(self, item: Dict[str, Any]) -> str:
        placement = self.policy.links_placement
        notes_style = self.policy.notes_style

        # inline 優先：直接在正文顯示【連結】
        if placement == "inline":
            url = safe_str(item.get("url") or item.get("link"))
            return f"【連結】" if url else ""

        # footnote 模式：正文顯示 ¹，文末附 URL
        if placement == "footnote":
            idx = self._get_or_create_index(item)
            return to_superscript(idx)

        # none：不顯示連結；但若 notes_style = endnote，可只在文末列來源
        if placement == "none":
            if notes_style == "endnote":
                idx = self._get_or_create_index(item)
                return f"[{idx}]"
            return ""

        return ""

    def register_for_endnote_only(self, item: Dict[str, Any]) -> Optional[int]:
        if self.policy.notes_style == "endnote":
            return self._get_or_create_index(item)
        return None

    def build_footnotes_block(self) -> str:
        if self.policy.notes_style != "footnote":
            return ""

        if not self._entries:
            return ""

        lines = []
        lines.append("")
        lines.append("註釋")
        lines.append("-" * 20)

        for entry in self._entries:
            idx = entry["index"]
            title = entry["title"]
            url = entry["url"]
            source = entry["source"]
            published_at = entry["published_at"]

            meta_parts = [p for p in [source, published_at] if p]
            meta = "｜".join(meta_parts)

            if meta and url:
                lines.append(f"{to_superscript(idx)} {title}｜{meta}｜{url}")
            elif url:
                lines.append(f"{to_superscript(idx)} {title}｜{url}")
            else:
                lines.append(f"{to_superscript(idx)} {title}")

        return "\n".join(lines)

    def build_endnotes_block(self) -> str:
        if self.policy.notes_style != "endnote":
            return ""

        if not self._entries:
            return ""

        lines = []
        lines.append("")
        lines.append("來源")
        lines.append("-" * 20)

        for entry in self._entries:
            idx = entry["index"]
            title = entry["title"]
            url = entry["url"]
            source = entry["source"]
            published_at = entry["published_at"]

            meta_parts = [p for p in [source, published_at] if p]
            meta = "｜".join(meta_parts)

            if meta and url:
                lines.append(f"[{idx}] {title}｜{meta}｜{url}")
            elif url:
                lines.append(f"[{idx}] {title}｜{url}")
            else:
                lines.append(f"[{idx}] {title}")

        return "\n".join(lines)

    def finalize_report(self, report_text: str) -> str:
        report_text = report_text.rstrip()

        if self.policy.notes_style == "footnote":
            footnotes = self.build_footnotes_block()
            if footnotes:
                report_text += "\n" + footnotes

        elif self.policy.notes_style == "endnote":
            endnotes = self.build_endnotes_block()
            if endnotes:
                report_text += "\n" + endnotes

        return report_text


def attach_marker(text: str, marker: str) -> str:
    """
    把 marker 盡量附在句尾，而不是額外換行。
    """
    text = safe_str(text)
    marker = safe_str(marker)

    if not text or not marker:
        return text

    # 如果已經有同樣標記，就不要重複加
    if text.endswith(marker):
        return text

    # 常見中英文句尾標點
    punct_pattern = r"([。！？.!?])$"
    m = re.search(punct_pattern, text)
    if m:
        punct = m.group(1)
        return text[:-1] + marker + punct

    return text + marker


def pick_display_link(item: Dict[str, Any]) -> str:
    return safe_str(item.get("url") or item.get("link"))


def format_item_reference_line(item: Dict[str, Any], citation_manager: CitationManager) -> str:
    """
    單條 item 摘要用。
    """
    title = safe_str(item.get("title"), "Untitled")
    source = safe_str(item.get("source") or item.get("source_name"))
    summary = safe_str(item.get("summary") or item.get("snippet") or item.get("content"))

    base = f"• {title}"
    if source:
        base += f"（{source}）"

    if summary:
        base += f"：{summary}"

    marker = citation_manager.build_marker(item)
    return attach_marker(base, marker)