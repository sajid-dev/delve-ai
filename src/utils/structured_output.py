"""Utilities for analysing LLM output and building structured responses."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..models.enums import MessageContentType


@dataclass
class ContentAnalysis:
    """Lightweight representation of parsed assistant output."""

    content_type: MessageContentType
    structured_data: Any | None
    text: str


def analyse_content(content: str) -> ContentAnalysis:
    """Classify assistant output and provide optional structured data."""
    stripped = content.strip()
    if not stripped:
        return ContentAnalysis(MessageContentType.TEXT, None, content)

    image_payload = _extract_image_payload(stripped)
    if image_payload is not None:
        return ContentAnalysis(MessageContentType.IMAGE, image_payload, content)

    table_payload = _parse_table(stripped)
    if table_payload is not None:
        return ContentAnalysis(MessageContentType.TABLE, table_payload, content)

    list_payload = _parse_list(stripped)
    if list_payload is not None:
        return ContentAnalysis(MessageContentType.LIST, list_payload, content)

    parsed_json = _parse_json(stripped)
    if parsed_json is not None:
        if _looks_like_chart_spec(parsed_json):
            return ContentAnalysis(MessageContentType.CHART, parsed_json, content)
        return ContentAnalysis(MessageContentType.JSON, parsed_json, content)

    code_payload = _parse_code_block(content)
    if code_payload is not None:
        return ContentAnalysis(MessageContentType.CODE, code_payload, content)

    if _looks_like_html(stripped):
        return ContentAnalysis(MessageContentType.HTML, None, content)

    if _looks_like_markdown(stripped):
        return ContentAnalysis(MessageContentType.MARKDOWN, None, content)

    return ContentAnalysis(MessageContentType.TEXT, None, content)


def build_response_payload(analysis: ContentAnalysis) -> dict[str, Any]:
    """Normalise analysis output into a component-based structure."""
    components = _build_components_from_analysis(analysis)
    if not components:
        components = [_text_component(analysis.text)]
    return {"components": components}


def _build_components_from_analysis(analysis: ContentAnalysis) -> list[dict[str, Any]]:
    """Translate classified content into frontend component payloads."""
    ctype = analysis.content_type
    data = analysis.structured_data
    text = analysis.text

    if ctype == MessageContentType.TABLE and isinstance(data, dict):
        component = _table_component(data, text)
        return [component] if component else []

    if ctype == MessageContentType.LIST and isinstance(data, dict):
        component = _list_component(data, text)
        return component

    if ctype == MessageContentType.IMAGE and isinstance(data, dict):
        component = _image_component(data)
        return [component]

    if ctype == MessageContentType.CODE and isinstance(data, dict):
        component = _code_component(data, text)
        return [component]

    if ctype == MessageContentType.CHART and isinstance(data, dict):
        component = _chart_component(data, text)
        return [component]

    if ctype == MessageContentType.JSON and data is not None:
        return [_custom_component(data, text)]

    if ctype == MessageContentType.HTML:
        return [_custom_component({"html": text}, text)]

    if ctype in {MessageContentType.MARKDOWN, MessageContentType.TEXT}:
        return [_text_component(text)]

    return []


def _text_component(text: str) -> dict[str, Any]:
    return {
        "type": "text",
        "payload": {
            "content": text.strip() if text else "",
        },
    }


def _list_component(payload: dict[str, Any], fallback_text: str) -> list[dict[str, Any]]:
    items_payload: list[str] = []
    for item in payload.get("items", []) or []:
        parts: list[str] = []
        title = item.get("title")
        description = item.get("description")
        if title:
            parts.append(str(title).strip())
        if description:
            parts.append(str(description).strip())
        for bullet in item.get("bullets", []) or []:
            parts.append(f"- {bullet}")
        for block in item.get("code_blocks", []) or []:
            code = block.get("code")
            language = block.get("language")
            if code:
                parts.append(f"```{language or ''}\n{code}\n```")
        if not parts:
            raw = item.get("raw")
            if raw:
                parts.append(str(raw).strip())
        entry = "\n".join(part for part in parts if part).strip()
        if entry:
            items_payload.append(entry)

    if not items_payload and fallback_text:
        items_payload.append(fallback_text.strip())

    list_components: list[dict[str, Any]] = []
    if items_payload:
        list_components.append(
            {
                "type": "list",
                "payload": {
                    "items": items_payload,
                },
            }
        )
    else:
        list_components.append(_text_component(fallback_text))
    return list_components


def _table_component(payload: dict[str, Any], fallback_text: str) -> dict[str, Any] | None:
    headers = payload.get("headers")
    rows = payload.get("rows")
    table_payload: dict[str, Any] = {}

    if isinstance(headers, list) and all(isinstance(item, str) for item in headers):
        table_payload["headers"] = headers

    normalised_rows: list[list[str]] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and table_payload.get("headers"):
                header_list = table_payload.get("headers", [])
                normalised_rows.append([str(row.get(header, "")) for header in header_list])
            elif isinstance(row, list):
                normalised_rows.append([str(cell) for cell in row])

    if normalised_rows:
        table_payload["rows"] = normalised_rows

    if not table_payload:
        if fallback_text:
            return _text_component(fallback_text)
        return None

    return {"type": "table", "payload": table_payload}


def _image_component(payload: dict[str, Any]) -> dict[str, Any]:
    component_payload: dict[str, Any] = {}
    url = payload.get("url")
    if url:
        component_payload["url"] = url
    alt = payload.get("alt")
    if alt is not None:
        component_payload["alt"] = alt
    return {"type": "image", "payload": component_payload}


def _code_component(payload: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    component_payload: dict[str, Any] = {
        "code": payload.get("code", fallback_text or ""),
    }
    language = payload.get("language")
    if language:
        component_payload["language"] = language
    parsed = payload.get("parsed")
    if parsed is not None:
        component_payload["data"] = parsed
    return {"type": "code", "payload": component_payload}


def _chart_component(payload: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    component_payload: dict[str, Any] = {"data": payload}

    chart_type = payload.get("type") or payload.get("chartType") or payload.get("chart_type")
    if isinstance(chart_type, str):
        component_payload["chart_type"] = chart_type

    labels = payload.get("labels")
    if isinstance(labels, list):
        component_payload["labels"] = labels

    values = payload.get("values")
    if isinstance(values, list):
        component_payload["values"] = values

    data_section = payload.get("data")
    if isinstance(data_section, dict):
        if "labels" not in component_payload and isinstance(data_section.get("labels"), list):
            component_payload["labels"] = data_section["labels"]
        datasets = data_section.get("datasets") or data_section.get("series")
        if isinstance(datasets, list):
            first_dataset = datasets[0]
            if isinstance(first_dataset, dict):
                data_values = first_dataset.get("data") or first_dataset.get("values")
                if isinstance(data_values, list):
                    component_payload.setdefault("values", data_values)

    if not component_payload.get("chart_type") and fallback_text:
        component_payload["title"] = fallback_text.strip()

    return {"type": "chart", "payload": component_payload}


def _custom_component(data: object, fallback_text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"data": data}
    if fallback_text:
        payload["content"] = fallback_text.strip()
    return {"type": "custom", "payload": payload}


def _extract_image_payload(content: str) -> dict[str, str] | None:
    """Return image details when the content resembles an image reference."""
    direct_url = re.match(r"^https?://\S+\.(png|jpe?g|gif|webp|svg)$", content, re.IGNORECASE)
    if direct_url:
        return {"url": content.strip(), "alt": ""}

    md_match = re.match(r"^!\[([^\]]*)\]\((https?://[^\s)]+)\)$", content.strip())
    if md_match:
        alt_text = md_match.group(1).strip()
        return {"url": md_match.group(2).strip(), "alt": alt_text}
    return None


def _parse_table(content: str) -> dict[str, Any] | None:
    """Attempt to parse markdown or HTML table payloads."""
    markdown_table = _parse_markdown_table(content)
    if markdown_table is not None:
        return markdown_table
    if "<table" in content.lower() and "</table>" in content.lower():
        return {"html": content}
    if _looks_like_table(content):
        return {"raw": content}
    return None


def _parse_markdown_table(content: str) -> dict[str, Any] | None:
    """Parse a Markdown table into headers/rows if present."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    for idx in range(len(lines) - 1):
        header_line = lines[idx]
        separator_line = lines[idx + 1]
        if "|" not in header_line:
            continue
        if not re.match(r"^\|?\s*:?-{3,}.*", separator_line):
            continue

        headers = [cell.strip() for cell in header_line.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for row_line in lines[idx + 2 :]:
            if "|" not in row_line:
                break
            cells = [cell.strip() for cell in row_line.strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))

        if rows:
            return {"headers": headers, "rows": rows}
    return None


def _parse_list(content: str) -> dict[str, Any] | None:
    """Detect ordered/unordered list structures and return items."""
    ordered_items = _parse_ordered_list(content)
    if ordered_items:
        return {"ordered": True, "items": _normalise_list_items(ordered_items)}

    unordered_items = _parse_unordered_list(content)
    if unordered_items:
        return {"ordered": False, "items": _normalise_list_items(unordered_items)}

    return None


def _parse_ordered_list(content: str) -> list[str]:
    """Return ordered list items if the content resembles a numbered list."""
    items: list[str] = []
    current: list[str] | None = None
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped and current is not None:
            current.append("")
            continue

        match = re.match(r"^(\d+)[\.)]\s+(.*)", stripped)
        if match:
            if current is not None:
                items.append("\n".join(part for part in current if part is not None).strip())
            current = [match.group(2).strip()]
            continue

        if current is not None:
            if re.match(r"^#{1,6}\s", stripped):
                items.append("\n".join(part for part in current if part is not None).strip())
                current = None
                continue
            bullet = re.match(r"^[-*+]\s+(.*)", stripped)
            if bullet:
                current.append(f"- {bullet.group(1).strip()}")
            else:
                current.append(stripped)

    if current is not None:
        items.append("\n".join(part for part in current if part is not None).strip())
    return [item for item in items if item]


def _parse_unordered_list(content: str) -> list[str]:
    """Return bullet list items detected in the content."""
    items: list[str] = []
    current: list[str] | None = None
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped and current is not None:
            current.append("")
            continue

        match = re.match(r"^[-*+]\s+(.*)", stripped)
        if match:
            if current is not None:
                items.append("\n".join(part for part in current if part is not None).strip())
            current = [match.group(1).strip()]
            continue

        if current is not None:
            if re.match(r"^#{1,6}\s", stripped):
                items.append("\n".join(part for part in current if part is not None).strip())
                current = None
                continue
            nested = re.match(r"^(?:[-*+]|\d+[\.)])\s+(.*)", stripped)
            if nested:
                current.append(f"- {nested.group(1).strip()}")
            else:
                current.append(stripped)

    if current is not None:
        items.append("\n".join(part for part in current if part is not None).strip())
    return [item for item in items if item]


def _normalise_list_items(items: Iterable[str]) -> list[dict[str, Any]]:
    """Convert raw markdown list entries into structured payloads."""

    def extract_code_blocks(text: str) -> tuple[str, list[dict[str, str]]]:
        code_blocks: list[dict[str, str]] = []

        def _collect(match: re.Match[str]) -> str:
            language = (match.group(1) or "text").strip()
            code = match.group(2).strip()
            code_blocks.append({"language": language, "code": code})
            return ""

        cleaned = re.sub(r"```(\w+)?\n(.*?)```", _collect, text, flags=re.DOTALL)
        return cleaned.strip(), code_blocks

    normalised: list[dict[str, Any]] = []
    for raw in items:
        body, code_blocks = extract_code_blocks(raw)

        title: str | None = None
        description = body.strip()

        heading_match = re.match(r"^\*\*(.+?)\*\*:?\s*(.*)", description, re.DOTALL)
        if heading_match:
            title = heading_match.group(1).strip()
            description = heading_match.group(2).strip()

        bullet_points: list[str] = []
        remaining_lines: list[str] = []
        for line in description.splitlines():
            stripped = line.strip()
            bullet_match = re.match(r"^-\s+(.*)", stripped)
            if bullet_match:
                bullet_points.append(bullet_match.group(1).strip())
            else:
                remaining_lines.append(line)

        clean_description = "\n".join(line for line in remaining_lines if line.strip()).strip()

        entry: dict[str, Any] = {
            "raw": raw,
            "title": title or "",
            "description": clean_description or "",
            "bullets": bullet_points,
            "code_blocks": code_blocks,
        }
        normalised.append(entry)

    return normalised


def _parse_code_block(content: str) -> dict[str, Any] | None:
    """Extract language, code, and optionally parsed data from fenced blocks."""
    match = re.search(r"```(\w+)?\n(.*?)```", content, re.DOTALL)
    if not match:
        return None
    language = (match.group(1) or "text").strip()
    code = match.group(2).strip()
    payload: dict[str, Any] = {"language": language, "code": code}
    if language.lower() in {"json", "javascript"}:
        parsed = _parse_json(code)
        if parsed is None:
            parsed = _parse_json(code.strip("`;"))
        if parsed is not None:
            payload["parsed"] = parsed
    return payload


def _looks_like_table(content: str) -> bool:
    """Return True when the content matches a table representation."""
    lowered = content.lower()
    if "<table" in lowered and "</table>" in lowered:
        return True
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for idx in range(len(lines) - 1):
        header = lines[idx]
        separator = lines[idx + 1]
        if header.count("|") >= 2 and re.match(r"^\|?\s*:?-{3,}.*", separator):
            return True
    return False


def _parse_json(content: str) -> object | None:
    """Return parsed JSON when the content is a valid payload, else None."""
    try:
        parsed = json.loads(content)
    except ValueError:
        try:
            parsed = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _looks_like_chart_spec(payload: object) -> bool:
    """Heuristically determine whether parsed JSON resembles a chart spec."""
    if not isinstance(payload, dict):
        return False

    chart_type = payload.get("type") or payload.get("chartType")
    data = payload.get("data") or payload.get("datasets") or payload.get("series")

    if isinstance(chart_type, str) and data:
        return True

    if {"mark", "encoding"}.issubset(payload.keys()):
        return True

    chart_keys = {"datasets", "series", "axes", "scales"}
    if any(key in payload for key in chart_keys):
        return True

    return False


def _looks_like_html(content: str) -> bool:
    """Return True if the content appears to contain generic HTML."""
    return bool(re.search(r"<[^>]+>", content) and re.search(r"</[^>]+>", content))


def _looks_like_markdown(content: str) -> bool:
    """Return True when the content contains common Markdown features."""
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("# ", "## ", "### ", "- ", "* ", "> ", "1. ")):
            return True
        if re.search(r"\[[^\]]+\]\([^)]+\)", content):
            return True
        if "```" in content:
            return True
    return False
