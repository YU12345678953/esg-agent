import html
import re
from html.parser import HTMLParser
from typing import Dict, List


FENCED_BLOCK_PATTERN = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)")
HTML_TABLE_PATTERN = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
PIPE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def sanitize_markdown_for_pypandoc(markdown_text: str) -> str:
    """
    Normalize LLM-produced Markdown into a conservative Pandoc-friendly subset.

    The main target is Word conversion via pypandoc, where malformed pipe
    tables and HTML image tables are common failure points.
    """
    text = _normalize_text(markdown_text)
    text = _unwrap_markdown_code_fence(text)

    parts = FENCED_BLOCK_PATTERN.split(text)
    sanitized_parts = []
    for part in parts:
        if not part:
            continue
        if FENCED_BLOCK_PATTERN.fullmatch(part):
            sanitized_parts.append(part.strip())
            continue
        sanitized_parts.append(_sanitize_markdown_fragment(part))

    text = "\n\n".join(part.strip() for part in sanitized_parts if part.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _normalize_text(text: str) -> str:
    return (
        (text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\xa0", " ")
    )


def _unwrap_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*\n([\s\S]*?)\n```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _sanitize_markdown_fragment(text: str) -> str:
    text = HTML_TABLE_PATTERN.sub(lambda match: _convert_html_table(match.group(0)), text)
    text = _convert_centered_html_images(text)
    text = _remove_orphan_image_attrs(text)
    lines = [line.rstrip() for line in text.split("\n")]
    lines = _normalize_pipe_tables(lines)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _convert_html_table(table_html: str) -> str:
    parser = _TableParser()
    parser.feed(table_html)
    rows = parser.rows
    if not rows:
        return ""

    if any(cell["images"] for row in rows for cell in row):
        blocks = []
        for row in rows:
            for cell in row:
                cell_text = _clean_inline_text(cell["text"])
                for image in cell["images"]:
                    alt = image.get("alt") or cell_text or "图片"
                    src = image.get("src", "").strip()
                    if not src:
                        continue
                    width = _pandoc_width(image.get("width", ""))
                    blocks.append(f"![{_escape_image_alt(alt)}]({_escape_image_src(src)}){width}".strip())
        return "\n\n".join(blocks)

    table_rows = []
    max_cols = max(len(row) for row in rows)
    for row in rows:
        values = [_clean_table_cell(cell["text"]) for cell in row]
        values.extend([""] * (max_cols - len(values)))
        table_rows.append(values)
    return _build_pipe_table(table_rows)


def _convert_centered_html_images(text: str) -> str:
    text = re.sub(
        r"<p\b[^>]*align=[\"']center[\"'][^>]*>\s*(<img\b[^>]*>)\s*</p>",
        lambda match: _convert_img_tag(match.group(1)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<p\b[^>]*align=[\"']center[\"'][^>]*>\s*<em>([\s\S]*?)</em>\s*</p>",
        lambda match: f"*{_clean_inline_text(match.group(1))}*",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _convert_img_tag(img_tag: str) -> str:
    attrs = _parse_attrs(img_tag)
    src = attrs.get("src", "").strip()
    if not src:
        return ""
    alt = attrs.get("alt", "图片")
    width = _pandoc_width(attrs.get("width", ""))
    return f"![{_escape_image_alt(alt)}]({_escape_image_src(src)}){width}".strip()


def _remove_orphan_image_attrs(text: str) -> str:
    text = re.sub(
        r"(?m)^(\s*)\{(?:width|height)=[^}\n]+\}\s*(图\s*[:：])",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*)\*\s*\{(?:width|height)=[^}\n]+\}\s*(图\s*[:：][^*\n]*)\*?\s*$",
        r"\1*\2*",
        text,
    )
    return text


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Dict[str, object]]] = []
        self._current_row: List[Dict[str, object]] | None = None
        self._current_cell: Dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = {"text": "", "images": []}
        elif tag == "img" and self._current_cell is not None:
            images = self._current_cell["images"]
            assert isinstance(images, list)
            images.append({key.lower(): value or "" for key, value in attrs})
        elif tag == "br" and self._current_cell is not None:
            self._current_cell["text"] = f"{self._current_cell['text']} "

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(self._current_cell)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell["text"] = f"{self._current_cell['text']} {data}"


def _parse_attrs(tag_text: str) -> Dict[str, str]:
    return {
        key.lower(): html.unescape(value)
        for key, _, value in re.findall(r"([:\w-]+)\s*=\s*([\"'])(.*?)\2", tag_text)
    }


def _normalize_pipe_tables(lines: List[str]) -> List[str]:
    output = []
    i = 0
    while i < len(lines):
        if not _looks_like_pipe_table_line(lines[i]):
            output.append(lines[i])
            i += 1
            continue

        block = []
        while i < len(lines) and _looks_like_pipe_table_line(lines[i]):
            block.append(lines[i])
            i += 1

        if len(block) < 2:
            output.extend(block)
            continue
        output.extend(_normalize_pipe_table_block(block))
    return output


def _looks_like_pipe_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.count("|") >= 2 and not stripped.startswith("```")


def _normalize_pipe_table_block(lines: List[str]) -> List[str]:
    rows = [_split_pipe_row(line) for line in lines if not PIPE_SEPARATOR_PATTERN.match(line)]
    if len(rows) < 2:
        return lines
    max_cols = max(len(row) for row in rows)
    normalized_rows = []
    for row in rows:
        cells = [_clean_table_cell(cell) for cell in row]
        cells.extend([""] * (max_cols - len(cells)))
        normalized_rows.append(cells[:max_cols])
    return _build_pipe_table(normalized_rows).split("\n")


def _split_pipe_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _build_pipe_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    header = rows[0] + [""] * (max_cols - len(rows[0]))
    body = [row + [""] * (max_cols - len(row)) for row in rows[1:]]
    lines = [
        "| " + " | ".join(_clean_table_cell(cell) or " " for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(max_cols)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_clean_table_cell(cell) or " " for cell in row) + " |")
    return "\n".join(lines)


def _clean_table_cell(text: object) -> str:
    clean = _clean_inline_text(str(text or ""))
    clean = clean.replace("|", "\\|")
    return clean


def _clean_inline_text(text: str) -> str:
    text = TAG_PATTERN.sub(" ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _escape_image_alt(text: str) -> str:
    return _clean_inline_text(text).replace("[", "\\[").replace("]", "\\]")


def _escape_image_src(src: str) -> str:
    return src.replace(" ", "%20")


def _pandoc_width(width: str) -> str:
    width = (width or "").strip()
    if not width:
        return ""
    if width.endswith("%"):
        try:
            value = float(width[:-1]) / 100
        except ValueError:
            return ""
        return f"{{width={value:.0%}}}"
    if width.isdigit():
        return f"{{width={width}px}}"
    return ""
