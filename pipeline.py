import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mineru import MinerULoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

IMAGE_LINK_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
IMAGE_REF_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
PAGE_BLOCK_PATTERN = re.compile(
    r"<!--\s*PAGE_START:\s*(\d+)\s*-->\s*(.*?)\s*<!--\s*PAGE_END:\s*\1\s*-->",
    re.DOTALL,
)
PAGE_START_PATTERN = re.compile(r"<!--\s*PAGE_START:\s*(\d+)\s*-->")

PARSE_MODE_FAST = "fast"
PARSE_MODE_PRECISE = "precise"
VALID_PARSE_MODES = {PARSE_MODE_FAST, PARSE_MODE_PRECISE}
MINERU_MAX_PAGES_PER_REQUEST = int(os.getenv("MINERU_MAX_PAGES_PER_REQUEST", "200"))


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def pdf_page_count(pdf_path: Path) -> int:
    pdf = fitz.open(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def split_pdf_for_mineru(pdf_path: Path, output_dir: Path, max_pages: int = MINERU_MAX_PAGES_PER_REQUEST) -> List[Dict[str, Any]]:
    total_pages = pdf_page_count(pdf_path)
    if total_pages <= max_pages:
        return [{
            "part_path": pdf_path,
            "page_offset": 0,
            "part_index": 0,
            "part_start_page": 1,
            "part_end_page": total_pages,
            "original_page_count": total_pages,
        }]

    parts_dir = output_dir / "pdf_parts" / pdf_path.stem
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    source_pdf = fitz.open(str(pdf_path))
    try:
        for part_index, start_page in enumerate(range(1, total_pages + 1, max_pages)):
            end_page = min(start_page + max_pages - 1, total_pages)
            part_path = parts_dir / f"{pdf_path.stem}_part_{part_index + 1:03d}_pages_{start_page}_{end_page}.pdf"
            if not part_path.exists():
                part_pdf = fitz.open()
                try:
                    part_pdf.insert_pdf(
                        source_pdf,
                        from_page=start_page - 1,
                        to_page=end_page - 1,
                    )
                    part_pdf.save(str(part_path))
                finally:
                    part_pdf.close()
            parts.append({
                "part_path": part_path,
                "page_offset": start_page - 1,
                "part_index": part_index,
                "part_start_page": start_page,
                "part_end_page": end_page,
                "original_page_count": total_pages,
            })
    finally:
        source_pdf.close()
    return parts


def normalize_page_no(raw_page: Any, fallback: int) -> int:
    try:
        page_no = int(raw_page)
    except Exception:
        page_no = fallback
    return page_no + 1 if page_no == 0 else page_no


def save_images_and_fix_paths(
    docs: List[Document],
    pdf_path: Path,
    output_dir: Path,
    split_pages: bool,
    original_pdf_path: Path | None = None,
    source_pdf_index: int = 0,
    page_offset: int = 0,
    original_page_count: int | None = None,
    part_index: int = 0,
) -> List[Document]:
    mineru_dir = output_dir / "mineru_output"
    image_dir = mineru_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_docs = []
    original_pdf_path = original_pdf_path or pdf_path
    total_pages = original_page_count or pdf_page_count(original_pdf_path)

    for i, doc in enumerate(docs):
        part_page_no = normalize_page_no(
            doc.metadata.get("page") or doc.metadata.get("page_number"),
            i + 1,
        )
        page_no = part_page_no + page_offset
        images = doc.metadata.get("images") or []
        image_name_to_abs = {}

        for img in images:
            img_name = img.name
            img_path = (image_dir / img_name).resolve()
            with open(img_path, "wb") as f:
                f.write(img.data)
            image_name_to_abs[img_name] = str(img_path)

        doc.metadata.pop("images", None)

        def replace_md_image(match):
            alt = match.group(1)
            raw_path = match.group(2)
            img_name = Path(raw_path).name
            fixed_path = image_name_to_abs.get(
                img_name,
                str((image_dir / img_name).resolve()),
            )
            return f"![{alt}]({fixed_path})"

        doc.page_content = IMAGE_LINK_PATTERN.sub(replace_md_image, doc.page_content)

        image_paths = []
        for raw_path in IMAGE_REF_PATTERN.findall(doc.page_content):
            img_path = Path(raw_path)
            if img_path.exists():
                image_paths.append(str(img_path))

        doc.metadata.update({
            "source": str(original_pdf_path),
            "filename": original_pdf_path.name,
            "source_pdf_path": str(original_pdf_path),
            "source_pdf_name": original_pdf_path.name,
            "source_pdf_index": source_pdf_index,
            "mineru_pdf_part_path": str(pdf_path),
            "mineru_pdf_part_index": part_index,
            "mineru_pdf_part_page": part_page_no,
            "page_offset": page_offset,
            "split_pages": split_pages,
            "page_count": total_pages,
            "has_image": bool(image_paths),
        })
        if split_pages:
            doc.metadata["page"] = page_no
            doc.metadata["pages"] = [page_no]
        else:
            doc.metadata["pages"] = list(range(1, total_pages + 1))
        if image_paths:
            doc.metadata["image_paths"] = "|".join(dict.fromkeys(image_paths))

        processed_docs.append(doc)

    return processed_docs


def parse_pdf_with_mineru(
    pdf_path: Path,
    output_dir: Path,
    status_callback=None,
    parse_mode: str = PARSE_MODE_FAST,
    source_pdf_index: int = 0,
) -> List[Document]:
    if parse_mode not in VALID_PARSE_MODES:
        raise ValueError(f"Unknown parse_mode: {parse_mode}")

    split_pages = parse_mode == PARSE_MODE_PRECISE
    pdf_parts = split_pdf_for_mineru(pdf_path, output_dir)
    all_docs = []
    if status_callback:
        if len(pdf_parts) > 1:
            status_callback(f"{pdf_path.name} 共 {pdf_parts[0]['original_page_count']} 页，已按每 {MINERU_MAX_PAGES_PER_REQUEST} 页拆成 {len(pdf_parts)} 个 PDF 分别调用 MinerU")
        elif split_pages:
            status_callback(f"正在调用 MinerU API 精准分页解析 PDF：{pdf_path.name}")
        else:
            status_callback(f"正在调用 MinerU API 快速整篇解析 PDF：{pdf_path.name}")

    for part in pdf_parts:
        part_path = Path(part["part_path"])
        part_output_dir = output_dir / "mineru_parts" / pdf_path.stem / f"part_{part['part_index'] + 1:03d}"
        print(f"[pipeline] Starting MinerU parsing: {part_path}")
        print(f"[pipeline] original_pdf: {pdf_path}, pages: {part['part_start_page']}-{part['part_end_page']}")
        print(f"[pipeline] parse_mode: {parse_mode}, split_pages: {split_pages}")
        print(f"[pipeline] MINERU_TOKEN exists: {bool(os.getenv('MINERU_TOKEN'))}")

        if status_callback:
            status_callback(
                f"正在调用 MinerU：{pdf_path.name} 第 {part['part_start_page']}-{part['part_end_page']} 页"
            )
        loader = MinerULoader(
            source=str(part_path),
            mode="precision",
            token=os.getenv("MINERU_TOKEN"),
            timeout=30000,
            split_pages=split_pages,
        )
        docs = loader.load()
        print(f"[pipeline] MinerU returned docs: {len(docs)}")

        docs = save_images_and_fix_paths(
            docs=docs,
            pdf_path=part_path,
            output_dir=part_output_dir,
            split_pages=split_pages,
            original_pdf_path=pdf_path,
            source_pdf_index=source_pdf_index,
            page_offset=int(part["page_offset"]),
            original_page_count=int(part["original_page_count"]),
            part_index=int(part["part_index"]),
        )
        all_docs.extend(docs)

    docs = all_docs
    if status_callback:
        status_callback(f"MinerU 解析完成，返回 {len(docs)} 个 Document，正在整理图片和页码")

    if split_pages or len(docs) <= 1:
        return docs

    merged_text = "\n\n".join(doc.page_content for doc in docs)
    merged_meta = dict(docs[0].metadata)
    return [Document(page_content=merged_text, metadata=merged_meta)]


def clean_images(text: str) -> str:
    pattern = r"""
    (
        !\[\]\([^)]+\)
        \s*
        <details>\s*
        <summary>\s*natural_image\s*</summary>
        .*?
        </details>
    )
    |
    (
        !\[\]\([^)]+\)
        (?!\s*<details>\s*<summary>)
    )
    """
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE | re.VERBOSE)


def clean_docs(docs: List[Document]) -> List[Document]:
    cleaned_docs = []
    for doc in docs:
        cleaned_text = clean_images(doc.page_content)
        if not cleaned_text.strip():
            continue
        doc.page_content = cleaned_text
        cleaned_docs.append(doc)
    return cleaned_docs


def merge_page_docs(docs: List[Document], pdf_path: Path) -> List[Document]:
    merged_parts = []
    for i, doc in enumerate(docs):
        page_no = doc.metadata.get("page") or doc.metadata.get("page_number") or i + 1
        merged_parts.append(
            f"\n\n<!-- PAGE_START: {page_no} -->\n\n"
            + doc.page_content
            + f"\n\n<!-- PAGE_END: {page_no} -->\n\n"
        )

    first_meta = dict(docs[0].metadata) if docs else {}
    merged_doc = Document(
        page_content="\n".join(merged_parts),
        metadata={
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "source_pdf_path": first_meta.get("source_pdf_path", str(pdf_path)),
            "source_pdf_name": first_meta.get("source_pdf_name", pdf_path.name),
            "source_pdf_index": first_meta.get("source_pdf_index", 0),
            "split_pages": False,
            "page_count": len(docs),
            "pages": list(range(1, len(docs) + 1)),
        },
    )
    return [merged_doc]


def split_merged_docs_to_page_docs(docs: List[Document]) -> List[Document]:
    page_docs = []
    for merged_doc in docs:
        matches = PAGE_BLOCK_PATTERN.findall(merged_doc.page_content)
        if not matches:
            page_docs.append(merged_doc)
            continue

        for page_no, page_text in matches:
            page_no = int(page_no)
            metadata = dict(merged_doc.metadata)
            metadata["page"] = page_no
            metadata["pages"] = [page_no]
            page_docs.append(Document(
                page_content=page_text.strip(),
                metadata=metadata,
            ))
    return page_docs


def get_pages_from_text(text: str) -> List[int]:
    pages = [int(x) for x in PAGE_START_PATTERN.findall(text)]
    return list(dict.fromkeys(pages))


def normalize_text(text: str) -> str:
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text.lower()


def replace_span_with_spaces(match: re.Match) -> str:
    return " " * (match.end() - match.start())


def strip_non_text_preserve_offsets(text: str) -> str:
    text = re.sub(
        r"<details>.*?</details>",
        replace_span_with_spaces,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", replace_span_with_spaces, text)
    text = re.sub(r"<[^>]+>", replace_span_with_spaces, text)
    return text


def normalize_with_offsets(text: str) -> tuple[str, List[int]]:
    stripped = strip_non_text_preserve_offsets(text)
    chars = []
    offsets = []
    for index, char in enumerate(stripped):
        if char == "_":
            continue
        if char.isalnum():
            chars.append(char.lower())
            offsets.append(index)
    return "".join(chars), offsets


def extract_pdf_page_texts(pdf_path: Path) -> Dict[int, str]:
    pdf = fitz.open(str(pdf_path))
    page_texts = {}
    try:
        for i, page in enumerate(pdf, start=1):
            page_texts[i] = normalize_text(page.get_text("text"))
    finally:
        pdf.close()
    return page_texts


def page_anchors(page_text: str, anchor_len: int = 28) -> List[str]:
    text = page_text.strip()
    if len(text) < 12:
        return []
    if len(text) <= anchor_len:
        return [text]

    starts = [
        0,
        max(0, len(text) // 4 - anchor_len // 2),
        max(0, len(text) // 2 - anchor_len // 2),
        max(0, len(text) * 3 // 4 - anchor_len // 2),
        max(0, len(text) - anchor_len),
    ]
    anchors = []
    for start in starts:
        anchor = text[start:start + anchor_len]
        if len(anchor) >= 12 and anchor not in anchors:
            anchors.append(anchor)
    return anchors


def find_ordered_page_positions(
    markdown_norm: str,
    page_texts: Dict[int, str],
) -> tuple[Dict[int, int], Dict[int, str]]:
    positions: Dict[int, int] = {}
    sources: Dict[int, str] = {}
    cursor = 0

    for page_no in sorted(page_texts):
        found_positions = []
        for anchor in page_anchors(page_texts[page_no]):
            pos = markdown_norm.find(anchor, cursor)
            if pos != -1:
                found_positions.append(pos)

        if found_positions:
            pos = min(found_positions)
            positions[page_no] = pos
            sources[page_no] = "anchor"
            cursor = max(cursor, pos + 1)

    return positions, sources


def fill_missing_page_positions(
    page_numbers: List[int],
    found_positions: Dict[int, int],
    norm_length: int,
) -> tuple[Dict[int, int], Dict[int, str]]:
    if not page_numbers:
        return {}, {}

    positions = dict(found_positions)
    sources = {
        page_no: "anchor"
        for page_no in found_positions
    }
    first_page = page_numbers[0]
    positions[first_page] = 0
    sources.setdefault(first_page, "document_start")

    known_pages = sorted(positions)
    for page_no in page_numbers:
        if page_no in positions:
            continue

        previous_known = [
            known_page for known_page in known_pages
            if known_page < page_no
        ]
        next_known = [
            known_page for known_page in known_pages
            if known_page > page_no
        ]

        if previous_known and next_known:
            prev_page = previous_known[-1]
            next_page = next_known[0]
            span_pages = next_page - prev_page
            span_chars = positions[next_page] - positions[prev_page]
            ratio = (page_no - prev_page) / span_pages
            positions[page_no] = positions[prev_page] + int(span_chars * ratio)
        elif previous_known:
            prev_page = previous_known[-1]
            remaining_pages = max(1, page_numbers[-1] - prev_page + 1)
            remaining_chars = max(0, norm_length - positions[prev_page])
            ratio = (page_no - prev_page) / remaining_pages
            positions[page_no] = positions[prev_page] + int(remaining_chars * ratio)
        elif next_known:
            next_page = next_known[0]
            ratio = (page_no - first_page) / max(1, next_page - first_page)
            positions[page_no] = int(positions[next_page] * ratio)
        else:
            ratio = (page_no - first_page) / max(1, page_numbers[-1] - first_page + 1)
            positions[page_no] = int(norm_length * ratio)

        sources[page_no] = "interpolated"
        known_pages = sorted(positions)

    last_pos = 0
    for page_no in page_numbers:
        positions[page_no] = max(last_pos, min(positions[page_no], norm_length))
        last_pos = positions[page_no]

    return positions, sources


def original_index_from_norm_pos(norm_pos: int, offsets: List[int], original_length: int) -> int:
    if norm_pos <= 0:
        return 0
    if norm_pos >= len(offsets):
        return original_length
    return offsets[norm_pos]


def align_fast_doc_to_pages(
    docs: List[Document],
    pdf_path: Path,
    page_texts: Dict[int, str],
    output_dir: Path,
) -> List[Document]:
    if not docs:
        return docs

    if len(docs) == 1:
        source_doc = docs[0]
    else:
        merged_text = "\n\n".join(doc.page_content for doc in docs)
        source_doc = Document(page_content=merged_text, metadata=dict(docs[0].metadata))

    markdown_text = source_doc.page_content
    markdown_norm, offsets = normalize_with_offsets(markdown_text)
    page_numbers = sorted(page_texts)

    found_positions, found_sources = find_ordered_page_positions(markdown_norm, page_texts)
    positions, sources = fill_missing_page_positions(
        page_numbers,
        found_positions,
        len(markdown_norm),
    )
    sources.update(found_sources)

    merged_parts = []
    alignment_rows = []
    for index, page_no in enumerate(page_numbers):
        norm_start = positions.get(page_no, 0)
        if index + 1 < len(page_numbers):
            norm_end = positions.get(page_numbers[index + 1], len(markdown_norm))
        else:
            norm_end = len(markdown_norm)

        orig_start = original_index_from_norm_pos(norm_start, offsets, len(markdown_text))
        orig_end = original_index_from_norm_pos(norm_end, offsets, len(markdown_text))
        page_content = markdown_text[orig_start:orig_end].strip()

        merged_parts.append(
            f"\n\n<!-- PAGE_START: {page_no} -->\n\n"
            + page_content
            + f"\n\n<!-- PAGE_END: {page_no} -->\n\n"
        )
        alignment_rows.append({
            "page": page_no,
            "norm_start": norm_start,
            "norm_end": norm_end,
            "orig_start": orig_start,
            "orig_end": orig_end,
            "source": sources.get(page_no, "unknown"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "page_alignment.json").write_text(
        json.dumps(alignment_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merged_meta = dict(source_doc.metadata)
    merged_meta.update({
        "source": str(pdf_path),
        "filename": pdf_path.name,
        "split_pages": False,
        "page_count": len(page_numbers),
        "pages": page_numbers,
        "page_alignment": "global_anchor",
    })
    return [Document(page_content="\n".join(merged_parts), metadata=merged_meta)]


def sample_pieces(text: str, piece_len: int = 28, max_pieces: int = 12) -> List[str]:
    clean = normalize_text(text)
    if len(clean) <= piece_len:
        return [clean] if len(clean) >= 8 else []

    span = max(1, (len(clean) - piece_len) // max(1, max_pieces - 1))
    pieces = []
    for start in range(0, len(clean) - piece_len + 1, span):
        piece = clean[start:start + piece_len]
        if piece and piece not in pieces:
            pieces.append(piece)
        if len(pieces) >= max_pieces:
            break
    return pieces


def guess_pages_for_chunk(text: str, page_texts: Dict[int, str]) -> List[int]:
    pieces = sample_pieces(text)
    if not pieces:
        return []

    scores = []
    for page_no, page_text in page_texts.items():
        score = 0
        for piece in pieces:
            if piece and piece in page_text:
                score += len(piece)
        if score:
            scores.append((page_no, score))

    if not scores:
        return []

    scores.sort(key=lambda x: x[1], reverse=True)
    best_score = scores[0][1]
    pages = [
        page_no
        for page_no, score in scores
        if score >= max(28, int(best_score * 0.72))
    ]
    return sorted(pages[:3])


def merge_metadata_pages(buffer_meta: Dict[str, Any], chunk_meta: Dict[str, Any]) -> None:
    pages = []
    for meta in (buffer_meta, chunk_meta):
        values = meta.get("pages")
        if values is None:
            values = [meta.get("page")]
        elif not isinstance(values, list):
            values = [values]
        for page in values:
            if page is not None and page not in pages:
                pages.append(page)

    if pages:
        buffer_meta["pages"] = pages
        buffer_meta["page"] = pages[0]

    sources = []
    for meta in (buffer_meta, chunk_meta):
        source_pdf_index = meta.get("source_pdf_index")
        source_pdf_name = meta.get("source_pdf_name") or meta.get("filename")
        source_pdf_path = meta.get("source_pdf_path") or meta.get("source")
        if source_pdf_name or source_pdf_path:
            source = {
                "source_pdf_index": source_pdf_index,
                "source_pdf_name": source_pdf_name,
                "source_pdf_path": source_pdf_path,
            }
            if source not in sources:
                sources.append(source)
    if sources:
        buffer_meta["source_pdfs"] = sources


def is_tiny_header_only(text: str, max_chars: int = 80) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return len(text.strip()) <= max_chars and len(lines) == 1 and lines[0].startswith("#")


def same_source_pdf(left_meta: Dict[str, Any] | None, right_meta: Dict[str, Any]) -> bool:
    if left_meta is None:
        return True
    return (
        left_meta.get("source_pdf_index", 0) == right_meta.get("source_pdf_index", 0)
        and (left_meta.get("source_pdf_path") or left_meta.get("source"))
        == (right_meta.get("source_pdf_path") or right_meta.get("source"))
    )


def merge_small_chunks(chunks: List[Document], max_size: int = 2000) -> List[Document]:
    merged = []
    buffer_text = ""
    buffer_meta: Optional[Dict[str, Any]] = None

    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        if not buffer_text:
            buffer_text = text
            buffer_meta = dict(chunk.metadata)
            continue

        can_merge = same_source_pdf(buffer_meta, chunk.metadata) and (
            len(buffer_text) + len(text) <= max_size or is_tiny_header_only(buffer_text)
        )
        if can_merge:
            buffer_text += "\n\n" + text
            old_h1 = buffer_meta.get("Header 1", "") if buffer_meta else ""
            new_h1 = chunk.metadata.get("Header 1", "")
            if buffer_meta is not None and new_h1 and new_h1 not in old_h1:
                buffer_meta["Header 1"] = (old_h1 + " / " + new_h1).strip(" / ")
            if buffer_meta is not None:
                merge_metadata_pages(buffer_meta, chunk.metadata)
        else:
            merged.append(Document(page_content=buffer_text.strip(), metadata=buffer_meta or {}))
            buffer_text = text
            buffer_meta = dict(chunk.metadata)

    if buffer_text:
        merged.append(Document(page_content=buffer_text.strip(), metadata=buffer_meta or {}))

    return merged


def parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"无法解析 JSON:\n{raw}")


def get_cleaner_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CLEAN_LLM_MODEL", "deepseek-v4-pro"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )


def find_useless_chunks(llm: ChatOpenAI, chunks: List[Document], ids: List[int], preview_chars: int = 800) -> List[int]:
    batch_text = ""
    for i in ids:
        chunk = chunks[i]
        header = chunk.metadata.get("Header 1", "")
        preview = chunk.page_content[:preview_chars].replace("\n", " ")
        batch_text += f"""
CHUNK_ID: {i}
HEADER: {header}
CONTENT_PREVIEW:
{preview}

---
"""

    prompt = f"""
你是 ESG 报告 RAG chunk 清洗助手。

下面是一批从企业 ESG / 可持续发展报告中切出来的 chunk。

你的任务不是分类，也不是判断环境/社会/治理。
你的任务只有一个：

找出其中“明显不应该进入 RAG 检索库”的 chunk_id。

请非常保守。
只有在你非常确定该 chunk 对 ESG 正文生成没有帮助时，才删除。

应删除的类型包括：
1. 目录。
2. 索引表、指标索引、GRI/HKEX/交易所指标对照表。
3. 附录。
4. 封面、封底。
6. 只有页眉页脚、页码、图片路径、空标题，没有实质正文。
7. 大量重复、没有实质披露信息的格式性内容。

不要删除：
1. 任何可能包含环境、社会、治理实质披露的 chunk。
2. 任何包含公司措施、目标、数据、风险、治理架构、案例、图表说明的 chunk。
3. 任何你不确定是否有用的 chunk。
4. 只要有一点可能相关，就必须保留。

输出要求：
只输出 JSON，不要解释。

格式：
{{
  "drop_ids": [12, 35, 48]
}}

待判断 chunk：
{batch_text}
"""

    data = parse_json(llm.invoke(prompt).content)
    drop_ids = data.get("drop_ids", [])
    return [
        int(x) for x in drop_ids
        if isinstance(x, int) and x in ids
    ]


def clean_useless_chunks_with_llm(chunks: List[Document], output_dir: Path, batch_size: int = 20) -> List[Document]:
    if os.getenv("SKIP_LLM_CHUNK_CLEAN", "0") == "1":
        return chunks

    llm = get_cleaner_llm()
    all_drop_ids = []
    drop_review = []

    for start in range(0, len(chunks), batch_size):
        ids = list(range(start, min(start + batch_size, len(chunks))))
        try:
            drop_ids = find_useless_chunks(llm, chunks, ids, preview_chars=800)
        except Exception:
            drop_ids = []

        all_drop_ids.extend(drop_ids)
        drop_review.append({
            "batch_start": ids[0],
            "batch_end": ids[-1],
            "drop_ids": drop_ids,
        })
        (output_dir / "drop_review.json").write_text(
            json.dumps(drop_review, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_drop_ids = sorted(set(all_drop_ids))
    (output_dir / "drop_ids.json").write_text(
        json.dumps(all_drop_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    clean_chunks = [
        chunk for i, chunk in enumerate(chunks)
        if i not in all_drop_ids
    ]
    for clean_id, chunk in enumerate(clean_chunks):
        chunk.metadata["clean_chunk_id"] = clean_id
    return clean_chunks


def build_chunks(
    docs: List[Document],
    page_texts: Optional[Dict[int, str]] = None,
) -> List[Document]:
    docs = split_merged_docs_to_page_docs(docs)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1")],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    final_chunks = []
    for source_doc in docs:
        header_chunks = markdown_splitter.split_text(source_doc.page_content)
        for chunk in header_chunks:
            chunk.metadata.update(source_doc.metadata)
            pages = get_pages_from_text(chunk.page_content)
            if not pages and page_texts is not None:
                pages = guess_pages_for_chunk(chunk.page_content, page_texts)
            if pages:
                chunk.metadata["page"] = pages[0]
                chunk.metadata["pages"] = pages

            if len(chunk.page_content) <= 3000:
                final_chunks.append(chunk)
            else:
                split_docs = text_splitter.split_documents([chunk])
                if page_texts is not None:
                    for split_doc in split_docs:
                        split_pages = guess_pages_for_chunk(split_doc.page_content, page_texts)
                        if split_pages:
                            split_doc.metadata["page"] = split_pages[0]
                            split_doc.metadata["pages"] = split_pages
                final_chunks.extend(split_docs)

    chunks = merge_small_chunks(final_chunks, max_size=2000)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)
    return chunks


def normalize_pdf_paths(pdf_paths: Path | Sequence[Path]) -> List[Path]:
    if isinstance(pdf_paths, (str, Path)):
        return [Path(pdf_paths)]
    return [Path(path) for path in pdf_paths]


def run_preprocessing(
    pdf_path: Path | Sequence[Path],
    output_dir: Path,
    status_callback=None,
    parse_mode: str = PARSE_MODE_FAST,
) -> List[Document]:
    if parse_mode not in VALID_PARSE_MODES:
        raise ValueError(f"Unknown parse_mode: {parse_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = normalize_pdf_paths(pdf_path)
    docs_raw = []
    for source_pdf_index, source_pdf_path in enumerate(pdf_paths):
        if status_callback:
            status_callback(f"正在解析第 {source_pdf_index + 1}/{len(pdf_paths)} 个 PDF：{source_pdf_path.name}")
        docs_raw.extend(parse_pdf_with_mineru(
            source_pdf_path,
            output_dir,
            status_callback=status_callback,
            parse_mode=parse_mode,
            source_pdf_index=source_pdf_index,
        ))
    save_pickle(docs_raw, output_dir / "docs_raw.pkl")

    if parse_mode == PARSE_MODE_PRECISE:
        if status_callback:
            status_callback("正在合并分页 Document 并写入 PAGE_START 标记")
        docs_for_cleaning = []
        for source_pdf_index, source_pdf_path in enumerate(pdf_paths):
            source_docs = [
                doc for doc in docs_raw
                if doc.metadata.get("source_pdf_index") == source_pdf_index
            ]
            docs_for_cleaning.extend(merge_page_docs(source_docs, source_pdf_path))
        save_pickle(docs_for_cleaning, output_dir / "docs_merged.pkl")
        page_texts = None
    else:
        if status_callback:
            status_callback("正在用 PyMuPDF 提取 PDF 页文本并全局对齐页码")
        docs_for_cleaning = []
        for source_pdf_index, source_pdf_path in enumerate(pdf_paths):
            source_docs = [
                doc for doc in docs_raw
                if doc.metadata.get("source_pdf_index") == source_pdf_index
            ]
            page_texts = extract_pdf_page_texts(source_pdf_path)
            docs_for_cleaning.extend(align_fast_doc_to_pages(
                source_docs,
                source_pdf_path,
                page_texts,
                output_dir / f"page_alignment_pdf_{source_pdf_index + 1}",
            ))
        save_pickle(docs_for_cleaning, output_dir / "docs_merged.pkl")

    if status_callback:
        status_callback("正在清洗图片和无效内容")
    docs_cleaned = clean_docs(docs_for_cleaning)
    save_pickle(docs_cleaned, output_dir / "docs_cleaned.pkl")

    if status_callback:
        if parse_mode == PARSE_MODE_PRECISE:
            status_callback("正在按 PAGE_START 拆回分页并构建 chunk")
        else:
            status_callback("正在按全局页码标记拆回分页并构建 chunk")
    chunks = build_chunks(docs_cleaned, page_texts=None)
    save_pickle(chunks, output_dir / "chunks.pkl")

    if status_callback:
        status_callback("正在用 LLM 清理无用 chunk")
    clean_chunks = clean_useless_chunks_with_llm(chunks, output_dir)
    save_pickle(clean_chunks, output_dir / "chunks_clean_for_rag.pkl")
    return clean_chunks
