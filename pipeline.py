import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def split_pdf_into_chunks(pdf_path: Path, chunk_size: int = 200) -> List[Path]:
    """将大 PDF 分割成多个小 PDF，每个不超过 chunk_size 页"""
    total_pages = pdf_page_count(pdf_path)
    if total_pages <= chunk_size:
        return [pdf_path]

    pdf = fitz.open(str(pdf_path))
    chunk_paths = []

    try:
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)
            new_pdf = fitz.open()

            new_pdf.insert_pdf(pdf, from_page=start_page, to_page=end_page - 1)

            chunk_path = pdf_path.parent / f"{pdf_path.stem}_part{start_page // chunk_size + 1}{pdf_path.suffix}"
            new_pdf.save(str(chunk_path))
            new_pdf.close()
            chunk_paths.append(chunk_path)
    finally:
        pdf.close()

    return chunk_paths

#确保页码从1开始，而不是0
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
) -> List[Document]:
    mineru_dir = output_dir / "mineru_output"
    image_dir = mineru_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_docs = []
    total_pages = pdf_page_count(pdf_path)

    for i, doc in enumerate(docs):
        page_no = normalize_page_no(
            doc.metadata.get("page") or doc.metadata.get("page_number"),
            i + 1,
        )
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
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "split_pages": split_pages,
            "page_count": total_pages,
            "has_image": bool(image_paths),
        })
        if split_pages:
            doc.metadata["page"] = page_no
            doc.metadata["pages"] = [page_no]
        else:
            doc.metadata["pages"] = list(range(1, total_pages + 1))
        # if image_paths:
        #     doc.metadata["image_paths"] = "|".join(dict.fromkeys(image_paths))

        processed_docs.append(doc)

    return processed_docs


def parse_pdf_with_mineru(
    pdf_path: Path,
    output_dir: Path,
    status_callback=None,
    parse_mode: str = PARSE_MODE_FAST,
) -> List[Document]:
    if parse_mode not in VALID_PARSE_MODES:
        raise ValueError(f"Unknown parse_mode: {parse_mode}")

    split_pages = parse_mode == PARSE_MODE_PRECISE

    # 检查 PDF 页数，如果超过 200 页则分割处理
    total_pages = pdf_page_count(pdf_path)
    if total_pages > 200:
        print(f"[pipeline] PDF has {total_pages} pages, splitting into chunks of 200")
        if status_callback:
            status_callback(f"PDF 共 {total_pages} 页，超过 200 页限制，正在分割处理...")

        chunk_paths = split_pdf_into_chunks(pdf_path, chunk_size=200)
        print(f"[pipeline] Split into {len(chunk_paths)} chunks")

        all_docs = []
        for i, chunk_path in enumerate(chunk_paths):
            if status_callback:
                status_callback(f"正在解析第 {i + 1}/{len(chunk_paths)} 部分 (每部分≤200页)...")
            print(f"[pipeline] Parsing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

            loader = MinerULoader(
                source=str(chunk_path),
                mode="precision",
                token=os.getenv("MINERU_TOKEN"),
                timeout=30000,
                split_pages=split_pages,
            )
            chunk_docs = loader.load()

            # 调整页码偏移
            page_offset = i * 200
            for doc in chunk_docs:
                if "page" in doc.metadata and doc.metadata["page"] is not None:
                    doc.metadata["page"] = doc.metadata["page"] + page_offset

            all_docs.extend(chunk_docs)

            # 删除临时分割文件
            if chunk_path != pdf_path:
                chunk_path.unlink()

        docs = all_docs
        print(f"[pipeline] All chunks parsed, total docs: {len(docs)}")
        if status_callback:
            status_callback(f"MinerU 解析完成，返回 {len(docs)} 个 Document，正在保存图片")

        docs = save_images_and_fix_paths(
            docs=docs,
            pdf_path=pdf_path,
            output_dir=output_dir,
            split_pages=split_pages,
        )

        if split_pages or len(docs) <= 1:
            return docs

        merged_text = "\n\n".join(doc.page_content for doc in docs)
        merged_meta = dict(docs[0].metadata)
        return [Document(page_content=merged_text, metadata=merged_meta)]

    # -----单文件处理逻辑--------
    if status_callback:
        if split_pages:
            status_callback("正在调用 MinerU API 精准分页解析 PDF")
        else:
            status_callback("正在调用 MinerU API 快速整篇解析 PDF")
    print(f"[pipeline] Starting MinerU parsing: {pdf_path}")
    print(f"[pipeline] parse_mode: {parse_mode}, split_pages: {split_pages}")
    print(f"[pipeline] MINERU_TOKEN exists: {bool(os.getenv('MINERU_TOKEN'))}")

    loader = MinerULoader(
        source=str(pdf_path),
        mode="precision",
        token=os.getenv("MINERU_TOKEN"),
        timeout=30000,
        split_pages=split_pages,
    )
    docs = loader.load()
    print(f"[pipeline] MinerU returned docs: {len(docs)}")
    if status_callback:
        status_callback(f"MinerU 解析完成，返回 {len(docs)} 个 Document，正在保存图片")

    docs = save_images_and_fix_paths(
        docs=docs,
        pdf_path=pdf_path,
        output_dir=output_dir,
        split_pages=split_pages,
    )

    if split_pages or len(docs) <= 1:
        return docs

    merged_text = "\n\n".join(doc.page_content for doc in docs)
    merged_meta = dict(docs[0].metadata)
    return [Document(page_content=merged_text, metadata=merged_meta)]


#------------------------------------
#删除风景图
#没有detail信息的图片
#------------------------------------
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

#对每一个docs执行clean_images()
def clean_docs(docs: List[Document]) -> List[Document]:
    cleaned_docs = []
    for doc in docs:
        cleaned_text = clean_images(doc.page_content)
        if not cleaned_text.strip():
            continue
        doc.page_content = cleaned_text
        cleaned_docs.append(doc)
    return cleaned_docs

#对于spilt_pages = True情况用
def merge_page_docs(docs: List[Document], pdf_path: Path) -> List[Document]:
    merged_parts = []
    for i, doc in enumerate(docs):
        page_no = doc.metadata.get("page") or doc.metadata.get("page_number") or i + 1
        merged_parts.append(
            f"\n\n<!-- PAGE_START: {page_no} -->\n\n"
            + doc.page_content
            + f"\n\n<!-- PAGE_END: {page_no} -->\n\n"
        )

    merged_doc = Document(
        page_content="\n".join(merged_parts),
        metadata={
            "source": str(pdf_path),
            "filename": pdf_path.name,
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

#========================================================
#快速模式页码对齐
#========================================================
def normalize_text(text: str) -> str:
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text.lower()


def extract_pdf_page_texts(pdf_path: Path) -> Dict[int, str]:
    pdf = fitz.open(str(pdf_path))
    page_texts = {}
    try:
        for i, page in enumerate(pdf, start=1):
            page_texts[i] = normalize_text(page.get_text("text"))
    finally:
        pdf.close()
    return page_texts


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


#========================================================
#splitter后做
#========================================================

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


def is_tiny_header_only(text: str, max_chars: int = 80) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return len(text.strip()) <= max_chars and len(lines) == 1 and lines[0].startswith("#")


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

        if len(buffer_text) + len(text) <= max_size or is_tiny_header_only(buffer_text):
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
    #每个 Chunk 最大长度约为 1200 个字符，相邻 Chunk 保留约 200 个字符的重叠区域
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    final_chunks = []
    for source_doc in docs:
        header_chunks = markdown_splitter.split_text(source_doc.page_content)
        for chunk in header_chunks:
            chunk.metadata.update(source_doc.metadata)
            if page_texts is not None:
                pages = guess_pages_for_chunk(chunk.page_content, page_texts)
            else:
                pages = get_pages_from_text(chunk.page_content)
            if pages:
                chunk.metadata["page"] = pages[0]
                chunk.metadata["pages"] = pages
            elif page_texts is not None:
                chunk.metadata.pop("page", None)
                chunk.metadata.pop("pages", None)

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


def run_preprocessing(
    pdf_path: Path,
    output_dir: Path,
    status_callback=None,
    parse_mode: str = PARSE_MODE_FAST,
) -> List[Document]:
    if parse_mode not in VALID_PARSE_MODES:
        raise ValueError(f"Unknown parse_mode: {parse_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_raw = parse_pdf_with_mineru(
        pdf_path,
        output_dir,
        status_callback=status_callback,
        parse_mode=parse_mode,
    )
    save_pickle(docs_raw, output_dir / "docs_raw.pkl")

    if parse_mode == PARSE_MODE_PRECISE:
        if status_callback:
            status_callback("正在合并分页 Document 并写入 PAGE_START 标记")
        docs_for_cleaning = merge_page_docs(docs_raw, pdf_path)
        save_pickle(docs_for_cleaning, output_dir / "docs_merged.pkl")
        page_texts = None
    else:
        if status_callback:
            status_callback("正在用 PyMuPDF 提取 PDF 页文本，稍后为 chunk 映射页码")
        page_texts = extract_pdf_page_texts(pdf_path)
        docs_for_cleaning = docs_raw
        save_pickle(docs_for_cleaning, output_dir / "docs_merged.pkl")

    if status_callback:
        status_callback("正在清洗图片和无效内容")
    docs_cleaned = clean_docs(docs_for_cleaning)
    save_pickle(docs_cleaned, output_dir / "docs_cleaned.pkl")

    if status_callback:
        if parse_mode == PARSE_MODE_PRECISE:
            status_callback("正在按 PAGE_START 拆回分页并构建 chunk")
        else:
            status_callback("正在按标题构建 chunk，并用 PyMuPDF 文本匹配映射页码")
    chunks = build_chunks(
        docs_cleaned,
        page_texts=page_texts if parse_mode == PARSE_MODE_FAST else None,
    )
    save_pickle(chunks, output_dir / "chunks.pkl")

    if status_callback:
        status_callback("正在用 LLM 清理无用 chunk")
    clean_chunks = clean_useless_chunks_with_llm(chunks, output_dir)
    save_pickle(clean_chunks, output_dir / "chunks_clean_for_rag.pkl")
    return clean_chunks
