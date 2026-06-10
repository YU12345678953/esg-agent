import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

import fitz
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

try:
    from .graph_runner import run_check_section, run_esg_graph, run_regenerate_section
    from .generation import chat_with_report, load_or_build_vector_store
    from .pipeline import load_pickle
    from .session_store import SessionStore, now_text
    from .word_export import convert_markdown_to_word
except ImportError:
    from graph_runner import run_check_section, run_esg_graph, run_regenerate_section
    from generation import chat_with_report, load_or_build_vector_store
    from pipeline import load_pickle
    from session_store import SessionStore, now_text
    from word_export import convert_markdown_to_word

AGENT_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = AGENT_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(AGENT_DIR / ".env")

app = FastAPI(title="ESG Agent API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_STORE = SessionStore(SESSIONS_DIR)
TEXT_LLM_PROVIDERS = {"deepseek", "kimi", "minimax"}
VISION_LLM_PROVIDERS = {"kimi", "minimax"}


def session_dir(session_id: str) -> Path:
    return SESSION_STORE.session_dir(session_id)


def load_session(session_id: str) -> Dict[str, Any]:
    try:
        return SESSION_STORE.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


def update_session(session_id: str, **updates: Any) -> None:
    SESSION_STORE.update(session_id, **updates)


def append_message(session_id: str, role: str, content: str, **extra: Any) -> None:
    SESSION_STORE.append_message(session_id, role, content, **extra)


def normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def provider_api_key_name(provider: str) -> str:
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    if provider == "kimi":
        return "MOONSHOT_API_KEY"
    if provider == "minimax":
        return "MINIMAX_API_KEY"
    return ""


def validate_provider(provider: str, allowed: set[str], label: str) -> str:
    provider = normalize_provider(provider)
    if provider not in allowed:
        allowed_text = "、".join(sorted(allowed))
        raise HTTPException(status_code=400, detail=f"{label} 只支持：{allowed_text}")

    key_name = provider_api_key_name(provider)
    if not key_name or not os.getenv(key_name):
        raise HTTPException(status_code=400, detail=f"已选择 {provider} 作为{label}，但没有配置 {key_name}")
    return provider


def run_session(session_id: str) -> None:
    session = load_session(session_id)
    try:
        run_esg_graph(session, update_session, append_message)
    except Exception:
        pass


def regenerate_section_task(session_id: str, section_id: str) -> None:
    session = load_session(session_id)
    try:
        run_regenerate_section(session, section_id, update_session, append_message)
    except Exception:
        pass


def check_section_task(session_id: str, section_id: str, check_mode: str) -> None:
    session = load_session(session_id)
    try:
        run_check_section(session, section_id, check_mode, update_session, append_message)
    except Exception:
        pass


@app.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    load_session(session_id)

    update_session(
        session_id,
        status="queued",
        error_message=None,
        traceback=None,
        progress_message="已加入继续生成队列",
    )
    append_message(session_id, role="user", content="继续生成", action="resume")
    background_tasks.add_task(run_session, session_id)
    return load_session(session_id)


@app.post("/sessions/{session_id}/sections/{section_id}/regenerate")
async def regenerate_section(
    session_id: str,
    section_id: str,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    load_session(session_id)
    update_session(
        session_id,
        status="queued",
        error_message=None,
        traceback=None,
        progress_message="已加入章节重新生成队列",
    )
    append_message(session_id, role="user", content=f"重新生成章节：{section_id}", action="regenerate")
    background_tasks.add_task(regenerate_section_task, session_id, section_id)
    return load_session(session_id)


@app.post("/sessions/{session_id}/sections/{section_id}/check")
async def check_section(
    session_id: str,
    section_id: str,
    background_tasks: BackgroundTasks,
    check_mode: str = Form("text"),
) -> Dict[str, Any]:
    if check_mode not in {"text", "vision"}:
        raise HTTPException(status_code=400, detail="check_mode must be text or vision")
    load_session(session_id)
    update_session(
        session_id,
        status="queued",
        error_message=None,
        traceback=None,
        progress_message=f"已加入章节{'视觉' if check_mode == 'vision' else '普通'}检查队列",
    )
    append_message(
        session_id,
        role="user",
        content=f"{'视觉' if check_mode == 'vision' else '普通'}检查章节：{section_id}",
        action="check",
        check_mode=check_mode,
    )
    background_tasks.add_task(check_section_task, session_id, section_id, check_mode)
    return load_session(session_id)


@app.post("/sessions/{session_id}/chat")
async def chat_session(
    session_id: str,
    question: str = Form(...),
) -> Dict[str, Any]:
    session = load_session(session_id)
    if session.get("status") == "building_vector_store":
        raise HTTPException(status_code=409, detail="检索库正在准备中，请稍后再问")
    sdir = session_dir(session_id)
    chunks_path = sdir / "chunks_clean_for_rag.pkl"
    if not chunks_path.exists():
        raise HTTPException(status_code=409, detail="检索库尚未准备好，请等待 PDF 解析和向量库构建完成")

    chunks = load_pickle(chunks_path)
    chroma_dir = sdir / "chroma"
    vs = load_or_build_vector_store(chunks, chroma_dir)
    try:
        result = chat_with_report(
            question=question,
            chunks=chunks,
            vs=vs,
            llm_config=session.get("llm_config"),
            top_k=12,
            keyword_k=10,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc

    SESSION_STORE.mutate(
        session_id,
        lambda current: current.setdefault("chat_history", []).append({
            "question": result["question"],
            "answer": result["answer"],
            "source_chunk_ids": result["source_chunk_ids"],
            "evidence_pages": result["evidence_pages"],
            "created_at": now_text(),
        }),
    )
    return result


@app.post("/sessions")
async def create_session(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    excel_path: str = Form("ESG披露框架.xlsx"),
    parse_mode: str = Form("fast"),
    selector_provider: str = Form("deepseek"),
    writer_provider: str = Form("kimi"),
    figure_provider: str = Form("kimi"),
    checker_provider: str = Form("deepseek"),
    vision_checker_provider: str = Form("kimi"),
) -> Dict[str, Any]:
    if parse_mode not in {"fast", "precise"}:
        raise HTTPException(status_code=400, detail="parse_mode must be fast or precise")

    llm_config = {
        "selector_provider": validate_provider(selector_provider, TEXT_LLM_PROVIDERS, "材料筛选 LLM"),
        "writer_provider": validate_provider(writer_provider, TEXT_LLM_PROVIDERS, "写作 LLM"),
        "figure_provider": validate_provider(figure_provider, VISION_LLM_PROVIDERS, "图片插入 LLM"),
        "checker_provider": validate_provider(checker_provider, TEXT_LLM_PROVIDERS, "普通检查 LLM"),
        "vision_checker_provider": validate_provider(vision_checker_provider, VISION_LLM_PROVIDERS, "视觉检查 LLM"),
    }

    session_id = uuid.uuid4().hex[:10]
    sdir = session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=True)

    pdf_path = sdir / "source.pdf"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    excel = Path(excel_path)
    if not excel.is_absolute():
        excel = AGENT_DIR / excel_path
    if not excel.exists():
        raise HTTPException(status_code=400, detail=f"Excel file not found: {excel}")

    SESSION_STORE.create(session_id, {
        "session_id": session_id,
        "status": "uploaded",
        "progress_message": "文件已上传，等待后台处理",
        "created_at": now_text(),
        "updated_at": now_text(),
        "pdf_path": str(pdf_path),
        "pdf_filename": pdf.filename,
        "excel_path": str(excel),
        "parse_mode": parse_mode,
        "llm_config": llm_config,
        "output_dir": str(sdir),
        "sections": [],
        "completed_sections": [],
        "current_section_id": None,
        "current_section_title": None,
        "chunk_count": 0,
    })
    append_message(session_id, role="user", content=f"上传文件：{pdf.filename}", action="upload")

    background_tasks.add_task(run_session, session_id)
    return load_session(session_id)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    return load_session(session_id)


@app.get("/sessions/{session_id}/sections/{section_id}")
async def get_section(session_id: str, section_id: str) -> Dict[str, Any]:
    session = load_session(session_id)
    for section in session.get("sections", []):
        if section["section_id"] == section_id:
            return section
    raise HTTPException(status_code=404, detail="Section not found")


@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str) -> Any:
    load_session(session_id)
    return SESSION_STORE.get_messages(session_id)


@app.get("/sessions/{session_id}/pdf")
async def get_pdf(session_id: str) -> FileResponse:
    session = load_session(session_id)
    return FileResponse(session["pdf_path"], media_type="application/pdf")


@app.get("/sessions/{session_id}/pdf_page/{page_no}.png")
async def get_pdf_page(session_id: str, page_no: int, scale: float = 2.4) -> FileResponse:
    session = load_session(session_id)
    pdf_path = Path(session["pdf_path"])
    pages_dir = session_dir(session_id) / "pdf_pages"
    pages_dir.mkdir(exist_ok=True)
    scale = max(1.0, min(float(scale), 4.0))
    scale_key = str(scale).replace(".", "_")
    image_path = pages_dir / f"page_{page_no}_scale_{scale_key}.png"

    if not image_path.exists():
        pdf = fitz.open(str(pdf_path))
        try:
            if page_no < 1 or page_no > len(pdf):
                raise HTTPException(status_code=404, detail="Page not found")
            page = pdf[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(str(image_path))
        finally:
            pdf.close()

    return FileResponse(str(image_path), media_type="image/png")


@app.get("/sessions/{session_id}/download")
async def download_report(session_id: str) -> FileResponse:
    session = load_session(session_id)
    path = Path(session.get("full_report_path") or session_dir(session_id) / "full_report.md")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet")
    return FileResponse(str(path), filename=f"esg_report_{session_id}.md", media_type="text/markdown")


@app.get("/sessions/{session_id}/download_docx")
async def download_report_docx(session_id: str) -> FileResponse:
    session = load_session(session_id)
    md_path = Path(session.get("full_report_path") or session_dir(session_id) / "full_report.md")
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet")

    docx_path = session_dir(session_id) / f"esg_report_{session_id}.docx"
    sanitized_path = session_dir(session_id) / f"esg_report_{session_id}.pandoc.md"
    try:
        convert_markdown_to_word(
            md_path,
            docx_path,
            sanitized_markdown_file=sanitized_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Word 导出失败：{exc}") from exc

    SESSION_STORE.update(
        session_id,
        word_report_path=str(docx_path),
        sanitized_report_path=str(sanitized_path),
    )
    return FileResponse(
        str(docx_path),
        filename=f"esg_report_{session_id}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
