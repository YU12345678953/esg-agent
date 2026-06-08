import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

try:
    from .graph_runner import run_check_section, run_esg_graph, run_regenerate_section
    from .word_export import convert_markdown_to_word
except ImportError:
    from graph_runner import run_check_section, run_esg_graph, run_regenerate_section
    from word_export import convert_markdown_to_word

ROOT_DIR = Path(__file__).resolve().parent.parent
AGENT_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = AGENT_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT_DIR / ".env")

app = FastAPI(title="ESG Agent API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Dict[str, Any]] = {}
TEXT_LLM_PROVIDERS = {"deepseek", "kimi", "minimax"}
VISION_LLM_PROVIDERS = {"kimi", "minimax"}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def session_json_path(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def save_session(session_id: str) -> None:
    path = session_json_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sessions[session_id], f, ensure_ascii=False, indent=2)


def load_session(session_id: str) -> Dict[str, Any]:
    if session_id in sessions:
        return sessions[session_id]
    path = session_json_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    with open(path, "r", encoding="utf-8") as f:
        sessions[session_id] = json.load(f)
    return sessions[session_id]


def update_session(session_id: str, **updates: Any) -> None:
    session = load_session(session_id)
    session.update(updates)
    session["updated_at"] = now_text()
    save_session(session_id)


def append_message(session_id: str, role: str, content: str, **extra: Any) -> None:
    sdir = session_dir(session_id)
    path = sdir / "messages.json"
    if path.exists():
        messages = json.loads(path.read_text(encoding="utf-8"))
    else:
        messages = []
    messages.append({
        "role": role,
        "content": content,
        "created_at": now_text(),
        **extra,
    })
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


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


@app.post("/sessions")
async def create_session(
    background_tasks: BackgroundTasks,
    pdf: List[UploadFile] = File(...),
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

    pdf_files = pdf if isinstance(pdf, list) else [pdf]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="至少需要上传一个 PDF")
    pdf_dir = sdir / "source_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = []
    pdf_filenames = []
    for index, uploaded_pdf in enumerate(pdf_files, start=1):
        suffix = Path(uploaded_pdf.filename or "").suffix or ".pdf"
        safe_name = Path(uploaded_pdf.filename or f"source_{index}.pdf").name
        pdf_path = pdf_dir / f"{index:02d}_{safe_name}"
        if pdf_path.suffix.lower() != ".pdf":
            pdf_path = pdf_path.with_suffix(suffix)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(uploaded_pdf.file, f)
        pdf_paths.append(str(pdf_path))
        pdf_filenames.append(uploaded_pdf.filename)

    excel = Path(excel_path)
    if not excel.is_absolute():
        excel = ROOT_DIR / excel_path
    if not excel.exists():
        raise HTTPException(status_code=400, detail=f"Excel file not found: {excel}")

    sessions[session_id] = {
        "session_id": session_id,
        "status": "uploaded",
        "progress_message": "文件已上传，等待后台处理",
        "created_at": now_text(),
        "updated_at": now_text(),
        "pdf_path": pdf_paths[0],
        "pdf_paths": pdf_paths,
        "pdf_filename": pdf_filenames[0],
        "pdf_filenames": pdf_filenames,
        "excel_path": str(excel),
        "parse_mode": parse_mode,
        "llm_config": llm_config,
        "output_dir": str(sdir),
        "sections": [],
        "completed_sections": [],
        "current_section_id": None,
        "current_section_title": None,
        "chunk_count": 0,
    }
    save_session(session_id)
    append_message(session_id, role="user", content=f"上传文件：{'、'.join(pdf_filenames)}", action="upload")

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
    path = session_dir(session_id) / "messages.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def session_pdf_path(session: Dict[str, Any], pdf_index: int = 0) -> Path:
    pdf_paths = session.get("pdf_paths") or [session["pdf_path"]]
    if pdf_index < 0 or pdf_index >= len(pdf_paths):
        raise HTTPException(status_code=404, detail="PDF not found")
    return Path(pdf_paths[pdf_index])


@app.get("/sessions/{session_id}/pdf")
async def get_pdf(session_id: str, pdf_index: int = 0) -> FileResponse:
    session = load_session(session_id)
    return FileResponse(str(session_pdf_path(session, pdf_index)), media_type="application/pdf")


@app.get("/sessions/{session_id}/pdf_page/{page_no}.png")
async def get_pdf_page(session_id: str, page_no: int, scale: float = 2.4, pdf_index: int = 0) -> FileResponse:
    session = load_session(session_id)
    pdf_path = session_pdf_path(session, pdf_index)
    pages_dir = session_dir(session_id) / "pdf_pages"
    pages_dir.mkdir(exist_ok=True)
    scale = max(1.0, min(float(scale), 4.0))
    scale_key = str(scale).replace(".", "_")
    image_path = pages_dir / f"pdf_{pdf_index}_page_{page_no}_scale_{scale_key}.png"

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

    session["word_report_path"] = str(docx_path)
    session["sanitized_report_path"] = str(sanitized_path)
    session["updated_at"] = now_text()
    save_session(session_id)
    return FileResponse(
        str(docx_path),
        filename=f"esg_report_{session_id}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
