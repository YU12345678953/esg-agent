import json
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

try:
    from .markdown_sanitizer import sanitize_markdown_for_pypandoc
    from .generation import (
        SECTION_GROUPS,
        assign_chunks_to_sections,
        build_candidate_context,
        build_chunk_catalogue,
        build_section_requirements,
        build_selected_context,
        check_generated_section,
        extract_image_items_from_final_chunks,
        figure_section,
        format_all_section_overview,
        format_section_requirement_text,
        get_figure_llm,
        get_selector_llm,
        get_writer_llm,
        load_or_build_vector_store,
        load_requirements_from_excel,
        rag_search_ids,
        rerank_material_chunks_for_section,
        revise_whole_report,
        unique_evidence_pages,
        write_section,
    )
    from .pipeline import load_pickle, run_preprocessing
except ImportError:
    from markdown_sanitizer import sanitize_markdown_for_pypandoc
    from generation import (
        SECTION_GROUPS,
        assign_chunks_to_sections,
        build_candidate_context,
        build_chunk_catalogue,
        build_section_requirements,
        build_selected_context,
        check_generated_section,
        extract_image_items_from_final_chunks,
        figure_section,
        format_all_section_overview,
        format_section_requirement_text,
        get_figure_llm,
        get_selector_llm,
        get_writer_llm,
        load_or_build_vector_store,
        load_requirements_from_excel,
        rag_search_ids,
        rerank_material_chunks_for_section,
        revise_whole_report,
        unique_evidence_pages,
        write_section,
    )
    from pipeline import load_pickle, run_preprocessing


class ESGGraphState(TypedDict, total=False):
    session_id: str
    pdf_path: str
    excel_path: str
    parse_mode: str
    llm_config: Dict[str, str]
    output_dir: str
    chunks_path: str
    chroma_dir: str
    chunk_count: int
    requirements: List[Dict[str, str]]
    section_chunk_map: Dict[str, List[int]]
    section_chunk_votes: Dict[str, Dict[str, int]]
    sections: List[Dict[str, Any]]
    current_section_work: Dict[str, Any]
    timings: List[Dict[str, Any]]
    current_section_index: int
    full_report_path: str


def graph_state_path(output_dir: Path) -> Path:
    return output_dir / "graph_state.json"


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoints.sqlite"


def save_graph_state(state: ESGGraphState) -> None:
    path = graph_state_path(Path(state["output_dir"]))
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph_state(output_dir: Path) -> ESGGraphState | None:
    path = graph_state_path(output_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def initial_graph_state(session: Dict[str, Any]) -> ESGGraphState:
    output_dir = Path(session["output_dir"])
    saved = load_graph_state(output_dir)
    if saved:
        saved["session_id"] = session["session_id"]
        saved["pdf_path"] = session["pdf_path"]
        saved["excel_path"] = session["excel_path"]
        saved["parse_mode"] = session["parse_mode"]
        saved["llm_config"] = session["llm_config"]
        saved["output_dir"] = session["output_dir"]
        return saved

    sections = session.get("sections") or []
    current_section_index = len([
        section for section in sections
        if section.get("status") == "generated"
    ])
    return {
        "session_id": session["session_id"],
        "pdf_path": session["pdf_path"],
        "excel_path": session["excel_path"],
        "parse_mode": session["parse_mode"],
        "llm_config": session["llm_config"],
        "output_dir": session["output_dir"],
        "chunks_path": str(output_dir / "chunks_clean_for_rag.pkl"),
        "chroma_dir": str(output_dir / "chroma"),
        "chunk_count": session.get("chunk_count", 0),
        "section_chunk_map": {},
        "section_chunk_votes": {},
        "sections": sections,
        "timings": session.get("timings", []),
        "current_section_index": current_section_index,
        "full_report_path": str(output_dir / "full_report.md"),
    }


def graph_state_to_session_projection(
    state: ESGGraphState,
    **updates: Any,
) -> Dict[str, Any]:
    """Build the UI-facing session snapshot from the workflow state."""
    sections = state.get("sections", [])
    projection = {
        "sections": sections,
        "completed_sections": [
            section["section_id"]
            for section in sections
            if section.get("status") == "generated"
        ],
        "chunk_count": state.get("chunk_count", 0),
        "timings": state.get("timings", []),
        "full_report_path": state.get("full_report_path"),
    }
    projection.update(updates)
    return projection


def project_graph_state_to_session(
    state: ESGGraphState,
    update_session: Callable[..., None],
    **updates: Any,
) -> None:
    """Project LangGraph's internal state into session.json for the UI."""
    update_session(
        state["session_id"],
        **graph_state_to_session_projection(state, **updates),
    )


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_section_version(
    section_result: Dict[str, Any],
    version_number: int,
    action: str,
) -> Dict[str, Any]:
    return {
        "version": version_number,
        "action": action,
        "created_at": now_text(),
        "content": section_result.get("content", ""),
        "selected_chunk_ids": section_result.get("selected_chunk_ids", []),
        "evidence_pages": section_result.get("evidence_pages", []),
        "title_ids": section_result.get("title_ids", []),
        "rag_ids": section_result.get("rag_ids", []),
        "candidate_ids": section_result.get("candidate_ids", []),
    }


def ensure_section_versions(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    versions = section.get("versions")
    if isinstance(versions, list) and versions:
        return versions

    if section.get("content"):
        return [build_section_version(section, 1, section.get("version_action", "generate"))]
    return []


def upsert_section_result(
    sections: List[Dict[str, Any]],
    section_result: Dict[str, Any],
    action: str = "generate",
) -> List[Dict[str, Any]]:
    by_id = {
        section.get("section_id"): dict(section)
        for section in sections
    }
    existing = by_id.get(section_result["section_id"])
    versions = ensure_section_versions(existing) if existing else []
    next_version = len(versions) + 1
    versions.append(build_section_version(section_result, next_version, action))

    merged = dict(existing or {})
    merged.update(section_result)
    merged["versions"] = versions
    merged["active_version"] = next_version
    merged["version_action"] = action
    merged["versioned_at"] = now_text()
    by_id[section_result["section_id"]] = merged

    ordered = []
    for section_group in SECTION_GROUPS:
        section = by_id.get(section_group["section_id"])
        if section:
            ordered.append(section)
    return ordered


def append_timing(
    state: ESGGraphState,
    task: str,
    seconds: float,
    **extra: Any,
) -> None:
    timings = state.get("timings") or []
    timings.append({
        "task": task,
        "seconds": round(seconds, 2),
        "ended_at": now_text(),
        **extra,
    })
    state["timings"] = timings


def build_esg_graph(
    update_session: Callable[..., None],
    append_message: Callable[..., None],
    checkpointer: SqliteSaver | None = None,
    entry_point: str = "preprocess",
):
    graph = StateGraph(ESGGraphState)

    def preprocess_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        output_dir = Path(state["output_dir"])
        chunks_path = output_dir / "chunks_clean_for_rag.pkl"
        state["chunks_path"] = str(chunks_path)
        
    #断点续点方式
        if chunks_path.exists():
            chunks = load_pickle(chunks_path)
            state["chunk_count"] = len(chunks)
            append_timing(
                state,
                "preprocess",
                time.perf_counter() - started,
                detail="skip_existing_chunks",
            )
            project_graph_state_to_session(
                state,
                update_session,
                status="building_vector_store",
                progress_message=f"检测到已有 {len(chunks)} 个 chunk，跳过 PDF 解析",
            )
            save_graph_state(state)
            return state

        project_graph_state_to_session(
            state,
            update_session,
            status="parsing_pdf",
            progress_message="正在解析 PDF 并保存图片",
        )

        def status_callback(message: str) -> None:
            update_session(state["session_id"], progress_message=message)

        chunks = run_preprocessing(
            Path(state["pdf_path"]),
            output_dir,
            status_callback=status_callback,
            parse_mode=state["parse_mode"],
        )
        state["chunk_count"] = len(chunks)
        append_timing(state, "preprocess", time.perf_counter() - started)
        save_graph_state(state)
        return state

    def vector_store_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        chunks = load_pickle(Path(state["chunks_path"]))
        state["chunk_count"] = len(chunks)
        state["chroma_dir"] = str(Path(state["output_dir"]) / "chroma")
        project_graph_state_to_session(
            state,
            update_session,
            status="building_vector_store",
            progress_message=f"已准备 {len(chunks)} 个 chunk，正在准备检索库",
        )
        load_or_build_vector_store(chunks, Path(state["chroma_dir"]))
        append_timing(state, "build_vector_store", time.perf_counter() - started)
        save_graph_state(state)
        return state

    def requirements_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        requirements = load_requirements_from_excel(Path(state["excel_path"]))
        state["requirements"] = requirements
        append_timing(state, "load_requirements", time.perf_counter() - started)
        save_graph_state(state)
        return state

    def assign_chunks_to_sections_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        update_session(
            state["session_id"],
            progress_message="正在用完整披露框架为各章节分配候选 chunk",
        )
        chunks = load_pickle(Path(state["chunks_path"]))
        req_map = {
            req["指引条目"].strip(): req
            for req in state.get("requirements", [])
        }
        selector_llm = get_selector_llm(state["llm_config"])
        assignment_result = assign_chunks_to_sections(
            llm=selector_llm,
            chunks=chunks,
            section_groups=SECTION_GROUPS,
            req_map=req_map,
            chunk_catalogue=build_chunk_catalogue(chunks),
            runs=1,
            min_votes=1,
            max_chunks_per_section=10,
            max_sections_per_chunk=3,
        )
        state["section_chunk_map"] = assignment_result["section_chunk_map"]
        state["section_chunk_votes"] = assignment_result["section_chunk_votes"]
        (Path(state["output_dir"]) / "section_chunk_map.json").write_text(
            json.dumps(assignment_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        append_timing(
            state,
            "assign_chunks_to_sections",
            time.perf_counter() - started,
            runs=1,
            max_chunks_per_section=10,
            max_sections_per_chunk=3,
        )
        save_graph_state(state)
        return state

    def start_section_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        if state.get("run_mode") == "regenerate_section":
            target_section_id = state.get("target_section_id")
            current_index = next(
                (
                    index for index, group in enumerate(SECTION_GROUPS)
                    if group["section_id"] == target_section_id
                ),
                int(state.get("current_section_index", 0)),
            )
            state["current_section_index"] = current_index
        else:
            current_index = int(state.get("current_section_index", 0))
        sections = state.get("sections") or []

        if current_index >= len(SECTION_GROUPS):
            return state

        section_group = SECTION_GROUPS[current_index]
        state["current_section_work"] = {}
        existing_by_id = {
            section["section_id"]: section
            for section in sections
            if section.get("status") == "generated"
        }
        if (
            state.get("run_mode") != "regenerate_section"
            and section_group["section_id"] in existing_by_id
        ):
            state["current_section_index"] = current_index + 1
            append_timing(
                state,
                "start_section",
                time.perf_counter() - started,
                section_id=section_group["section_id"],
                section_title=section_group["title"],
                detail="skip_existing_section",
            )
            save_graph_state(state)
            return state

        project_graph_state_to_session(
            state,
            update_session,
            status="generating",
            current_section_id=section_group["section_id"],
            current_section_title=section_group["title"],
            progress_message=(
                f"正在重新生成 {section_group['title']}"
                if state.get("run_mode") == "regenerate_section"
                else f"正在生成 {section_group['title']} ({current_index + 1}/{len(SECTION_GROUPS)})"
            ),
        )

        req_map = {
            req["指引条目"].strip(): req
            for req in state.get("requirements", [])
        }
        section_reqs = build_section_requirements(section_group, req_map)
        if not section_reqs:
            state["sections"] = replace_section_result(sections, {
                "section_id": section_group["section_id"],
                "title": section_group["title"],
                "status": "skipped",
                "content": "",
                "selected_chunk_ids": [],
                "evidence_pages": [],
            })
            state["current_section_index"] = current_index + 1
            save_graph_state(state)
            return state

        section_requirement_text = format_section_requirement_text(section_group, section_reqs)
        state["current_section_work"] = {
            "started_perf": started,
            "section_index": current_index,
            "section_group": section_group,
            "section_reqs": section_reqs,
            "section_requirement_text": section_requirement_text,
            "all_section_overview": format_all_section_overview(SECTION_GROUPS, section_group),
            "title_ids": (state.get("section_chunk_map") or {}).get(section_group["section_id"], []),
        }
        save_graph_state(state)
        return state

    def rag_search_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        update_session(
            state["session_id"],
            progress_message=f"正在 RAG 检索 {work['section_group']['title']} 的候选证据",
        )
        chunks = load_pickle(Path(state["chunks_path"]))
        state["chroma_dir"] = str(Path(state["output_dir"]) / "chroma")
        vs = load_or_build_vector_store(chunks, Path(state["chroma_dir"]))
        work["rag_ids"] = rag_search_ids(vs, work["section_requirement_text"], k=20)
        state["current_section_work"] = work
        append_timing(
            state,
            "rag_search",
            time.perf_counter() - started,
            section_id=work["section_group"]["section_id"],
            section_title=work["section_group"]["title"],
        )
        save_graph_state(state)
        return state

    def build_candidate_context_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        update_session(
            state["session_id"],
            progress_message=f"正在整理 {work['section_group']['title']} 的候选证据上下文",
        )
        chunks = load_pickle(Path(state["chunks_path"]))
        candidate_ids = []
        for chunk_id in work.get("title_ids", []) + work.get("rag_ids", []):
            if 0 <= chunk_id < len(chunks) and chunk_id not in candidate_ids:
                candidate_ids.append(chunk_id)
        work["candidate_ids"] = candidate_ids
        work["candidate_context"] = build_candidate_context(chunks, candidate_ids)
        state["current_section_work"] = work
        append_timing(
            state,
            "build_candidate_context",
            time.perf_counter() - started,
            section_id=work["section_group"]["section_id"],
            section_title=work["section_group"]["title"],
        )
        save_graph_state(state)
        return state

    def rerank_chunks_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        update_session(
            state["session_id"],
            progress_message=f"正在 rerank {work['section_group']['title']} 的证据 chunk",
        )
        chunks = load_pickle(Path(state["chunks_path"]))
        selector_llm = get_selector_llm(state["llm_config"])
        candidate_ids = work.get("candidate_ids", [])
        if candidate_ids:
            final_ids = rerank_material_chunks_for_section(
                selector_llm,
                chunks,
                work["section_group"],
                work["section_requirement_text"],
                candidate_ids,
                work["all_section_overview"],
                max_k=12,
            )
            if not final_ids:
                final_ids = candidate_ids[:12]
        else:
            final_ids = []
        work["selected_chunk_ids"] = final_ids
        work["selected_context"] = build_selected_context(chunks, final_ids)
        state["current_section_work"] = work
        append_timing(
            state,
            "rerank_chunks",
            time.perf_counter() - started,
            section_id=work["section_group"]["section_id"],
            section_title=work["section_group"]["title"],
        )
        save_graph_state(state)
        return state

    def write_section_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        update_session(
            state["session_id"],
            progress_message=f"正在写作 {work['section_group']['title']} 正文",
        )
        writer_llm = get_writer_llm(state["llm_config"])
        work["draft_content"] = write_section(
            writer_llm,
            work["section_group"],
            work["section_reqs"],
            work["section_requirement_text"],
            work.get("selected_context", ""),
        )
        state["current_section_work"] = work
        append_timing(
            state,
            "write_section",
            time.perf_counter() - started,
            section_id=work["section_group"]["section_id"],
            section_title=work["section_group"]["title"],
        )
        save_graph_state(state)
        return state

    def insert_figures_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        update_session(
            state["session_id"],
            progress_message=f"正在处理 {work['section_group']['title']} 的图片插入",
        )
        chunks = load_pickle(Path(state["chunks_path"]))
        figure_llm = get_figure_llm(state["llm_config"])
        image_items = extract_image_items_from_final_chunks(
            chunks,
            work.get("selected_chunk_ids", []),
        )
        work["content"] = figure_section(
            figure_llm,
            work.get("draft_content", ""),
            work["section_requirement_text"],
            image_items,
        )
        state["current_section_work"] = work
        append_timing(
            state,
            "insert_figures",
            time.perf_counter() - started,
            section_id=work["section_group"]["section_id"],
            section_title=work["section_group"]["title"],
        )
        save_graph_state(state)
        return state

    def sanitize_markdown_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        update_session(
            state["session_id"],
            progress_message=f"正在规范化 {work['section_group']['title']} 的 Markdown",
        )
        work["content"] = sanitize_markdown_for_pypandoc(work.get("content", ""))
        state["current_section_work"] = work
        append_timing(
            state,
            "sanitize_markdown",
            time.perf_counter() - started,
            section_id=work["section_group"]["section_id"],
            section_title=work["section_group"]["title"],
        )
        save_graph_state(state)
        return state

    def save_section_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        work = state["current_section_work"]
        action = "regenerate" if state.get("run_mode") == "regenerate_section" else "generate"
        update_session(
            state["session_id"],
            progress_message=f"正在保存 {work['section_group']['title']} 章节",
        )
        section_group = work["section_group"]
        chunks = load_pickle(Path(state["chunks_path"]))
        final_ids = work.get("selected_chunk_ids", [])
        content = work.get("content", "")
        section_result = {
            "section_id": section_group["section_id"],
            "title": section_group["title"],
            "description": section_group["description"],
            "status": "generated",
            "content": content,
            "title_ids": work.get("title_ids", []),
            "rag_ids": work.get("rag_ids", []),
            "candidate_ids": work.get("candidate_ids", []),
            "selected_chunk_ids": final_ids,
            "evidence_pages": unique_evidence_pages(chunks, final_ids),
        }
        section_path = Path(state["output_dir"]) / f"{section_group['section_id']}.md"
        section_path.write_text(content, encoding="utf-8")

        sections = state.get("sections") or []
        state["sections"] = upsert_section_result(sections, section_result, action=action)
        if state.get("run_mode") == "regenerate_section":
            state["current_section_index"] = int(state.get("resume_section_index", len(state.get("sections", []))))
        else:
            state["current_section_index"] = int(work.get("section_index", 0)) + 1

        if section_result.get("content"):
            append_message(
                state["session_id"],
                role="assistant",
                content=section_result["content"],
                section_id=section_result["section_id"],
                action=action,
            )

        append_timing(
            state,
            "save_section",
            time.perf_counter() - started,
            section_id=section_group["section_id"],
            section_title=section_group["title"],
        )
        append_timing(
            state,
            "generate_section_total",
            time.perf_counter() - float(work.get("started_perf", started)),
            section_id=section_group["section_id"],
            section_title=section_group["title"],
        )
        project_graph_state_to_session(
            state,
            update_session,
            status="generating",
            current_section_id=section_group["section_id"],
            current_section_title=section_group["title"],
            progress_message=(
                f"{section_group['title']} 重新生成完成"
                if action == "regenerate"
                else f"{section_group['title']} 生成完成"
            ),
        )
        state["current_section_work"] = {}
        save_graph_state(state)
        return state

    def finish_regenerate_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        rebuild_full_report(state)
        append_timing(state, "finish_regenerate", time.perf_counter() - started)
        project_graph_state_to_session(
            state,
            update_session,
            status="completed" if len(state.get("sections", [])) >= len(SECTION_GROUPS) else "generating",
            current_section_id=None,
            current_section_title=None,
            progress_message="章节重新生成完成，已保留历史版本",
        )
        state.pop("run_mode", None)
        state.pop("target_section_id", None)
        state.pop("resume_section_index", None)
        save_graph_state(state)
        return state

    def finalize_node(state: ESGGraphState) -> ESGGraphState:
        started = time.perf_counter()
        sections = state.get("sections", [])
        full_report = "\n\n---\n\n".join(
            section.get("content", "")
            for section in sections
            if section.get("content")
        )
        output_dir = Path(state["output_dir"])
        draft_report_path = output_dir / "full_report_draft.md"
        draft_report_path.write_text(full_report, encoding="utf-8")

        update_session(
            state["session_id"],
            progress_message="正在进行全文结构审校、图片去重和最终修订",
        )
        revise_started = time.perf_counter()
        requirements = state.get("requirements") or load_requirements_from_excel(Path(state["excel_path"]))
        state["requirements"] = requirements
        req_map = {
            req["指引条目"].strip(): req
            for req in requirements
        }
        writer_llm = get_writer_llm(state["llm_config"])
        chunks = load_pickle(Path(state["chunks_path"]))
        full_report = revise_whole_report(
            writer_llm,
            SECTION_GROUPS,
            req_map,
            full_report,
            sections=sections,
            chunks=chunks,
        )
        full_report = sanitize_markdown_for_pypandoc(full_report)
        append_timing(state, "revise_whole_report", time.perf_counter() - revise_started)

        full_report_path = Path(state["output_dir"]) / "full_report.md"
        full_report_path.write_text(full_report, encoding="utf-8")
        state["full_report_path"] = str(full_report_path)
        append_timing(state, "finalize", time.perf_counter() - started)
        project_graph_state_to_session(
            state,
            update_session,
            status="completed",
            current_section_id=None,
            current_section_title=None,
            progress_message="全部章节生成完成",
        )
        save_graph_state(state)
        return state

    def should_continue_sections(state: ESGGraphState) -> str:
        if state.get("run_mode") == "regenerate_section":
            return "finish_regenerate"
        if int(state.get("current_section_index", 0)) >= len(SECTION_GROUPS):
            return "finalize"
        return "start_section"

    def should_continue_after_start(state: ESGGraphState) -> str:
        work = state.get("current_section_work") or {}
        if state.get("run_mode") == "regenerate_section" and not work:
            return "finish_regenerate"
        if int(state.get("current_section_index", 0)) >= len(SECTION_GROUPS) and not work:
            return "finalize"
        if not work:
            return "start_section"
        return "rag_search"

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("build_vector_store", vector_store_node)
    graph.add_node("load_requirements", requirements_node)
    graph.add_node("assign_chunks_to_sections", assign_chunks_to_sections_node)
    graph.add_node("start_section", start_section_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("build_candidate_context", build_candidate_context_node)
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("write_section", write_section_node)
    graph.add_node("insert_figures", insert_figures_node)
    graph.add_node("sanitize_markdown", sanitize_markdown_node)
    graph.add_node("save_section", save_section_node)
    graph.add_node("finish_regenerate", finish_regenerate_node)
    graph.add_node("finalize", finalize_node)

    if entry_point not in {"preprocess", "start_section"}:
        raise ValueError(f"Unsupported graph entry_point: {entry_point}")
    graph.set_entry_point(entry_point)
    graph.add_edge("preprocess", "build_vector_store")
    graph.add_edge("build_vector_store", "load_requirements")
    graph.add_edge("load_requirements", "assign_chunks_to_sections")
    graph.add_conditional_edges(
        "assign_chunks_to_sections",
        should_continue_sections,
        {
            "start_section": "start_section",
            "finalize": "finalize",
            "finish_regenerate": "finish_regenerate",
        },
    )
    graph.add_conditional_edges(
        "start_section",
        should_continue_after_start,
        {
            "rag_search": "rag_search",
            "start_section": "start_section",
            "finalize": "finalize",
            "finish_regenerate": "finish_regenerate",
        },
    )
    graph.add_edge("rag_search", "build_candidate_context")
    graph.add_edge("build_candidate_context", "rerank_chunks")
    graph.add_edge("rerank_chunks", "write_section")
    graph.add_edge("write_section", "insert_figures")
    graph.add_edge("insert_figures", "sanitize_markdown")
    graph.add_edge("sanitize_markdown", "save_section")
    graph.add_conditional_edges(
        "save_section",
        should_continue_sections,
        {
            "start_section": "start_section",
            "finalize": "finalize",
            "finish_regenerate": "finish_regenerate",
        },
    )
    graph.add_edge("finish_regenerate", END)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def run_esg_graph(
    session: Dict[str, Any],
    update_session: Callable[..., None],
    append_message: Callable[..., None],
) -> None:
    output_dir = Path(session["output_dir"])
    conn = sqlite3.connect(
        checkpoint_path(output_dir),
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    config = {
        "recursion_limit": 80,
        "configurable": {
            "thread_id": session["session_id"],
        },
    }
    graph = build_esg_graph(update_session, append_message, checkpointer=checkpointer)
    snapshot = graph.get_state(config)
    has_checkpoint = bool(snapshot.values) or bool(snapshot.next)
    state = snapshot.values if has_checkpoint else initial_graph_state(session)
    try:
        graph.invoke(None if has_checkpoint else state, config=config)
    except Exception as exc:
        snapshot = graph.get_state(config)
        latest_state = snapshot.values or load_graph_state(output_dir) or state
        project_graph_state_to_session(
            latest_state,
            update_session,
            status="failed",
            error_message=str(exc),
            traceback=traceback.format_exc(),
            progress_message="生成失败",
        )
        raise
    finally:
        conn.close()


def rebuild_full_report(state: ESGGraphState) -> None:
    sections = state.get("sections", [])
    full_report = "\n\n---\n\n".join(
        section.get("content", "")
        for section in sections
        if section.get("content")
    )
    full_report_path = Path(state["output_dir"]) / "full_report.md"
    full_report_path.write_text(full_report, encoding="utf-8")
    state["full_report_path"] = str(full_report_path)


def replace_section_result(
    sections: List[Dict[str, Any]],
    section_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    by_id = {
        section.get("section_id"): section
        for section in sections
    }
    by_id[section_result["section_id"]] = section_result

    ordered = []
    for section_group in SECTION_GROUPS:
        section = by_id.get(section_group["section_id"])
        if section:
            ordered.append(section)
    return ordered


def replace_section_check_result(
    sections: List[Dict[str, Any]],
    section_id: str,
    check_result: str,
    check_mode: str,
) -> List[Dict[str, Any]]:
    updated = []
    for section in sections:
        if section.get("section_id") == section_id:
            section = dict(section)
            section["check_status"] = "checked"
            section["check_mode"] = check_mode
            section["check_result"] = check_result
            section["checked_at"] = now_text()
        updated.append(section)
    return updated


def run_regenerate_section(
    session: Dict[str, Any],
    section_id: str,
    update_session: Callable[..., None],
    append_message: Callable[..., None],
) -> None:
    output_dir = Path(session["output_dir"])
    conn = sqlite3.connect(
        checkpoint_path(output_dir),
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    config = {
        "recursion_limit": 80,
        "configurable": {
            "thread_id": f"{session['session_id']}:regenerate:{section_id}:{int(time.time() * 1000)}",
        },
    }
    graph = build_esg_graph(
        update_session,
        append_message,
        checkpointer=checkpointer,
        entry_point="start_section",
    )

    try:
        state = load_graph_state(output_dir) or initial_graph_state(session)
        state["session_id"] = session["session_id"]
        state["pdf_path"] = session["pdf_path"]
        state["excel_path"] = session["excel_path"]
        state["parse_mode"] = session["parse_mode"]
        state["llm_config"] = session["llm_config"]
        state["output_dir"] = session["output_dir"]
        state["chunks_path"] = str(output_dir / "chunks_clean_for_rag.pkl")
        state["chroma_dir"] = str(output_dir / "chroma")

        section_group = next(
            (group for group in SECTION_GROUPS if group["section_id"] == section_id),
            None,
        )
        if section_group is None:
            raise ValueError(f"Unknown section_id: {section_id}")

        section_index = next(
            index for index, group in enumerate(SECTION_GROUPS)
            if group["section_id"] == section_id
        )
        chunks = load_pickle(Path(state["chunks_path"]))
        state["chunk_count"] = len(chunks)
        state["requirements"] = state.get("requirements") or load_requirements_from_excel(Path(state["excel_path"]))

        section_map_path = output_dir / "section_chunk_map.json"
        if section_map_path.exists() and not state.get("section_chunk_map"):
            assignment_result = json.loads(section_map_path.read_text(encoding="utf-8"))
            state["section_chunk_map"] = assignment_result.get("section_chunk_map", {})
            state["section_chunk_votes"] = assignment_result.get("section_chunk_votes", {})

        state["run_mode"] = "regenerate_section"
        state["target_section_id"] = section_id
        state["resume_section_index"] = int(state.get("current_section_index", len(state.get("sections", []))))
        state["current_section_index"] = section_index
        state["current_section_work"] = {}

        project_graph_state_to_session(
            state,
            update_session,
            status="generating",
            current_section_id=section_group["section_id"],
            current_section_title=section_group["title"],
            progress_message=f"正在重新生成 {section_group['title']}",
        )

        save_graph_state(state)
        graph.invoke(state, config=config)
        latest_state = load_graph_state(output_dir) or graph.get_state(config).values or state
        graph.update_state(
            {
                "recursion_limit": 80,
                "configurable": {
                    "thread_id": session["session_id"],
                },
            },
            latest_state,
        )
    except Exception as exc:
        latest_state = graph.get_state(config).values or load_graph_state(output_dir) or initial_graph_state(session)
        project_graph_state_to_session(
            latest_state,
            update_session,
            status="failed",
            error_message=str(exc),
            traceback=traceback.format_exc(),
            progress_message="重新生成失败",
        )
        raise
    finally:
        conn.close()


def run_check_section(
    session: Dict[str, Any],
    section_id: str,
    check_mode: str,
    update_session: Callable[..., None],
    append_message: Callable[..., None],
) -> None:
    output_dir = Path(session["output_dir"])
    conn = sqlite3.connect(
        checkpoint_path(output_dir),
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    config = {
        "recursion_limit": 80,
        "configurable": {
            "thread_id": session["session_id"],
        },
    }
    graph = build_esg_graph(update_session, append_message, checkpointer=checkpointer)

    try:
        started = time.perf_counter()
        snapshot = graph.get_state(config)
        state = load_graph_state(output_dir) or snapshot.values or initial_graph_state(session)
        state["session_id"] = session["session_id"]
        state["pdf_path"] = session["pdf_path"]
        state["excel_path"] = session["excel_path"]
        state["parse_mode"] = session["parse_mode"]
        state["llm_config"] = session["llm_config"]
        state["output_dir"] = session["output_dir"]
        state["chunks_path"] = str(output_dir / "chunks_clean_for_rag.pkl")

        section_group = next(
            (group for group in SECTION_GROUPS if group["section_id"] == section_id),
            None,
        )
        if section_group is None:
            raise ValueError(f"Unknown section_id: {section_id}")

        sections = state.get("sections") or session.get("sections") or []
        section_result = next(
            (section for section in sections if section.get("section_id") == section_id),
            None,
        )
        if section_result is None or section_result.get("status") != "generated":
            raise ValueError(f"Section has not been generated: {section_id}")

        project_graph_state_to_session(
            state,
            update_session,
            status="checking",
            current_section_id=section_group["section_id"],
            current_section_title=section_group["title"],
            progress_message=f"正在{'视觉' if check_mode == 'vision' else '普通'}检查 {section_group['title']}",
        )

        chunks = load_pickle(Path(state["chunks_path"]))
        requirements = state.get("requirements") or load_requirements_from_excel(Path(state["excel_path"]))
        state["requirements"] = requirements
        req_map = {
            req["指引条目"].strip(): req
            for req in requirements
        }
        check_result = check_generated_section(
            chunks=chunks,
            req_map=req_map,
            section_group=section_group,
            section_result=section_result,
            check_mode=check_mode,
            pdf_path=Path(state["pdf_path"]),
            llm_config=state["llm_config"],
        )
        check_suffix = "vision_check" if check_mode == "vision" else "check"
        (output_dir / f"{section_group['section_id']}_{check_suffix}.md").write_text(
            check_result,
            encoding="utf-8",
        )
        state["sections"] = replace_section_check_result(
            sections,
            section_id,
            check_result,
            check_mode,
        )
        append_timing(
            state,
            "check_section",
            time.perf_counter() - started,
            section_id=section_group["section_id"],
            section_title=section_group["title"],
            check_mode=check_mode,
        )
        save_graph_state(state)
        graph.update_state(config, state)

        append_message(
            session["session_id"],
            role="assistant",
            content=check_result,
            section_id=section_group["section_id"],
            action="check",
            check_mode=check_mode,
        )
        project_graph_state_to_session(
            state,
            update_session,
            status="completed" if len(state.get("sections", [])) >= len(SECTION_GROUPS) else "generating",
            current_section_id=None,
            current_section_title=None,
            progress_message=f"{section_group['title']} {'视觉' if check_mode == 'vision' else '普通'}检查完成",
        )
    except Exception as exc:
        latest_state = graph.get_state(config).values or load_graph_state(output_dir) or initial_graph_state(session)
        project_graph_state_to_session(
            latest_state,
            update_session,
            status="failed",
            error_message=str(exc),
            traceback=traceback.format_exc(),
            progress_message="章节检查失败",
        )
        raise
    finally:
        conn.close()
