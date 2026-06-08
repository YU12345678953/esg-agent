import os
import base64
import re
import time
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.getenv("ESG_API_BASE", "http://localhost:8000")
HTTP = requests.Session()
HTTP.trust_env = False
PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "kimi": "Kimi",
    "minimax": "MiniMax",
}
PROVIDER_VALUES = {label: value for value, label in PROVIDER_LABELS.items()}

st.set_page_config(
    page_title="ESG Report Generator",
    layout="wide",
)

st.title("ESG 报告生成工作台")

st.markdown(
    """
    <style>
    .report-image {
        display: block;
        max-width: 100%;
        height: auto;
        margin: 12px 0;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "selected_section_id" not in st.session_state:
    st.session_state.selected_section_id = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

SCROLL_HEIGHT = 820


def image_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def local_image_to_data_url(raw_path: str) -> str | None:
    path = Path(raw_path.strip())
    if not path.exists() or not path.is_file():
        return None
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{image_mime(path)};base64,{data}"


def render_markdown_with_local_images(markdown_text: str) -> str:
    def replace_md_image(match):
        alt = match.group(1)
        raw_path = match.group(2)
        data_url = local_image_to_data_url(raw_path)
        if not data_url:
            return match.group(0)
        return f'<img src="{data_url}" alt="{alt}" class="report-image">'

    def replace_html_image(match):
        before = match.group(1)
        raw_path = match.group(2)
        after = match.group(3)
        data_url = local_image_to_data_url(raw_path)
        if not data_url:
            return match.group(0)
        return f'<img{before}src="{data_url}"{after} class="report-image">'

    markdown_text = re.sub(
        r"!\[([^\]]*)\]\((/[^)]+)\)",
        replace_md_image,
        markdown_text,
    )
    markdown_text = re.sub(
        r"<img([^>]*?)src=[\"'](/[^\"']+)[\"']([^>]*)>",
        replace_html_image,
        markdown_text,
        flags=re.IGNORECASE,
    )
    return markdown_text


def api_get(path):
    response = HTTP.get(f"{API_BASE}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def load_session():
    if not st.session_state.session_id:
        return None
    try:
        return api_get(f"/sessions/{st.session_state.session_id}")
    except Exception as exc:
        st.error(f"获取会话失败：{exc}")
        return None

# =============================
# 侧边栏
# ==============================

with st.sidebar:
    st.subheader("文件")
    uploaded_files = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    excel_path = st.text_input("披露框架 Excel", value="ESG披露框架.xlsx")
    parse_mode_label = st.selectbox(
        "解析模式",
        options=[
            "快速模式",
            "精准模式：MinerU 分页解析",
        ],
        index=0,
    )
    parse_mode = "precise" if parse_mode_label.startswith("精准模式") else "fast"

    with st.expander("LLM 设置", expanded=False):
        selector_provider_label = st.selectbox(
            "材料筛选 / rerank",
            options=["DeepSeek", "Kimi", "MiniMax"],
            index=0,
        )
        writer_provider_label = st.selectbox(
            "正文写作",
            options=["DeepSeek", "Kimi", "MiniMax"],
            index=1,
        )
        figure_provider_label = st.selectbox(
            "图片插入 / 图题处理",
            options=["Kimi", "MiniMax"],
            index=0,
        )
        checker_provider_label = st.selectbox(
            "普通检查",
            options=["DeepSeek", "Kimi", "MiniMax"],
            index=0,
        )
        vision_checker_provider_label = st.selectbox(
            "视觉检查",
            options=["Kimi", "MiniMax"],
            index=0,
        )

    if st.button("开始生成", type="primary", use_container_width=True, disabled=not uploaded_files):
        try:
            files = [
                (
                    "pdf",
                    (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    ),
                )
                for uploaded_file in uploaded_files
            ]
            data = {
                "excel_path": excel_path,
                "parse_mode": parse_mode,
                "selector_provider": PROVIDER_VALUES[selector_provider_label],
                "writer_provider": PROVIDER_VALUES[writer_provider_label],
                "figure_provider": PROVIDER_VALUES[figure_provider_label],
                "checker_provider": PROVIDER_VALUES[checker_provider_label],
                "vision_checker_provider": PROVIDER_VALUES[vision_checker_provider_label],
            }
            response = HTTP.post(f"{API_BASE}/sessions", files=files, data=data, timeout=60)
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(detail)
            session = response.json()
            st.session_state.session_id = session["session_id"]
            st.session_state.selected_section_id = None
            st.session_state.selected_page = None
            st.rerun()
        except Exception as exc:
            st.error(f"创建会话失败：{exc}")

    st.divider()
    st.subheader("会话")
    manual_session = st.text_input("Session ID", value=st.session_state.session_id or "")
    if st.button("加载会话", use_container_width=True) and manual_session:
        st.session_state.session_id = manual_session.strip()
        st.rerun()

    st.session_state.auto_refresh = st.checkbox("生成中自动刷新", value=st.session_state.auto_refresh)
    if st.button("刷新", use_container_width=True):
        st.rerun()


session = load_session()

if not session:
    st.info("请上传 PDF 并开始生成。")
    st.stop()

with st.sidebar.expander("运行耗时", expanded=False):
    timings = session.get("timings") or []
    if timings:
        st.dataframe(
            timings,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("暂无耗时记录")

with st.sidebar.expander("当前 LLM 配置", expanded=False):
    llm_config = session["llm_config"]
    st.write({
        "材料筛选": PROVIDER_LABELS.get(llm_config["selector_provider"], llm_config["selector_provider"]),
        "正文写作": PROVIDER_LABELS.get(llm_config["writer_provider"], llm_config["writer_provider"]),
        "图片插入/图题处理": PROVIDER_LABELS.get(llm_config["figure_provider"], llm_config["figure_provider"]),
        "普通检查": PROVIDER_LABELS.get(llm_config["checker_provider"], llm_config["checker_provider"]),
        "视觉检查": PROVIDER_LABELS.get(llm_config["vision_checker_provider"], llm_config["vision_checker_provider"]),
    })

status = session.get("status", "unknown")
sections = session.get("sections", [])
completed = session.get("completed_sections", [])

top_cols = st.columns([1, 1, 1, 1, 2])
top_cols[0].metric("状态", status)
top_cols[1].metric("已生成章节", len(completed))
top_cols[2].metric("Chunks", session.get("chunk_count", 0))
top_cols[3].metric("解析模式", "精准" if session["parse_mode"] == "precise" else "快速")
top_cols[4].write(session.get("progress_message", ""))

if status == "failed":
    st.error(session.get("error_message", "生成失败"))
    if st.button("从中断处继续", type="primary"):
        try:
            response = HTTP.post(f"{API_BASE}/sessions/{session['session_id']}/resume", timeout=30)
            response.raise_for_status()
            st.rerun()
        except Exception as exc:
            st.error(f"继续生成失败：{exc}")
    with st.expander("Traceback"):
        st.code(session.get("traceback", ""))

if sections and st.session_state.selected_section_id is None:
    st.session_state.selected_section_id = sections[-1]["section_id"]
    pages = sections[-1].get("evidence_pages") or []
    st.session_state.selected_page = pages[0] if pages else None

left, right = st.columns([1.05, 1.05], gap="medium")

with left:
    st.subheader("生成章节")

    if not sections:
        st.info("后台正在解析 PDF 或生成第一章。")
    else:
        section_labels = {
            section["section_id"]: f"{idx + 1}. {section['title']}"
            for idx, section in enumerate(sections)
        }
        selected_id = st.radio(
            "选择章节",
            options=list(section_labels.keys()),
            format_func=lambda sid: section_labels[sid],
            index=max(0, list(section_labels.keys()).index(st.session_state.selected_section_id))
            if st.session_state.selected_section_id in section_labels else len(section_labels) - 1,
            horizontal=True,
        )

        if selected_id != st.session_state.selected_section_id:
            st.session_state.selected_section_id = selected_id
            selected_section = next(s for s in sections if s["section_id"] == selected_id)
            pages = selected_section.get("evidence_pages") or []
            st.session_state.selected_page = pages[0] if pages else None
            st.rerun()

        selected_section = next(s for s in sections if s["section_id"] == st.session_state.selected_section_id)
        action_cols = st.columns(3)
        if action_cols[0].button("重新生成当前章节", use_container_width=True):
            try:
                response = HTTP.post(
                    f"{API_BASE}/sessions/{session['session_id']}/sections/{selected_section['section_id']}/regenerate",
                    timeout=30,
                )
                response.raise_for_status()
                st.rerun()
            except Exception as exc:
                st.error(f"重新生成失败：{exc}")
        if action_cols[1].button("普通检查", use_container_width=True):
            try:
                response = HTTP.post(
                    f"{API_BASE}/sessions/{session['session_id']}/sections/{selected_section['section_id']}/check",
                    data={"check_mode": "text"},
                    timeout=30,
                )
                response.raise_for_status()
                st.rerun()
            except Exception as exc:
                st.error(f"检查失败：{exc}")
        if action_cols[2].button("视觉检查", use_container_width=True):
            try:
                response = HTTP.post(
                    f"{API_BASE}/sessions/{session['session_id']}/sections/{selected_section['section_id']}/check",
                    data={"check_mode": "vision"},
                    timeout=30,
                )
                response.raise_for_status()
                st.rerun()
            except Exception as exc:
                st.error(f"视觉检查失败：{exc}")

        with st.container(height=SCROLL_HEIGHT):
            st.markdown(
                render_markdown_with_local_images(selected_section.get("content", "")),
                unsafe_allow_html=True,
            )

        with st.expander("章节检查结果", expanded=False):
            check_result = selected_section.get("check_result")
            if check_result:
                mode_label = "视觉检查" if selected_section.get("check_mode") == "vision" else "普通检查"
                st.caption(f"{mode_label} | {selected_section.get('checked_at', '')}")
                st.markdown(check_result)
            else:
                st.caption("尚未检查当前章节")

        if status == "completed":
            download_cols = st.columns(2)
            try:
                response = HTTP.get(f"{API_BASE}/sessions/{session['session_id']}/download", timeout=20)
                if response.status_code == 200:
                    download_cols[0].download_button(
                        "下载 Markdown",
                        data=response.content,
                        file_name=f"esg_report_{session['session_id']}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            except Exception:
                pass

            download_cols[1].link_button(
                "导出/下载 Word",
                f"{API_BASE}/sessions/{session['session_id']}/download_docx",
                use_container_width=True,
            )

with right:
    st.subheader("证据页")

    selected_section = None
    if st.session_state.selected_section_id:
        selected_section = next(
            (s for s in sections if s["section_id"] == st.session_state.selected_section_id),
            None,
        )

    if not selected_section:
        st.info("生成章节后会在这里显示证据页。")
    else:
        evidence_sources = selected_section.get("evidence_sources") or []
        evidence_pages = sorted(dict.fromkeys(selected_section.get("evidence_pages") or []))
        selected_ids = selected_section.get("selected_chunk_ids") or []

        st.caption(f"证据 chunk：{selected_ids}")

        if not evidence_sources and not evidence_pages:
            st.info("该章节暂无可定位证据页。")
        else:
            if evidence_sources:
                st.caption("证据页按来源 PDF 和页码排列")
                display_sources = sorted(
                    evidence_sources,
                    key=lambda item: (
                        int(item.get("source_pdf_index") or 0),
                        int(item.get("page") or 0),
                    ),
                )
            else:
                st.caption("证据页按页码从小到大排列")
                display_sources = [
                    {
                        "source_pdf_index": 0,
                        "source_pdf_name": session.get("pdf_filename", "PDF"),
                        "page": page_no,
                    }
                    for page_no in evidence_pages
                ]
            with st.container(height=SCROLL_HEIGHT):
                for source in display_sources:
                    page_no = int(source.get("page") or 0)
                    pdf_index = int(source.get("source_pdf_index") or 0)
                    pdf_name = source.get("source_pdf_name") or session.get("pdf_filename", "PDF")
                    image_url = (
                        f"{API_BASE}/sessions/{session['session_id']}/pdf_page/{page_no}.png"
                        f"?scale=2.8&pdf_index={pdf_index}"
                    )
                    st.image(image_url, caption=f"{pdf_name} 第 {page_no} 页", use_container_width=True)

            first_source = display_sources[0]
            first_page = int(first_source.get("page") or 1)
            first_pdf_index = int(first_source.get("source_pdf_index") or 0)
            pdf_url = f"{API_BASE}/sessions/{session['session_id']}/pdf?pdf_index={first_pdf_index}#page={first_page}"
            st.link_button("打开原 PDF", pdf_url, use_container_width=True)

if status in {"uploaded", "queued", "parsing_pdf", "building_vector_store", "generating", "checking"} and st.session_state.auto_refresh:
    time.sleep(4)
    st.rerun()
