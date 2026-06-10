import ast
import base64
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("EMBEDDING_HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

try:
    from .markdown_sanitizer import sanitize_markdown_for_pypandoc
except ImportError:
    from markdown_sanitizer import sanitize_markdown_for_pypandoc

PROMPT_DIR = Path(__file__).parent / "prompts"
TOP_K_TITLE = 15
TEXT_LLM_PROVIDERS = {"deepseek", "kimi", "minimax"}
VISION_LLM_PROVIDERS = {"kimi", "minimax"}


SECTION_GROUPS = [
    {
        "section_id": "climate_governance",
        "title": "气候治理",
        "description": "气候治理机构、环境管理，气候管理，监督情况",
        "items": ["第二十一条"],
        "type": "环境",
    },
    {
        "section_id": "climate_strategy_transition",
        "title": "气候战略、风险与转型计划",
        "description": "气候变化对战略和业务模式的影响、气候风险与机遇、适应性评估、情景分析、转型计划、资源投入及实施进展",
        "items": ["第二十二条", "第二十三条"],
        "type": "环境",
    },
    {
        "section_id": "ghg_emissions",
        "title": "温室气体排放",
        "description": "温室气体排放量、不同范围排放情况、核算标准与方法",
        "items": ["第二十四条", "第二十五条", "第二十六条"],
        "type": "环境",
    },
    {
        "section_id": "emission_reduction",
        "title": "减排",
        "description": "减排目标、减排措施、减排机制、碳减排, 减排进展",
        "items": ["第二十七条", "第二十八条"],
        "type": "环境",
    },
    {
        "section_id": "pollution_ecology",
        "title": "污染防治与生态系统保护",
        "description": "污染物、废弃物、生态系统和生物多样性、环境事件",
        "items": ["第二十九条", "第三十一条", "第三十二条", "第三十三条"],
        "type": "环境",
    },
    {
        "section_id": "resource_circularity",
        "title": "资源利用与循环经济",
        "description": "循环经济、能源使用、水资源使用",
        "items": ["第三十五条", "第三十六条", "第三十七条"],
        "type": "环境",
    },
    {
        "section_id": "Rural Revitalization and Social Contribution",
        "title": "乡村振兴与社会贡献",
        "description": "乡村振兴、脱贫、公益慈善、社会贡献",
        "items": ["第三十九条", "第四十条"],
        "type": "社会",
    },
    {
        "section_id": "innovation-driven",
        "title": "创新驱动与科技伦理",
        "description": "创新驱动、科技创新、科技伦理",
        "items": ["第四十二条", "第四十三条"],
        "type": "社会",
    },
    {
        "section_id": "enterprise",
        "title": "供应链安全与平等对待中小企业",
        "description": "供应商、供应链风险、中小企业供应商的账期、逾期情况",
        "items": ["第四十四条", "第四十五条"],
        "type": "社会",
    },
    {
        "section_id": "customer",
        "title": "客户",
        "description": "产品及服务安全与质量、服务质量管理、数据安全与客户隐私保护",
        "items": ["第四十六条", "第四十七条"],
        "type": "社会",
    },
    {
        "section_id": "employee",
        "title": "员工",
        "description": "员工聘用、职业健康安全、员工培训",
        "items": ["第四十九条"],
        "type": "社会",
    },
    {
        "section_id": "sustainable development",
        "title": "可持续发展相关治理机制",
        "description": "按照不同可持续发展议题及重要性建立的公司治理结构、措施",
        "items": ["第五十条"],
        "type": "治理",
    },
    {
        "section_id": "unfair competition",
        "title": "防范商业贿赂与不正当竞争",
        "description": "反商业贿赂及反贪污",
        "items": ["第五十二条"],
        "type": "治理",
    },
]


def render_prompt(template_name: str, **kwargs: Any) -> str:
    text = (PROMPT_DIR / template_name).read_text(encoding="utf-8")
    for key, value in kwargs.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def llm_provider_config(provider: str, purpose: str) -> Dict[str, Any]:
    provider = (provider or "").strip().lower()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("已选择 DeepSeek，但没有配置 DEEPSEEK_API_KEY")
        model = os.getenv("DEEPSEEK_LLM_MODEL")
        if purpose == "select":
            model = os.getenv("SELECT_LLM_MODEL", model or "deepseek-v4-pro")
        elif purpose == "check":
            model = os.getenv("CHECK_LLM_MODEL", os.getenv("SELECT_LLM_MODEL", model or "deepseek-v4-pro"))
        else:
            model = model or "deepseek-v4-pro"
        return {
            "model": model,
            "api_key": api_key,
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        }

    if provider == "kimi":
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("已选择 Kimi，但没有配置 MOONSHOT_API_KEY")
        model = os.getenv("KIMI_LLM_MODEL")
        if purpose in {"write", "vision_check"}:
            model = os.getenv("WRITE_LLM_MODEL", model or "kimi-k2.5")
        elif purpose == "check":
            model = os.getenv("CHECK_KIMI_LLM_MODEL", model or os.getenv("WRITE_LLM_MODEL", "kimi-k2.5"))
        else:
            model = os.getenv("SELECT_KIMI_LLM_MODEL", model or os.getenv("WRITE_LLM_MODEL", "kimi-k2.5"))
        return {
            "model": model,
            "api_key": api_key,
            "base_url": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        }

    if provider == "minimax":
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("已选择 MiniMax，但没有配置 MINIMAX_API_KEY")
        model = os.getenv("MINIMAX_LLM_MODEL")
        if purpose == "write":
            model = os.getenv("WRITE_MINIMAX_LLM_MODEL", model or "MiniMax-M3")
        elif purpose == "check":
            model = os.getenv("CHECK_MINIMAX_LLM_MODEL", model or "MiniMax-M3")
        elif purpose == "vision_check":
            model = os.getenv("CHECK_VISION_MINIMAX_LLM_MODEL", model or "MiniMax-M3")
        else:
            model = os.getenv("SELECT_MINIMAX_LLM_MODEL", model or "MiniMax-M3")
        return {
            "model": model,
            "api_key": api_key,
            "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
        }

    raise ValueError(f"未知 LLM provider: {provider}")


def make_llm(provider: str, purpose: str, temperature: float = 0) -> ChatOpenAI:
    config = llm_provider_config(provider, purpose)
    if provider == "kimi":
        temperature = 1
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=temperature,
    )


def provider_from_config(
    llm_config: Dict[str, str] | None,
    key: str,
    default_provider: str,
) -> str:
    return ((llm_config or {}).get(key) or default_provider).strip().lower()


def get_selector_llm(llm_config: Dict[str, str] | None = None) -> ChatOpenAI:
    provider = provider_from_config(llm_config, "selector_provider", "deepseek")
    if provider not in TEXT_LLM_PROVIDERS:
        raise ValueError(f"材料筛选 LLM 不支持 provider: {provider}")
    return make_llm(provider, "select", temperature=0)


def get_writer_llm(llm_config: Dict[str, str] | None = None) -> ChatOpenAI:
    provider = provider_from_config(llm_config, "writer_provider", "kimi")
    if provider not in TEXT_LLM_PROVIDERS:
        raise ValueError(f"写作 LLM 不支持 provider: {provider}")
    return make_llm(
        provider,
        "write",
        temperature=float(os.getenv("WRITE_LLM_TEMPERATURE", "1")),
    )


def get_figure_llm(llm_config: Dict[str, str] | None = None) -> ChatOpenAI:
    provider = provider_from_config(llm_config, "figure_provider", "kimi")
    if provider not in VISION_LLM_PROVIDERS:
        raise ValueError("图片插入 LLM 只支持 Kimi 或 MiniMax，不能选择 DeepSeek")
    return make_llm(provider, "figure", temperature=0)


def get_checker_llm(llm_config: Dict[str, str] | None = None) -> ChatOpenAI:
    provider = provider_from_config(llm_config, "checker_provider", "deepseek")
    if provider not in TEXT_LLM_PROVIDERS:
        raise ValueError(f"普通检查 LLM 不支持 provider: {provider}")
    return make_llm(provider, "check", temperature=0)


def get_chat_llm(llm_config: Dict[str, str] | None = None) -> ChatOpenAI:
    provider = provider_from_config(llm_config, "chat_provider", provider_from_config(llm_config, "selector_provider", "deepseek"))
    if provider not in TEXT_LLM_PROVIDERS:
        raise ValueError(f"问答 LLM 不支持 provider: {provider}")
    return make_llm(provider, "select", temperature=0)


def get_vision_checker_llm(llm_config: Dict[str, str] | None = None) -> ChatOpenAI:
    provider = provider_from_config(llm_config, "vision_checker_provider", "kimi")
    if provider not in VISION_LLM_PROVIDERS:
        raise ValueError("视觉检查 LLM 只支持 Kimi 或 MiniMax，不能选择 DeepSeek")
    return make_llm(provider, "vision_check", temperature=0)


#============embedding==============================
def resolve_embedding_model_name() -> str:
    explicit_path = os.getenv("EMBEDDING_MODEL_PATH")
    if explicit_path and Path(explicit_path).exists():
        return explicit_path

    bundled_path = Path(__file__).parent / "local_models" / "bge-m3"
    if bundled_path.exists():
        return str(bundled_path)

    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def is_local_embedding_model(model_name: str) -> bool:
    return Path(model_name).exists()


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = resolve_embedding_model_name()
    model_kwargs = {"device": os.getenv("EMBEDDING_DEVICE", "cpu")}
    local_model = is_local_embedding_model(model_name)

    if not local_model and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = os.getenv("EMBEDDING_HF_ENDPOINT", "https://hf-mirror.com")

    local_only = os.getenv("EMBEDDING_LOCAL_ONLY")
    if local_only is None:
        local_only = "1" if local_model else "0"

    if local_only == "1":
        model_kwargs["local_files_only"] = True

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
            },
            cache_folder=os.getenv("EMBEDDING_CACHE_FOLDER"),
            show_progress=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding 模型加载失败。请确认已安装 sentence-transformers，"
            "并且可以通过 HF_ENDPOINT 镜像下载 BAAI/bge-m3；"
            "或者把 bge-m3 放到 esg_agent/local_models/bge-m3，"
            "再设置 EMBEDDING_LOCAL_ONLY=1 使用本地离线模式。"
        ) from exc


def build_vector_store(chunks: List[Any], persist_directory: Path) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_name="esg_report_chunks",
    )


def chunks_fingerprint(chunks: List[Any]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        metadata = chunk.metadata or {}
        digest.update(str(metadata.get("chunk_id", "")).encode("utf-8"))
        digest.update(str(metadata.get("clean_chunk_id", "")).encode("utf-8"))
        digest.update(str(metadata.get("chunk_length", len(chunk.page_content))).encode("utf-8"))
        digest.update(chunk.page_content.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def vector_store_manifest(chunks: List[Any]) -> Dict[str, Any]:
    return {
        "chunk_count": len(chunks),
        "chunks_hash": chunks_fingerprint(chunks),
        "collection_name": "esg_report_chunks",
    }


def is_valid_vector_store(persist_directory: Path, chunks: List[Any]) -> bool:
    manifest_path = persist_directory / "manifest.json"
    db_path = persist_directory / "chroma.sqlite3"
    if not db_path.exists() or not manifest_path.exists():
        return False

    try:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return saved_manifest == vector_store_manifest(chunks)


def load_or_build_vector_store(chunks: List[Any], persist_directory: Path) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    if is_valid_vector_store(persist_directory, chunks):
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=get_embeddings(),
            collection_name="esg_report_chunks",
        )

    if (persist_directory / "chroma.sqlite3").exists():
        shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = build_vector_store(chunks, persist_directory)
    (persist_directory / "manifest.json").write_text(
        json.dumps(vector_store_manifest(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return vector_store

#==================excel========================

def load_requirements_from_excel(excel_path: Path) -> List[Dict[str, str]]:
    raw = pd.read_excel(excel_path, header=None)
    header_row = None
    for i in range(min(10, len(raw))):
        row_values = raw.iloc[i].astype(str).tolist()
        if any("指引条目" in x for x in row_values):
            header_row = i
            break
    if header_row is None:
        raise ValueError("没有找到包含“指引条目”的表头行")

    df = pd.read_excel(excel_path, header=header_row)
    df = df.rename(columns={
        df.columns[0]: "指引条目",
        df.columns[1]: "章",
        df.columns[2]: "节",
        df.columns[3]: "指标",
        df.columns[4]: "指标详情",
        df.columns[5]: "形式",
        df.columns[6]: "重要性" if len(df.columns) > 6 else "Unnamed",
    })
    df = df.dropna(how="all").reset_index(drop=True)

    requirements = []
    current = None
    for _, row in df.iterrows():
        item = row.get("指引条目")
        chapter = row.get("章")
        section = row.get("节")
        indicator = row.get("指标")
        detail = row.get("指标详情")
        form = row.get("形式")
        importance = row.get("重要性")

        item_empty = pd.isna(item) or str(item).strip() == ""
        has_content = (
            not pd.isna(detail) and str(detail).strip() != ""
        ) or (
            not pd.isna(form) and str(form).strip() != ""
        )

        if not item_empty:
            if current is not None:
                requirements.append(current)
            current = {
                "指引条目": str(item).strip(),
                "章": "" if pd.isna(chapter) else str(chapter).strip(),
                "节": "" if pd.isna(section) else str(section).strip(),
                "指标": "" if pd.isna(indicator) else str(indicator).strip(),
                "指标详情": "" if pd.isna(detail) else str(detail).strip(),
                "形式": "" if pd.isna(form) else str(form).strip(),
                "重要性": "" if pd.isna(importance) else str(importance).strip(),
            }
        elif item_empty and has_content and current is not None:
            if not pd.isna(detail) and str(detail).strip():
                current["指标详情"] += "\n" + str(detail).strip()
            if not pd.isna(form) and str(form).strip():
                current["形式"] += "\n" + str(form).strip()
            if not pd.isna(importance) and str(importance).strip():
                current["重要性"] += "\n" + str(importance).strip()

    if current is not None:
        requirements.append(current)
    return requirements


def format_requirement(req: Dict[str, str]) -> str:
    return f"""
{req['指引条目']}
{req['章']}--{req['节']}
指标：{req['指标']}

指标详情：
{req['指标详情']}

形式：
{req['形式']}

重要性：
{req['重要性']}
""".strip()


def build_section_requirements(section_group: Dict[str, Any], req_map: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    return [req_map[item] for item in section_group["items"] if item in req_map]


def format_section_requirement_text(section_group: Dict[str, Any], section_reqs: List[Dict[str, str]]) -> str:
    parts = [
        f"章节主题：{section_group['title']}",
        f"章节说明：{section_group['description']}",
        "",
        "本章节包含以下披露要求：",
    ]
    for req in section_reqs:
        parts.append("\n" + "=" * 40)
        parts.append(format_requirement(req))
    return "\n".join(parts)


def format_all_section_overview(section_groups: List[Dict[str, Any]], current_section_group: Dict[str, Any]) -> str:
    current_type = current_section_group.get("type", "")
    lines = [
        f"当前属于【{current_type}】类议题。",
        "同一类议题下的章节结构如下，请用于判断当前章节与其他章节的边界：",
    ]
    filtered_groups = [group for group in section_groups if group.get("type", "") == current_type]
    for idx, group in enumerate(filtered_groups, start=1):
        items = "、".join(group["items"])
        lines.append(f"{idx}. {group['title']}（{items}）：{group['description']}")
    return "\n".join(lines)


def format_full_section_framework(
    section_groups: List[Dict[str, Any]],
    req_map: Dict[str, Dict[str, str]],
) -> str:
    parts = []
    for idx, group in enumerate(section_groups, start=1):
        section_reqs = build_section_requirements(group, req_map)
        parts.append(
            f"""
SECTION {idx}
section_id: {group["section_id"]}
章节标题: {group["title"]}
章节说明: {group["description"]}
章节类别: {group.get("type", "")}
包含指引条目: {"、".join(group["items"])}

披露要求:
{format_section_requirement_text(group, section_reqs) if section_reqs else "无匹配披露要求"}
""".strip()
        )
    return ("\n\n" + "=" * 80 + "\n\n").join(parts)


def build_chunk_catalogue(chunks: List[Any]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        header = chunk.metadata.get("Header 1", "")
        page = chunk.metadata.get("page", "")
        parts.append(f"CHUNK_ID: {i}\nHEADER: {header}\nPAGE: {page}")
    return "\n".join(parts)


def parse_selected_ids(raw: str) -> List[int]:
    result_match = re.search(r"<result>\s*(\[[\d,\s]+\])\s*</result>", raw, re.DOTALL)
    if result_match:
        list_text = result_match.group(1)
    else:
        list_match = re.search(r"\[[\d,\s]+\]", raw)
        if list_match:
            list_text = list_match.group(0)
        else:
            return list(dict.fromkeys(int(x) for x in re.findall(r"\d+", raw)))

    ids = ast.literal_eval(list_text)
    clean_ids = []
    for x in ids:
        if isinstance(x, int) and x not in clean_ids:
            clean_ids.append(x)
    return clean_ids


def parse_json_object(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"无法解析 JSON:\n{raw}")


def normalize_assignment_result(
    raw_assignments: Any,
    valid_section_ids: set[str],
    chunk_count: int,
    max_chunks_per_section: int,
    max_sections_per_chunk: int,
) -> Dict[str, List[int]]:
    normalized = {section_id: [] for section_id in valid_section_ids}
    if not isinstance(raw_assignments, dict):
        return normalized

    chunk_to_sections: Dict[int, List[str]] = {}
    for section_id, ids in raw_assignments.items():
        section_id = str(section_id)
        if section_id not in valid_section_ids or not isinstance(ids, list):
            continue
        for raw_id in ids:
            if isinstance(raw_id, bool):
                continue
            try:
                chunk_id = int(raw_id)
            except Exception:
                continue
            if chunk_id < 0 or chunk_id >= chunk_count:
                continue
            if chunk_id not in normalized[section_id]:
                normalized[section_id].append(chunk_id)
            chunk_to_sections.setdefault(chunk_id, [])
            if section_id not in chunk_to_sections[chunk_id]:
                chunk_to_sections[chunk_id].append(section_id)

    for section_id, ids in normalized.items():
        normalized[section_id] = ids[:max_chunks_per_section]

    chunk_to_sections = {}
    for section_id, ids in normalized.items():
        for chunk_id in ids:
            chunk_to_sections.setdefault(chunk_id, []).append(section_id)

    for chunk_id, section_ids in chunk_to_sections.items():
        for section_id in section_ids[max_sections_per_chunk:]:
            normalized[section_id] = [
                existing_id for existing_id in normalized[section_id]
                if existing_id != chunk_id
            ]
    return normalized


def assign_chunks_to_sections(
    llm: ChatOpenAI,
    chunks: List[Any],
    section_groups: List[Dict[str, Any]],
    req_map: Dict[str, Dict[str, str]],
    chunk_catalogue: str,
    runs: int = 3,
    min_votes: int = 2,
    max_chunks_per_section: int = 10,
    max_sections_per_chunk: int = 3,
) -> Dict[str, Any]:
    valid_section_ids = {group["section_id"] for group in section_groups}
    section_framework = format_full_section_framework(section_groups, req_map)
    votes: Dict[str, Dict[int, int]] = {
        section_id: {} for section_id in valid_section_ids
    }

    prompt = render_prompt(
        "select_material_chunks_for_section.md",
        section_framework=section_framework,
        chunk_catalogue=chunk_catalogue,
        max_chunks_per_section=max_chunks_per_section,
        max_sections_per_chunk=max_sections_per_chunk,
    )

    for _ in range(runs):
        try:
            data = parse_json_object(llm.invoke(prompt).content)
        except Exception:
            continue
        assignments = normalize_assignment_result(
            data.get("assignments"),
            valid_section_ids,
            len(chunks),
            max_chunks_per_section,
            max_sections_per_chunk,
        )
        for section_id, chunk_ids in assignments.items():
            for chunk_id in chunk_ids:
                votes[section_id][chunk_id] = votes[section_id].get(chunk_id, 0) + 1

    final_map: Dict[str, List[int]] = {}
    for group in section_groups:
        section_id = group["section_id"]
        ranked = sorted(
            votes.get(section_id, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
        selected = [
            chunk_id for chunk_id, vote_count in ranked
            if vote_count >= min_votes
        ][:max_chunks_per_section]
        if not selected:
            selected = [chunk_id for chunk_id, _ in ranked[:5]]
        final_map[section_id] = selected

    chunk_section_pairs = []
    for section_id, chunk_ids in final_map.items():
        for chunk_id in chunk_ids:
            chunk_section_pairs.append((
                chunk_id,
                section_id,
                votes.get(section_id, {}).get(chunk_id, 0),
            ))

    by_chunk: Dict[int, List[tuple[str, int]]] = {}
    for chunk_id, section_id, vote_count in chunk_section_pairs:
        by_chunk.setdefault(chunk_id, []).append((section_id, vote_count))

    for chunk_id, section_votes in by_chunk.items():
        section_votes.sort(key=lambda item: (-item[1], item[0]))
        for section_id, _vote_count in section_votes[max_sections_per_chunk:]:
            final_map[section_id] = [
                existing_id for existing_id in final_map[section_id]
                if existing_id != chunk_id
            ]

    return {
        "section_chunk_map": final_map,
        "section_chunk_votes": {
            section_id: {str(chunk_id): count for chunk_id, count in chunk_votes.items()}
            for section_id, chunk_votes in votes.items()
        },
    }


def select_material_chunks_for_section(
    llm: ChatOpenAI,
    chunks: List[Any],
    section_group: Dict[str, Any],
    section_requirement_text: str,
    chunk_catalogue: str,
    all_section_overview: str,
) -> List[int]:
    single_req_map = {
        item: {
            "指引条目": item,
            "章": "",
            "节": "",
            "指标": "",
            "指标详情": section_requirement_text,
            "形式": "",
            "重要性": "",
        }
        for item in section_group["items"]
    }
    result = assign_chunks_to_sections(
        llm=llm,
        chunks=chunks,
        section_groups=[section_group],
        req_map=single_req_map,
        chunk_catalogue=chunk_catalogue,
        runs=1,
        min_votes=1,
        max_chunks_per_section=TOP_K_TITLE,
    )
    return result["section_chunk_map"].get(section_group["section_id"], [])


def rag_search_ids(vs: Chroma, section_requirement_text: str, k: int = 20) -> List[int]:
    results = vs.similarity_search_with_score(section_requirement_text, k=k)
    ids = []
    for doc, _score in results:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        if chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def tokenize_for_keyword_search(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_%.]+", text)
    short_terms = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            short_terms.extend(token[i:i + 2] for i in range(len(token) - 1))
            short_terms.extend(token[i:i + 3] for i in range(max(0, len(token) - 2)))
            short_terms.extend(token[i:i + 4] for i in range(max(0, len(token) - 3)))
    return list(dict.fromkeys(tokens + short_terms))


def keyword_search_ids(
    chunks: List[Any],
    query: str,
    k: int = 8,
) -> List[int]:
    query_terms = tokenize_for_keyword_search(query)
    if not query_terms:
        return []

    docs_terms = []
    doc_freq: Dict[str, int] = {}
    doc_lengths = []
    for chunk in chunks:
        text = f"{chunk.metadata.get('Header 1', '')}\n{chunk.page_content}"
        terms = tokenize_for_keyword_search(text)
        docs_terms.append(terms)
        doc_lengths.append(len(terms))
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avg_len = sum(doc_lengths) / max(1, len(doc_lengths))
    total_docs = max(1, len(chunks))
    k1 = 1.5
    b = 0.75
    scores = []
    query_text = (query or "").lower().strip()

    for index, terms in enumerate(docs_terms):
        if not terms:
            continue
        term_counts: Dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        doc_len = max(1, doc_lengths[index])
        raw_text = f"{chunks[index].metadata.get('Header 1', '')}\n{chunks[index].page_content}".lower()

        if query_text and query_text in raw_text:
            score += 8.0

        for term in query_terms:
            freq = term_counts.get(term, 0)
            if not freq:
                continue
            idf = max(0.0, ((total_docs - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5)))
            bm25 = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / max(1.0, avg_len)))
            score += bm25
            if len(term) >= 4 and term in raw_text:
                score += 2.5
            elif len(term) >= 3 and term in raw_text:
                score += 1.0

        if score > 0:
            scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _score in scores[:k]]


def merge_retrieval_ids(*groups: List[int], limit: int = 12) -> List[int]:
    merged = []
    for group in groups:
        for chunk_id in group:
            if chunk_id not in merged:
                merged.append(chunk_id)
            if len(merged) >= limit:
                return merged
    return merged


def chat_with_report(
    question: str,
    chunks: List[Any],
    vs: Chroma,
    llm_config: Dict[str, str] | None = None,
    top_k: int = 12,
    keyword_k: int = 10,
) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    rag_ids = rag_search_ids(vs, question, k=top_k)
    keyword_ids = keyword_search_ids(chunks, question, k=keyword_k)
    selected_ids = merge_retrieval_ids(rag_ids, keyword_ids, limit=top_k + keyword_k)
    evidence_context = build_selected_context(chunks, selected_ids)
    llm = get_chat_llm(llm_config)
    prompt = f"""
你是 ESG 报告问答助手。你需要先判断用户问题的类型，再决定是否使用检索证据。

要求：
1. 如果用户是在问当前 ESG 报告、PDF 材料、指标、比例、数据、章节、图片、页码或证据，请只根据下面的检索证据回答。
2. 如果证据中能直接回答，给出简洁、明确的中文答案。
3. 如果涉及比例、金额、排放量、人数等数字，必须原样引用证据中的数值和单位。
4. 必须认真读取 Markdown 的 <details> 块、表格、图片说明和图表标题；这些内容也是证据。
5. 如果用户问的是普通聊天、寒暄、日程、能力说明、与你对话相关的问题，而不是当前报告内容，请自然回复，不要使用检索证据，不要列“来源”。
6. 如果用户问的是报告内容但证据不足，明确回答“当前材料中没有找到足够证据回答这个问题”，不要编造。
7. 只有当你实际使用了检索证据回答报告问题时，才在末尾列出引用来源，格式为“来源：CHUNK_ID 95，第 119 页”。必须使用证据中的真实 CHUNK_ID。

用户问题：
{question}

检索证据：
{evidence_context or "未检索到相关证据。"}
""".strip()
    answer = llm.invoke(prompt).content.strip()
    used_sources = bool(re.search(r"来源\s*[:：]|CHUNK_ID\s*\d+", answer, flags=re.IGNORECASE))
    visible_ids = selected_ids if used_sources else []
    return {
        "question": question,
        "answer": answer,
        "source_chunk_ids": visible_ids,
        "rag_chunk_ids": rag_ids,
        "keyword_chunk_ids": keyword_ids,
        "evidence_pages": unique_evidence_pages(chunks, visible_ids),
        "source_previews": [
            {
                "chunk_id": i,
                "header": chunks[i].metadata.get("Header 1", ""),
                "pages": chunks[i].metadata.get("pages") or [chunks[i].metadata.get("page")],
                "preview": re.sub(r"\s+", " ", chunks[i].page_content).strip()[:500],
            }
            for i in visible_ids
            if isinstance(i, int) and 0 <= i < len(chunks)
        ],
    }


def build_candidate_context(chunks: List[Any], candidate_ids: List[int]) -> str:
    parts = []
    for i in candidate_ids:
        if not isinstance(i, int) or i < 0 or i >= len(chunks):
            continue
        chunk = chunks[i]
        header = chunk.metadata.get("Header 1", "")
        page = chunk.metadata.get("page", "")
        content = chunk.page_content
        parts.append(f"""
CHUNK_ID: {i}
HEADER: {header}
PAGE: {page}
CONTENT:
{content}
---
""")
    return "\n".join(parts)


def rerank_material_chunks_for_section(
    llm: ChatOpenAI,
    chunks: List[Any],
    section_group: Dict[str, Any],
    section_requirement_text: str,
    candidate_ids: List[int],
    all_section_overview: str,
    max_k: int = 12,
) -> List[int]:
    candidate_ids = [
        i for i in candidate_ids
        if isinstance(i, int) and 0 <= i < len(chunks)
    ]
    if not candidate_ids:
        return []
    candidate_context = build_candidate_context(chunks, candidate_ids)
    prompt = render_prompt(
        "rerank_material_chunks_for_section.md",
        section_title=section_group["title"],
        section_description=section_group["description"],
        all_section_overview=all_section_overview,
        section_requirement_text=section_requirement_text,
        candidate_context=candidate_context,
        max_k=max_k,
    )
    selected_ids = parse_selected_ids(llm.invoke(prompt).content)
    valid_ids = [i for i in selected_ids if i in candidate_ids and 0 <= i < len(chunks)]
    return valid_ids[:max_k]


def build_selected_context(chunks: List[Any], selected_ids: List[int]) -> str:
    parts = []
    for n, i in enumerate(selected_ids, start=1):
        chunk = chunks[i]
        pages = chunk.metadata.get("pages") or [chunk.metadata.get("page")]
        page_text = "、".join(str(p) for p in pages if p is not None) or "未知"
        parts.append(f"""

## 选中证据 {n}

CHUNK_ID: {i}

PAGE:
{page_text}

HEADER:
{chunk.metadata.get("Header 1", "")}

CONTENT:
{chunk.page_content}

""")
    return "\n".join(parts)


def build_image_evidence(chunks: List[Any], selected_ids: List[int]) -> str:
    image_items = extract_image_items_from_final_chunks(chunks, selected_ids)
    if not image_items:
        return "未发现图片证据。"

    parts = []
    for idx, item in enumerate(image_items, start=1):
        parts.append(f"""
图片 {idx}
CHUNK_ID: {item["chunk_id"]}
IMAGE_PATH: {item["image_path"]}

图片前文：
{item.get("before_text", "")}

图片块：
{item.get("image_block", "")}

图片后文：
{item.get("after_text", "")}
""".strip())
    return "\n\n---\n\n".join(parts)


def extract_image_blocks_from_chunk(chunk: Any, chunk_id: int, window_up: int = 20, window_down: int = 100) -> List[Dict[str, Any]]:
    text = chunk.page_content
    items = []
    pattern = r'!\[[^\]]*\]\((.*?)\)(?:\s*<details>.*?</details>)?'

    for match in re.finditer(pattern, text, flags=re.DOTALL):
        image_path = match.group(1).strip()
        start = max(0, match.start() - window_up)
        end = min(len(text), match.end() + window_down)
        items.append({
            "chunk_id": chunk_id,
            "image_path": image_path,
            "image_block": match.group(0),
            "before_text": text[start:match.start()].strip(),
            "after_text": text[match.end():end].strip(),
        })
    return items


def extract_image_items_from_final_chunks(chunks: List[Any], final_ids: List[int]) -> List[Dict[str, Any]]:
    image_items = []
    for chunk_id in final_ids:
        image_items.extend(extract_image_blocks_from_chunk(chunks[chunk_id], chunk_id))
    return image_items


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: str) -> str:
    ext = image_path.lower().split(".")[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    return "image/jpeg"


def render_pdf_page_to_base64(pdf_path: Path, page_no: int, scale: float = 2.0) -> str:
    import fitz

    pdf = fitz.open(str(pdf_path))
    try:
        if page_no < 1 or page_no > len(pdf):
            return ""
        page = pdf[page_no - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")
    finally:
        pdf.close()


def figure_section(
    llm: ChatOpenAI,
    section_md: str,
    section_requirement_text: str,
    image_items: List[Dict[str, Any]],
) -> str:
    if not image_items:
        return section_md

    image_contents = []
    valid_count = 0
    for item in image_items:
        image_path = item["image_path"]
        if not os.path.exists(image_path):
            continue

        valid_count += 1
        image_contents.append({
            "type": "text",
            "text": f"""
图片 {valid_count}
IMAGE_PATH: {image_path}
CHUNK_ID: {item["chunk_id"]}


图片前文：
{item.get("before_text", "")}

图片块：
{item.get("image_block", "")}

图片后文：
{item.get("after_text", "")}
""",
        })
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{get_image_mime(image_path)};base64,{image_to_base64(image_path)}"
            },
        })

    if valid_count == 0:
        return section_md

    prompt = render_prompt(
        "figure_section.md",
        section_requirement_text=section_requirement_text,
        section_md=section_md,
    )
    msg = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + image_contents,
    }]
    return llm.invoke(msg).content.strip()


def write_section(
    llm: ChatOpenAI,
    section_group: Dict[str, Any],
    section_reqs: List[Dict[str, str]],
    section_requirement_text: str,
    selected_context: str,
) -> str:
    subsection_titles = "\n".join(
        f"- {req['指引条目']}：{req['指标']}"
        for req in section_reqs
    )
    first_req = section_reqs[0] if section_reqs else {"指引条目": "XX条", "指标": "指标"}
    prompt = render_prompt(
        "write_section.md",
        section_title=section_group["title"],
        section_description=section_group["description"],
        subsection_titles=subsection_titles,
        section_requirement_text=section_requirement_text,
        selected_context=selected_context,
        first_requirement_item=first_req["指引条目"],
        first_requirement_indicator=first_req["指标"],
    )
    return llm.invoke(prompt).content


def extract_image_occurrences_from_markdown(text: str, window: int = 260) -> List[Dict[str, str]]:
    occurrences = []
    pattern = r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        image_path = (match.group(1) or match.group(2) or "").strip()
        if not image_path:
            continue
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        occurrences.append({
            "image_path": image_path,
            "image_markup": match.group(0),
            "before_text": text[start:match.start()].strip(),
            "after_text": text[match.end():end].strip(),
        })
    return occurrences


def build_full_report_image_inventory(
    full_report: str,
    sections: List[Dict[str, Any]] | None = None,
    chunks: List[Any] | None = None,
) -> str:
    sections = sections or []
    chunks = chunks or []
    section_occurrences = []
    path_counts: Dict[str, int] = {}

    for section in sections:
        section_title = section.get("title", "")
        section_id = section.get("section_id", "")
        content = section.get("content", "")
        for occurrence in extract_image_occurrences_from_markdown(content):
            occurrence["section_title"] = section_title
            occurrence["section_id"] = section_id
            section_occurrences.append(occurrence)
            path = occurrence["image_path"]
            path_counts[path] = path_counts.get(path, 0) + 1

    if not section_occurrences:
        for occurrence in extract_image_occurrences_from_markdown(full_report):
            section_occurrences.append(occurrence)
            path = occurrence["image_path"]
            path_counts[path] = path_counts.get(path, 0) + 1

    if not section_occurrences:
        return "全文未发现图片。"

    evidence_by_path: Dict[str, List[Dict[str, Any]]] = {}
    for section in sections:
        for raw_id in section.get("selected_chunk_ids", []):
            try:
                chunk_id = int(raw_id)
            except Exception:
                continue
            if chunk_id < 0 or chunk_id >= len(chunks):
                continue
            for item in extract_image_blocks_from_chunk(chunks[chunk_id], chunk_id,window_up = 20, window_down = 100):
                evidence_by_path.setdefault(item["image_path"], []).append(item)

    lines = []
    for idx, occurrence in enumerate(section_occurrences, start=1):
        path = occurrence["image_path"]
        count = path_counts.get(path, 1)
        duplicate_note = "（重复出现）" if count > 1 else ""
        evidence_items = evidence_by_path.get(path, [])
        evidence_text = "未在选中 chunk 中找到该图片的原始图片块。"
        if evidence_items:
            item = evidence_items[0]
            evidence_text = f"""
原始 CHUNK_ID：{item["chunk_id"]}

原始图片前文：
{item.get("before_text", "")}

原始图片块 / details：
{item.get("image_block", "")}

原始图片后文：
{item.get("after_text", "")}
""".strip()

        lines.append(f"""
图片 {idx}
IMAGE_PATH: {path}
全文出现次数：{count}{duplicate_note}
所在 section：{occurrence.get("section_title", "未知")}（{occurrence.get("section_id", "未知")}）

全文当前位置前文：
{occurrence.get("before_text", "")}

全文图片标记：
{occurrence.get("image_markup", "")}

全文当前位置后文：
{occurrence.get("after_text", "")}

原始证据信息：
{evidence_text}
""".strip())
    return "\n\n---\n\n".join(lines)


def revise_whole_report(
    llm: ChatOpenAI,
    section_groups: List[Dict[str, Any]],
    req_map: Dict[str, Dict[str, str]],
    full_report: str,
    sections: List[Dict[str, Any]] | None = None,
    chunks: List[Any] | None = None,
) -> str:
    if not full_report.strip():
        return full_report

    prompt = render_prompt(
        "whole_section_revision.md",
        section_framework=format_full_section_framework(section_groups, req_map),
        image_inventory=build_full_report_image_inventory(full_report, sections, chunks),
        full_report=full_report,
    )
    revised = llm.invoke(prompt).content.strip()
    return revised or full_report


def check_generated_section(
    chunks: List[Any],
    req_map: Dict[str, Dict[str, str]],
    section_group: Dict[str, Any],
    section_result: Dict[str, Any],
    check_mode: str = "text",
    pdf_path: Path | None = None,
    llm_config: Dict[str, str] | None = None,
) -> str:
    check_mode = "vision" if check_mode == "vision" else "text"
    checker_llm = get_vision_checker_llm(llm_config) if check_mode == "vision" else get_checker_llm(llm_config)
    section_reqs = build_section_requirements(section_group, req_map)
    section_requirement_text = format_section_requirement_text(section_group, section_reqs)
    selected_ids = []
    for raw_id in section_result.get("selected_chunk_ids", []):
        try:
            chunk_id = int(raw_id)
        except Exception:
            continue
        if 0 <= chunk_id < len(chunks) and chunk_id not in selected_ids:
            selected_ids.append(chunk_id)
    evidence_pages = section_result.get("evidence_pages") or unique_evidence_pages(chunks, selected_ids)
    check_mode_note = (
        "视觉检查：除文本证据外，本次还提供 PDF 证据页截图。请直接查看截图中的图片、图表标题、图注、表格和页内上下文。"
        if check_mode == "vision"
        else "普通文本检查：本次只提供披露框架、生成正文、证据 chunk 文本、证据页码和图片前后文，不提供 PDF 页面截图。"
    )
    prompt = render_prompt(
        "check_section.md",
        section_title=section_group["title"],
        section_requirement_text=section_requirement_text,
        generated_section=section_result.get("content", ""),
        selected_chunk_ids=", ".join(str(i) for i in selected_ids) or "无",
        evidence_pages=", ".join(str(p) for p in evidence_pages) or "无",
        check_mode_note=check_mode_note,
        evidence_context=build_selected_context(chunks, selected_ids) or "未提供证据原文。",
        image_evidence=build_image_evidence(chunks, selected_ids),
    )

    if check_mode != "vision" or pdf_path is None:
        return checker_llm.invoke(prompt).content.strip()

    max_pages = int(os.getenv("CHECK_VISION_MAX_PAGES", "6"))
    page_contents = []
    for page_no in sorted(dict.fromkeys(evidence_pages))[:max_pages]:
        try:
            page_no = int(page_no)
        except Exception:
            continue
        page_b64 = render_pdf_page_to_base64(pdf_path, page_no)
        if not page_b64:
            continue
        page_contents.append({
            "type": "text",
            "text": f"PDF 证据页截图：第 {page_no} 页",
        })
        page_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page_b64}",
            },
        })

    if not page_contents:
        return checker_llm.invoke(prompt).content.strip()

    msg = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + page_contents,
    }]
    return checker_llm.invoke(msg).content.strip()


def unique_evidence_pages(chunks: List[Any], selected_ids: List[int]) -> List[int]:
    pages = []
    for i in selected_ids:
        if not isinstance(i, int) or i < 0 or i >= len(chunks):
            continue
        values = chunks[i].metadata.get("pages")
        if values is None:
            values = [chunks[i].metadata.get("page")]
        elif not isinstance(values, list):
            values = [values]
        for page in values:
            if page is None:
                continue
            try:
                page_no = int(page)
            except Exception:
                continue
            if page_no not in pages:
                pages.append(page_no)
    return sorted(pages)


def generate_section(
    chunks: List[Any],
    vs: Chroma,
    req_map: Dict[str, Dict[str, str]],
    section_group: Dict[str, Any],
    output_dir: Path,
    chunk_catalogue: str,
    llm_config: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    selector_llm = get_selector_llm(llm_config)
    writer_llm = get_writer_llm(llm_config)
    figure_llm = get_figure_llm(llm_config)

    section_reqs = build_section_requirements(section_group, req_map)
    if not section_reqs:
        return {
            "section_id": section_group["section_id"],
            "title": section_group["title"],
            "status": "skipped",
            "content": "",
            "selected_chunk_ids": [],
            "evidence_pages": [],
        }

    section_requirement_text = format_section_requirement_text(section_group, section_reqs)
    all_section_overview = format_all_section_overview(SECTION_GROUPS, section_group)
    title_ids = select_material_chunks_for_section(
        selector_llm,
        chunks,
        section_group,
        section_requirement_text,
        chunk_catalogue,
        all_section_overview,
    )
    rag_ids = rag_search_ids(vs, section_requirement_text, k=20)

    candidate_ids = []
    for i in title_ids + rag_ids:
        if 0 <= i < len(chunks) and i not in candidate_ids:
            candidate_ids.append(i)

    if candidate_ids:
        final_ids = rerank_material_chunks_for_section(
            selector_llm,
            chunks,
            section_group,
            section_requirement_text,
            candidate_ids,
            all_section_overview,
            max_k=12,
        )
        if not final_ids:
            final_ids = candidate_ids[:12]
    else:
        final_ids = []
    selected_context = build_selected_context(chunks, final_ids)
    content = write_section(
        writer_llm,
        section_group,
        section_reqs,
        section_requirement_text,
        selected_context,
    )
    image_items = extract_image_items_from_final_chunks(chunks, final_ids)
    content = figure_section(
        figure_llm,
        content,
        section_requirement_text,
        image_items,
    )
    content = sanitize_markdown_for_pypandoc(content)

    section_path = output_dir / f"{section_group['section_id']}.md"
    section_path.write_text(content, encoding="utf-8")

    return {
        "section_id": section_group["section_id"],
        "title": section_group["title"],
        "description": section_group["description"],
        "status": "generated",
        "content": content,
        "title_ids": title_ids,
        "rag_ids": rag_ids,
        "candidate_ids": candidate_ids,
        "selected_chunk_ids": final_ids,
        "evidence_pages": unique_evidence_pages(chunks, final_ids),
    }
