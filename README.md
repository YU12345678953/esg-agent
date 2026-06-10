# ESG Agent 
ESG 报告智能生成助手。上传 PDF 文件，自动解析、检索、生成结构化的 ESG 分析报告。


核心流程：

```text
上传 PDF
→ 选择解析模式
→ MinerU 解析 PDF
→ 清洗 Markdown / 图片路径
→ 按标题和长度分 chunk
→ 给 chunk 写入页码 metadata
→ 构建 Chroma 检索库
→ 按 step6 的逻辑逐章生成
→ 网页左侧看章节，右侧看证据页
```

## 文件结构

```text
esg_agent/
├── api.py                 # FastAPI 后端，上传文件和后台生成
├── web_ui.py              # Streamlit 前端
├── graph_runner.py        # LangGraph 流程编排和中断续跑
├── pipeline.py            # PDF 解析、清洗、chunk、页码映射
├── generation.py          # ESG section 生成逻辑
├── ESG披露框架.xlsx        # 默认披露框架
├── .env                   # 本地 API Key 配置（不要发给别人）
├── prompts/               # 可维护 prompt 模板
├── sessions/              # 每个上传任务的文件和结果
└── requirements.txt
```

## 输出

每个 session 会保存到：

```text
esg_agent/sessions/{session_id}/
├── source.pdf
├── session.json
├── messages.json
├── docs_raw.pkl
├── docs_cleaned.pkl
├── chunks.pkl
├── chroma/
├── {section_id}.md
└── full_report.md
```

后台还会保存 LangGraph 的运行状态：

```text
esg_agent/sessions/{session_id}/graph_state.json
esg_agent/sessions/{session_id}/checkpoints.sqlite
```

`checkpoints.sqlite` 是 LangGraph 官方 SQLite checkpointer，用于中断续跑；`graph_state.json` 只是方便人工查看的状态镜像。

如果生成中断，网页里加载同一个 Session ID 后点击“从中断处继续”，会复用已经完成的中间产物和章节结果继续生成。选中某个章节后，也可以点击“重新生成当前章节”，只替换该章节并重建完整报告。

## 页码说明

网页上传时可以选择两种模式：

```text
快速模式：MinerU split_pages=False 整篇解析，再用 PyMuPDF 页文本和 MinerU markdown 全局对齐，先插入 PAGE_START/PAGE_END，再分 chunk。
精准模式：MinerU split_pages=True 分页解析，之后按 step1-3 的 PAGE_START 逻辑合并、清洗、再拆回分页 chunk。
```

每个 chunk 会尽量写入：

```python
chunk.metadata["page"]
chunk.metadata["pages"]
```

网页右侧只展示当前 section 的证据页。

## BGE 模型

embedding 使用 `BAAI/bge-m3`。为了避免国内网络访问 Hugging Face 失败，系统在没有本地模型时会默认设置镜像：

```text
HF_ENDPOINT=https://hf-mirror.com
```

模型查找顺序：

```text
1. EMBEDDING_MODEL_PATH 指向的本地目录
2. esg_agent/local_models/bge-m3
3. EMBEDDING_MODEL，默认 BAAI/bge-m3
```

开发时可以直接让它第一次通过镜像下载。部署或给别人用时，推荐把 `bge-m3` 放到：

```text
esg_agent/local_models/bge-m3
```

或者启动前指定本地路径：

```bash
export EMBEDDING_MODEL_PATH=/path/to/bge-m3
```

如果要强制本地离线模式：

```bash
export EMBEDDING_LOCAL_ONLY=1
```

## LLM 选择

网页上传前可以在侧边栏的“LLM 设置”里选择不同环节使用的模型：

```text
材料筛选 / rerank：DeepSeek、Kimi、MiniMax
正文写作：DeepSeek、Kimi、MiniMax
图片插入 / 图题处理：Kimi、MiniMax
普通检查：DeepSeek、Kimi、MiniMax
视觉检查：Kimi、MiniMax
```

DeepSeek 不用于需要多模态能力的环节，因此不能选择为“图片插入 / 图题处理”和“视觉检查”。如果选择了某个 provider，但没有配置对应 API key，创建会话会直接报错：

```text
DEEPSEEK_API_KEY
MOONSHOT_API_KEY
MINIMAX_API_KEY
```

## 快速开始

### 1. 获取项目

```bash
git clone https://github.com/YU12345678953/esg-agent.git
cd esg-agent
```

如果你拿到的是别人发来的 `esg_agent` 文件夹，请进入它的上级目录。例如：

```bash
cd /path/to/project-parent
ls esg_agent
```

### 2. 安装依赖

```bash
pip3 install -r esg_agent/requirements.txt
```

### 3. 配置 API Key

项目需要以下 API Key，请选择至少一种 LLM 提供商：

| 提供商 | 环境变量 | 用途 |
|--------|----------|------|
| DeepSeek | `DEEPSEEK_API_KEY` | LLM 推理（文本任务） |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` | LLM 推理（支持多模态） |
| MiniMax | `MINIMAX_API_KEY` | LLM 推理（支持多模态） |
| MinerU | `MINERU_TOKEN` | PDF 解析服务 |

**配置方式一：环境变量（推荐）**

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="sk-..."
export MOONSHOT_API_KEY="sk-..."
export MINERU_TOKEN="..."
```

**配置方式二：.env 文件**

在 `esg_agent/` 目录内创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-...
MOONSHOT_API_KEY=sk-...
MINIMAX_API_KEY=sk-...
MINERU_TOKEN=...
```

> 注意：`.env` 包含密钥，不要发给别人。可以提供 `.env.example` 让使用者自行填写。

默认披露框架文件也放在 `esg_agent/` 目录内：

```text
esg_agent/ESG披露框架.xlsx
```

网页里的“披露框架 Excel”默认填写 `ESG披露框架.xlsx`，后端会从 `esg_agent/` 目录查找该文件。

### 4. 启动服务

**启动后端（FastAPI）：**

```bash
# 必须在 esg_agent 的上级目录执行
PORT=8001 python3 -m esg_agent.api
```

后端默认运行在：`http://localhost:8001`

**启动前端（Streamlit）：**

```bash
# 必须在 esg_agent 的上级目录执行
ESG_API_BASE=http://localhost:8001 python3 -m streamlit run esg_agent/web_ui.py --server.port 8510
```

前端默认运行在：`http://localhost:8510`

### 5. 开始使用

浏览器打开 `http://localhost:8510`，上传 PDF 文件开始分析。

---

## 完整使用示例
```
# 1. 克隆项目
git clone https://github.com/YU12345678953/esg-agent.git
cd esg-agent

# 2. 安装依赖
pip3 install -r esg_agent/requirements.txt

# 3. 配置 API Key
cat > esg_agent/.env <<'EOF'
DEEPSEEK_API_KEY=your-deepseek-key
MOONSHOT_API_KEY=your-moonshot-key
MINIMAX_API_KEY=your-minimax-key
MINERU_TOKEN=your-mineru-token
EOF

# 4. 启动后端
PORT=8001 python3 -m api

# 5. 新开终端，启动前端
cd /path/to/esg-agent
ESG_API_BASE=http://localhost:8001 python3 -m streamlit run web_ui.py --server.port 8510

# 6. 浏览器访问
http://localhost:8510
```
