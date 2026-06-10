## ESG 报告生成 Agent 项目报告

### 1 项目背景与目标

随着 ESG（Environmental, Social and Governance）信息披露要求的不断完善，企业需要按照监管机构发布的披露框架，对环境、社会及治理相关信息进行系统化、结构化披露。然而，现有可持续发展报告通常以企业自主编写为主，其内容组织方式与披露框架要求之间存在一定差异，导致信息检索、合规检查以及报告复用成本较高。

本项目旨在利用大语言模型（LLM）、检索增强生成（Retrieval-Augmented Generation, RAG）以及 Agent 工作流技术，对企业可持续发展报告进行自动化解析、检索与重组。系统以监管披露框架为目标结构，从原始报告中提取相关证据，并按照披露要求逐项生成内容，从而形成结构清晰、证据可追溯且符合披露规范的 ESG 报告。

项目主要目标包括：

1. 对企业 PDF 可持续发展报告进行结构化解析，提取文本、表格、图片及页码信息。
2. 将解析后的内容切分为适合检索的 Chunk，并保留来源页码、标题、图片路径等 Metadata。
3. 基于披露框架自动分配候选材料，并通过 RAG 检索与 LLM rerank 筛选证据。
4. 按披露章节自动生成 ESG 报告正文，并支持图片插入、Markdown 清洗与 Word 导出。
5. 提供 Web UI，使用户能够上传报告、查看生成章节、定位证据页、重新生成章节、检查章节质量以及对报告进行问答。

### 2 系统总体架构

本项目整体采用“PDF 解析预处理 + 向量检索 + LangGraph Agent 工作流 + Web UI”的架构。系统由以下几个核心模块组成：

```text
用户上传 PDF / 披露框架 Excel
        |
        v
PDF 解析与预处理 pipeline.py
        |
        v
Chunk 清洗、切分、页码映射、图片保存
        |
        v
向量库构建 Chroma + HuggingFace Embeddings
        |
        v
LangGraph 工作流 graph_runner.py
        |
        +--> 材料分配
        +--> RAG 检索
        +--> 候选证据整理
        +--> LLM rerank
        +--> 章节写作
        +--> 图片插入
        +--> Markdown 标准化
        +--> 章节保存
        +--> 全文修订与导出
        |
        v
FastAPI 后端 api.py
        |
        v
Streamlit 前端 web_ui.py
```

从职责划分上看：

- `pipeline.py` 负责 PDF 解析、图片保存、文本清洗、Chunk 构建和页码映射。
- `generation.py` 负责 LLM 配置、披露框架读取、材料筛选、RAG 检索、章节写作、图片插入、检查和问答。
- `graph_runner.py` 负责 LangGraph 工作流编排、断点续跑、章节版本管理和状态投影。
- `api.py` 负责对外提供 FastAPI 接口，包括创建会话、继续生成、重新生成、章节检查、报告问答和文件下载。
- `web_ui.py` 负责 Streamlit 用户界面。
- `markdown_sanitizer.py` 和 `word_export.py` 负责 Markdown 标准化和 Word 导出。
- `session_store.py` 负责统一管理 UI 层的 `session.json` 和 `messages.json` 状态写入。

系统状态被区分为两类：

- `graph_state.json` 与 `checkpoints.sqlite`：LangGraph 工作流状态，用于断点续跑和内部流程恢复。
- `session.json`：UI 展示状态，是从工作流状态投影出来的用户可见快照。

这种设计避免了 UI 状态与工作流状态混淆，使系统在重新生成、检查、聊天问答等并发操作下更容易保持一致。

### 3 文件预处理

#### 3.1 PDF 解析

本项目采用 MinerU 作为 PDF 解析工具。相比传统 PDF 文本提取工具，MinerU 能够较好地保留文档结构信息，并支持表格、图片以及部分矢量图内容的解析，因此适用于可持续发展报告等图文混排文档的处理。为了便于与后续 Agent 工作流集成，本项目使用 LangChain 提供的 `MinerULoader` 对 PDF 文件进行加载，并将解析结果统一转换为 LangChain 的 `Document` 格式，从而能够直接接入文本切分、向量化以及检索模块。

本项目提供两种解析方式：快速模式与精准模式。

在**快速模式**下，`MinerULoader` 以整篇文档为单位进行解析，通常会将整个 PDF 转换为一个 Markdown 文档。该模式解析速度较快，但由于未按页返回 `Document`，原始页码信息会在解析过程中丢失。为了解决这一问题，系统会使用 PyMuPDF 提取原始 PDF 中每一页的文本内容，并在完成文本切分后，对每个 Chunk 进行页码匹配，以便实现 UI 中的证据页对照功能。

在**精准模式**下，MinerU 按页解析文档，每页对应一个独立的 `Document` 对象，并在元数据中保留页码信息。随后系统将这些分页 `Document` 合并为一个带有 `PAGE_START` 与 `PAGE_END` 标记的 Markdown 文档。相比快速模式，精准模式的页码来源更准确，因此更适合需要精确证据页映射和原文定位的场景。

在实际使用过程中发现，`MinerULoader` 默认仅返回图片引用路径，而不会自动将图片保存到本地目录。为了支持后续 ESG 报告中的图文引用功能，本项目在解析阶段提取图片二进制数据并保存至本地文件系统，同时将 Markdown 中的图片引用替换为本地绝对路径，使后续图片筛选、图片插入和 Word 导出能够正常使用。

此外，MinerU 对单个 PDF 文件存在页数与文件大小限制。当前实践中发现，当报告超过 200 页时，需要将原始 PDF 拆分为多个子文件，再分别调用 MinerU 进行解析，最后合并解析结果。系统使用 PyMuPDF 完成 PDF 拆分和页码偏移修正。不过从效率角度看，拆分并不一定线性提升速度。例如，一个 197 页文件 MinerU 解析时间约为 5-8 分钟，而上传 212 页文件并拆分为 200 页与 12 页后，解析耗时可能显著增加。因此，后续需要对大文件解析策略继续优化，例如引入解析缓存、失败重试、分段状态记录和异步任务队列。

#### 3.2 文档清洗与切分

##### 3.2.1 移除特定图片

在最终 ESG 报告中，并非所有原报告图片都适合保留。例如风景图、人物图、装饰图以及没有实际信息量的矢量图，往往不能支撑披露要求，反而会降低报告的专业性。

MinerU 对部分自然图片会生成带有 `<details>` 和 `natural_image` 标记的结构，而一些无意义矢量图则可能表现为没有有效说明的图片引用。系统通过正则表达式识别并移除此类图片块，主要包括：

- 空 alt 的自然图片。
- 带有 `natural_image` 标记的图片。
- 没有 `<details>` 说明信息、且缺少实际上下文价值的图片引用。

该步骤的目标不是删除所有图片，而是尽量过滤掉无法支撑 ESG 披露内容的低价值图片，为后续图片插入环节保留更高质量的候选素材。

##### 3.2.2 文本切分

为了兼顾语义完整性与检索精度，本项目采用分层切分（Hierarchical Chunking）策略构建检索单元（Chunk）。

第一步，系统使用 `MarkdownHeaderTextSplitter` 按一级标题进行切分。这样可以尽量保持章节语义边界，使 Chunk 与原报告章节结构保持一致。

第二步，对于长度超过 3000 字符的文本块，进一步采用 `RecursiveCharacterTextSplitter` 进行递归切分。当前系统设置为：

```python
chunk_size = 1200
chunk_overlap = 200
```

即每个 Chunk 最大长度约为 1200 个字符，相邻 Chunk 保留约 200 个字符的重叠区域。该策略能够降低关键信息被切断的风险，并提高后续向量检索的召回效果。

第三步，系统会对过小 Chunk 进行合并。标题切分后可能产生仅包含章节标题或内容过短的 Chunk，例如：

```markdown
# 环境管理
```

此类 Chunk 单独进行向量化意义有限。因此系统设计了 Chunk 合并机制，对于长度较短的 Chunk 或仅包含标题的 Chunk，将其与相邻 Chunk 合并。同时在合并过程中同步更新页码信息和标题 Metadata，从而减少碎片化文本对检索效果的影响。

精准模式与快速模式在页码处理上有所不同：

- 精准模式下，MinerU 按页返回 `Document`，页码信息由解析阶段直接提供，并在 Chunk 构建过程中保留于 Metadata 中。
- 快速模式下，MinerU 没有可靠页码，系统会对整篇 Markdown 进行标题切分和递归切分。Chunk 生成后，再利用 PyMuPDF 提取的原始 PDF 页文本与 Chunk 内容进行匹配，根据文本匹配结果确定其最可能对应的页码范围，并写入 Metadata。

##### 3.2.3 单元 Chunk 清洗

在完成文本切分后，系统引入 LLM 对 Chunk 进行筛选。模型依据标题及正文内容预览识别目录、索引表、指标对照表、附录、封面、封底等低价值 Chunk。

该步骤采用保守删除策略：只有当模型高度确定某个 Chunk 对 ESG 正文生成没有帮助时，才将其删除。对于可能包含措施、目标、数据、治理结构、案例、图表说明等信息的 Chunk，系统默认保留。这样可以在提升检索质量的同时，降低误删关键证据的风险。

#### 3.3 内容检索

系统使用 Chroma 作为本地向量数据库，并通过 HuggingFace Embeddings 对清洗后的 Chunk 进行向量化。向量库保存在每个会话目录下的 `chroma/` 文件夹中，并通过 manifest 文件记录 Chunk 指纹。当 Chunk 内容未变化时，系统会复用已有向量库，避免重复 embedding。

检索主要服务于三个场景：

1. 章节生成中的 RAG 检索。
2. 用户对报告材料的即时问答。
3. 证据页定位与来源追踪。

在章节生成过程中，系统会基于当前章节的披露要求构造查询文本，并从向量库中检索相关 Chunk。检索结果与前置材料分配结果合并后，形成候选证据集合。随后，LLM 会对候选证据进行 rerank，选出最适合当前章节的材料。

在问答场景中，为了提高响应速度，系统不进行 LLM rerank，而是直接使用向量检索得到 top-k Chunk，并要求 LLM 严格基于检索证据回答用户问题。例如用户询问“女性员工的比例是多少”，系统会检索员工相关 Chunk，并在回答中返回证据页和 Chunk ID。

### 4 披露框架与章节组织

本项目以 Excel 格式的 ESG 披露框架作为目标结构来源。系统读取 Excel 后，将监管披露要求整理为结构化数据，包括：

- 指引条目。
- 章、节。
- 指标。
- 指标详情。
- 形式。
- 重要性。

在 `generation.py` 中，系统预定义了 ESG 报告的目标章节结构 `SECTION_GROUPS`，包括气候治理、气候战略与转型计划、温室气体排放、污染防治、资源利用、员工、客户、供应链、可持续发展治理机制等章节。每个章节会关联一个或多个披露条目。

生成某一章节时，系统会根据该章节包含的披露条目，从 Excel 框架中抽取对应要求，并拼接为章节级披露要求文本。该文本既用于 RAG 检索，也用于后续写作提示词，使生成内容能够围绕监管披露要求展开，而不是简单复述原报告结构。

### 5 Agent 工作流设计

本项目使用 LangGraph 编排 ESG 报告生成流程。相比普通脚本串行执行，LangGraph 能够将复杂流程拆分为多个节点，并通过 checkpoint 机制支持断点续跑。

当前图结构主要包括以下节点：

```text
preprocess
  -> build_vector_store
  -> load_requirements
  -> assign_chunks_to_sections
  -> start_section
  -> rag_search
  -> build_candidate_context
  -> rerank_chunks
  -> write_section
  -> insert_figures
  -> sanitize_markdown
  -> save_section
  -> finalize
```

#### 5.1 预处理节点

`preprocess` 节点负责调用 `run_preprocessing()` 完成 PDF 解析、图片保存、文本清洗、Chunk 构建和无用 Chunk 清理。若会话目录中已经存在 `chunks_clean_for_rag.pkl`，系统会跳过 PDF 解析，从已有 Chunk 继续执行。

#### 5.2 向量库构建节点

`build_vector_store` 节点读取清洗后的 Chunk，并调用 Chroma 构建或加载向量库。系统会根据 Chunk 指纹判断已有向量库是否有效，从而避免重复构建。

#### 5.3 披露要求加载节点

`load_requirements` 节点读取 ESG 披露框架 Excel，并转化为结构化要求列表。该结果会写入 LangGraph state，供后续章节生成使用。

#### 5.4 材料分配节点

`assign_chunks_to_sections` 节点使用 LLM 将 Chunk 初步分配到目标 ESG 章节。模型会根据披露框架、章节说明和 Chunk 标题信息，判断每个 Chunk 更可能服务于哪些章节。

该步骤不同于 RAG 检索。材料分配更像是全局视角下的预筛选，有助于将明显相关的材料提前放入章节候选池中。之后的 RAG 检索会从语义相似角度进一步补充材料。

#### 5.5 章节生成节点

每个章节会依次经过：

1. `start_section`：准备当前章节的披露要求、章节上下文和预分配 Chunk。
2. `rag_search`：基于披露要求从向量库中检索相关 Chunk。
3. `build_candidate_context`：合并预分配 Chunk 和 RAG Chunk，构造候选上下文。
4. `rerank_chunks`：使用 LLM 从候选 Chunk 中选择最终证据。
5. `write_section`：基于披露要求和最终证据生成章节正文。
6. `insert_figures`：从最终证据 Chunk 中提取图片候选，并由视觉模型判断是否插入。
7. `sanitize_markdown`：规范 Markdown 格式，使其更适合后续 pypandoc 转 Word。
8. `save_section`：保存章节结果、证据页、Chunk ID 和版本信息。

#### 5.6 全文修订节点

所有章节生成完成后，`finalize` 节点会将章节内容合并为完整报告，并调用 LLM 进行全文结构审校、图片去重和最终修订。修订后的 Markdown 会再次经过标准化处理，并保存为 `full_report.md`。

### 6 章节重新生成与版本管理

系统支持用户对已生成章节进行重新生成。重新生成并不是简单调用一个单独函数，而是复用 LangGraph 中的章节生成节点链路：

```text
start_section
  -> rag_search
  -> build_candidate_context
  -> rerank_chunks
  -> write_section
  -> insert_figures
  -> sanitize_markdown
  -> save_section
  -> finish_regenerate
```

这样可以保证重新生成与初次生成使用相同的材料选择、检索、写作、插图和 Markdown 清洗逻辑。

同时，系统为章节引入版本管理机制。第一次生成保存为版本 1，每次重新生成会追加版本 2、版本 3 等。UI 中用户可以查看不同版本的正文、证据页和 Chunk ID。最新版本会作为当前 active 内容参与完整报告重建。

### 7 检查与 Human-in-the-loop

系统提供章节检查功能，包括普通文本检查和视觉检查。

普通检查主要基于章节正文、披露要求和证据 Chunk，判断生成内容是否存在遗漏、泛化、证据不足或与披露要求不一致的问题。

视觉检查进一步引入 PDF 证据页截图，使模型能够结合原始页面视觉信息判断图文关系、表格引用和图片标题是否合理。

虽然当前检查功能尚未完全内嵌为 LangGraph 主流程中的人工审核节点，但从交互逻辑上已经具备 Human-in-the-loop 的基础能力：用户可以查看章节、检查问题、重新生成章节并比较不同版本。

### 8 Markdown 标准化与 Word 导出

LLM 输出的 Markdown 经常存在不规范情况，例如：

- 表格列数不一致。
- HTML 图片结构不适合 Word 转换。
- 图片宽度属性显示异常。
- 并排图片使用 HTML table 导致 pypandoc 转换不稳定。

为此，系统实现了 `markdown_sanitizer.py`，用于将 LLM 生成内容转换为较保守、Pandoc 友好的 Markdown 格式。主要处理包括：

- 规范换行和不可见字符。
- 识别并转换 HTML table。
- 将 HTML 图片标签转换为 Markdown 图片语法。
- 清洗图片宽度属性在 UI 中误显示的问题。
- 规范 pipe table，保证表格行列一致。

Word 导出由 `word_export.py` 完成。导出时系统会先生成一份 `.pandoc.md` 标准化中间文件，再调用 pypandoc 转换为 `.docx` 文件。这样可以降低 LLM 输出格式不稳定导致 Word 乱码或表格错乱的概率。

### 9 Web UI 与用户交互

前端使用 Streamlit 实现，主要功能包括：

1. 上传 PDF 文件。
2. 选择快速模式或精准模式。
3. 配置不同阶段使用的 LLM provider。
4. 查看当前生成状态、进度信息和耗时记录。
5. 浏览已生成章节。
6. 查看每个章节的证据页截图。
7. 对章节进行重新生成、普通检查和视觉检查。
8. 查看章节不同版本。
9. 下载 Markdown 报告。
10. 导出 Word 报告。
11. 对已构建向量库的报告进行即时问答。

证据页展示通过 PyMuPDF 将原始 PDF 页渲染为图片，使用户能够直接对照生成内容与原文证据。该设计提高了报告生成过程的透明度，也方便人工审核。

### 10 状态管理与断点续跑

本项目的断点续跑主要依赖 LangGraph 的 checkpointer 机制。系统使用 `SqliteSaver` 将 checkpoint 保存到每个会话目录下：

```text
checkpoints.sqlite
```

当用户点击继续生成时，系统会根据相同的 `thread_id` 从 checkpoint 中恢复 LangGraph 状态，并从中断位置继续执行。

除此之外，系统还保存 `graph_state.json` 作为最近一次工作流状态的 JSON 备份。它并不是 LangGraph 的正式断点机制，而是用于辅助恢复、调试和 API 状态同步。

为了避免状态混乱，系统将状态分为：

- `checkpoints.sqlite` / `graph_state.json`：工作流真实状态。
- `session.json`：UI 展示快照。
- `messages.json`：用户与系统消息记录。

新增的 `session_store.py` 将 `session.json` 和 `messages.json` 的读写统一收敛到 `SessionStore` 中，并通过 `project_graph_state_to_session()` 将 LangGraph 内部状态投影为 UI 可见状态。这样可以减少 UI 显示状态与工作流真实状态不一致的问题。

### 11 报告问答功能

系统支持用户在向量库构建完成后，对当前报告进行 RAG 问答。例如用户可以询问：

```text
女性员工的比例是多少？
```

问答流程如下：

```text
用户问题
  -> Chroma 向量检索 top-k Chunk
  -> 构造证据上下文
  -> LLM 基于证据回答
  -> 返回答案、证据页和 Chunk ID
```

为了保证响应速度，问答功能目前不进行 LLM rerank。系统会要求模型只基于检索证据回答，如果证据不足，则明确说明当前材料中未找到足够信息，而不是编造答案。

### 12 当前不足与改进方向

虽然系统已经实现了 ESG 报告生成 Agent 的主要功能，但距离工业级稳定应用仍有进一步优化空间。

#### 12.1 后台任务队列

当前系统使用 FastAPI `BackgroundTasks` 执行后台任务。该方式适合原型开发，但在长时间任务、并发任务和服务重启场景下稳定性有限。后续可引入 Celery、RQ、Dramatiq 或 arq 等任务队列，将 PDF 解析、报告生成、章节检查和 Word 导出交由独立 worker 执行。

#### 12.2 LLM 调用稳定性

当前 LLM 调用缺少统一的 retry、timeout、限流和 token 统计机制。后续应封装统一的 LLM 调用层，实现：

- 超时控制。
- 指数退避重试。
- API 限流处理。
- 模型名称、耗时和 token 用量记录。
- Prompt 与输出日志追踪。

#### 12.3 MinerU 解析优化

MinerU 对大文件解析耗时不稳定，且失败后缺少精细化恢复能力。后续可增加：

- 分段解析状态记录。
- 每个 PDF part 独立重试。
- 解析结果缓存。
- 失败 part 保留与复用。
- 大文件解析策略评估。

#### 12.4 测试体系

当前系统需要补充自动化测试，尤其是：

- Markdown sanitizer 测试。
- Chunk 切分与页码映射测试。
- LangGraph 状态流测试。
- API smoke test。
- Word 导出格式测试。
- 小样本 PDF 端到端测试。

#### 12.5 多用户与权限管理

当前系统主要面向本地使用，尚未加入用户鉴权、session 权限隔离、上传文件大小限制和 CORS 安全配置。若部署为多用户系统，需要补充认证、授权、文件隔离和访问控制。

### 13 总结

本项目构建了一个面向 ESG 信息披露场景的报告生成 Agent。系统通过 MinerU 对企业可持续发展报告进行解析，通过分层 Chunking 和向量检索实现证据召回，通过 LangGraph 编排材料选择、RAG 检索、LLM rerank、章节写作、图片插入和 Markdown 清洗等步骤，最终生成结构化、可追溯、可导出的 ESG 报告。

项目的核心价值在于将非结构化企业报告转化为面向监管披露框架的结构化内容，并在生成过程中保留证据页、Chunk ID、图片来源和章节版本信息。这不仅提高了 ESG 报告复用和合规检查效率，也为后续构建更完整的 ESG 智能披露系统提供了基础。
