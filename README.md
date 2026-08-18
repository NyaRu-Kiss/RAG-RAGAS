# LlamaIndex RAG

一个可直接运行的本地 RAG 问答应用：

- 后端：FastAPI
- 前端：单页聊天界面
- LLM：支持 Gemini / DeepSeek，两者都可通过 `.env` 切换
- Embedding：本地 `BAAI/bge-m3`
- 向量库：PostgreSQL + pgvector
- 文档加载：LlamaIndex `SimpleDirectoryReader`

这个项目的目标不是做一个“RAG 示例片段”，而是给你一个已经接好上传、重建索引、问答、引用展示的最小可用产品骨架，后续可以继续往 Agent、多工具、重排、权限控制这些方向扩展。

## 项目能做什么

当前版本已经支持下面这些结果导向的能力：

- 上传本地知识文件到项目目录
- 基于上传文件重建向量索引
- 使用 Gemini 或 DeepSeek 对检索结果做问答
- 使用本地 `bge-m3` 生成向量，不依赖远程 embedding API
- 把向量写入 pgvector，避免每次启动都重新做一套纯内存索引
- 在回答下方展示引用来源，方便核对答案依据
- 显示引用对应的文件名、文件路径、页码、相似度分数、文本片段
- 用网页界面完成上传、重建索引、聊天问答
- 离线运行 `ragas` 评估并落盘报告
- 可选：PDF/Office 走 `pymupdf4llm` 或 `docling` 做版面感知解析（`PDF_PARSER`）
- 可选：语义分块、句子窗口分块、按文件结构分块（`CHUNK_MODE`）
- 可选：BM25 + 向量的混合检索，RRF 融合（`HYBRID_SEARCH_ENABLED`）
- 可选：Multi-Query / HyDE 查询变换（`QUERY_TRANSFORM_MODE`）
- 可选：交叉编码器重排序（`RERANKER_ENABLED`）

这些"可选"能力默认关闭，通过 `.env` 中的开关启用，详见下方[技术细节与策略](#技术细节与策略)。

## 当前适合的使用场景

- 简历问答：上传简历，问候选人的经历、项目、技能
- 内部知识库问答：上传说明文档、制度文档、项目材料
- 小规模资料验证：快速确认回答是否来自原文，而不是模型臆测
- 本地 RAG 骨架：作为后续扩展 Agent、工具调用、重排器的基础版本

## 当前不包含的能力

这些能力目前还没有做，后续可以继续加：

- 多轮会话历史持久化
- 用户体系和权限隔离
- 在线文档预览和页码跳转
- 文档删除、单文档重建、增量更新
- 多知识库管理
- ReAct Agent / 工具调用
- 流式输出
- 生产级部署配置

## 评估系统

项目现在包含一个独立于 Web 请求链路的离线评估子系统，目录在 `eval/`。
评估系统使用独立的 `.env.eval` 配置文件，不与主应用共用 `.env`。

当前接入的指标有：

- `Context Precision`
- `Context Recall`
- `Context Entities Recall`
- `Noise Sensitivity`
- `Response Relevancy`
- `Faithfulness`
- `Multimodal Faithfulness`
- `Multimodal Relevance`

说明：

- 前 6 个指标会在你提供评测数据集后正常参与评分
- 当前 RAG 还没有完整多模态证据链，所以 `Multimodal Faithfulness` 和 `Multimodal Relevance` 会在报告里明确标记为 `skipped`

### 数据集格式

先复制评估配置模板：

```bash
cp .env.eval.example .env.eval
```

评测数据集使用仓库内的 `jsonl` 文件，每行一个样本。默认路径：

```text
eval/datasets/rag_eval_v1.jsonl
```

样本格式：

```json
{
  "id": "sample_001",
  "user_input": "候选人有多少年 .NET 经验？",
  "reference": "6 years",
  "reference_contexts": [
    "6 years .NET software engineer"
  ],
  "tags": ["resume", "factoid"],
  "difficulty": "easy",
  "question_type": "fact",
  "images": [],
  "reference_images": []
}
```

字段说明：

- `reference_contexts` 建议尽量填写，这会提升检索类指标的可解释性
- `images` 和 `reference_images` 是为未来多模态扩展预留的

### 固定离线评测流程

HotpotQA distractor validation 使用固定的 200 条 retrieval 集和其中确定性抽取的 20 条 generation 集。所有 query 共享同一个去重 corpus；不会为单个问题注入其 supporting facts。

```bash
. .venv/bin/activate
python3 -m eval.cli data fetch --dataset hotpotqa-distractor
python3 -m eval.cli data prepare --dataset hotpotqa-distractor --seed 20260812
python3 -m eval.cli index rebuild --dataset hotpotqa-distractor
python3 -m eval.cli run generation --dataset hotpotqa-distractor
python3 -m eval.cli report --run eval/reports/<run_id>
```

准备产物位于 `eval/datasets/hotpotqa-distractor/`，包含共享 `corpus.jsonl`、固定 `retrieval.jsonl`、`generation.jsonl`、`dataset_manifest.json` 和 `index_state.json`。索引只能使用独立的 `eval_hotpotqa_distractor` 表；不会操作主应用的上传目录或向量表。

### 输出结果

每次运行都会在 `eval/reports/<run_id>/` 下生成：

- `manifest.json`
- `summary.json`
- `summary.md`
- `samples.jsonl`
- `failures.json`

评估 judge 默认也支持 provider 切换，配置放在 `.env.eval`：

```env
EVAL_JUDGE_PROVIDER=openai_compatible
EVAL_JUDGE_API_KEY=...
EVAL_JUDGE_BASE_URL=https://third-party.example/v1
EVAL_JUDGE_MODEL=...
EVAL_JUDGE_TEMPERATURE=0
EVAL_PIPELINE_MAX_WORKERS=1
EVAL_MAX_WORKERS=1
```

`EVAL_PIPELINE_MAX_WORKERS` 控制 RAG 生成样本并发数，`EVAL_MAX_WORKERS` 控制 Ragas Judge 并发数。两者默认均为 `1`；建议先将 pipeline 提升到 `2`，确认本地 BGE-M3 内存与 DeepSeek 限流稳定后再逐步增加。

评估 embeddings 直接复用应用当前配置的本地 embedding 模型，不额外调用远端 embedding API。Judge 与应用主链路的 `LLM_PROVIDER` 独立配置；旧的 Gemini/DeepSeek 变量仅为本地已有配置兼容保留。

## 系统结构

请求链路如下：

1. 用户在前端上传文件
2. 文件保存到 `data/uploads/`
3. 点击“重建索引”后，按 `PDF_PARSER` 选择的加载策略读取目录（见下文）
4. 按 `CHUNK_MODE` 选择的分块策略切成 Node
5. 本地 `bge-m3` 生成 embedding
6. 向量写入 PostgreSQL 的 pgvector 表（整表重建，非增量）
7. 用户提问时，按配置组装检索器（向量 / 混合 / 查询变换）从 pgvector 检索候选片段
8. 如果启用了 reranker，交叉编码器对候选片段重新打分排序
9. 检索结果（完整文本，非摘要）拼进 Prompt，交给当前启用的 LLM 生成最终回答
10. 返回答案和引用信息（含摘要片段、页码、相似度分数）给前端展示

整条链路的每一步都尽量直接用 LlamaIndex 官方组件实现（`SimpleDirectoryReader`、各类 `NodeParser`、`VectorStoreIndex`、`PGVectorStore`、`QueryFusionRetriever`、`SentenceTransformerRerank`、`HyDEQueryTransform`），只有向量库落地存储这一层换成自管的 PostgreSQL + pgvector，而不是 LlamaIndex 自带的内存/本地向量库。

## 技术细节与策略

以下内容对应 `app/rag.py`、`app/config.py`、`app/pdf_utils.py` 的实际实现，不是设计草案。所有策略都通过 `.env` 里的开关切换，默认值保持当前"最小可用"配置，开启新策略后需要重新点一次"重建索引"。

### 1. 文档加载策略（`PDF_PARSER`）

由 `RagService.rebuild_index()`（`app/rag.py`）按 `PDF_PARSER` 三选一分派：

| 取值 | 实现方式 | 说明 |
|---|---|---|
| `default`（默认） | `SimpleDirectoryReader(input_dir=..., recursive=True).load_data()` | LlamaIndex 官方默认加载器，PDF 用其内置解析，速度快，复杂版面（多栏、表格）效果一般 |
| `pymupdf4llm` | `app/pdf_utils.py::convert_pdfs_to_markdown_temp()` 先把每个 PDF 转成 Markdown，写到临时目录；非 PDF 文件用符号链接原样接入；再交给 `SimpleDirectoryReader` 统一读取临时目录 | 转换后文本更干净，索引完成后临时目录会被 `shutil.rmtree` 清理 |
| `docling` | `RagService._build_docling_nodes()`：`.pdf`/`.docx`/`.pptx`/`.xlsx` 用 LlamaIndex 官方 `llama-index-readers-docling` 的 `DoclingReader`（`export_type=JSON`）逐文件加载（避免其 `extra_info` 在多文件场景下被覆盖的问题），其余扩展名仍走 `SimpleDirectoryReader` 兜底 | 版面分析 + 可选 OCR，对表格、多栏、扫描件效果最好，但耗时明显更长，首次运行会下载 docling 的版面分析模型 |

三种模式下，非 PDF/Office 文件的加载都统一走 `SimpleDirectoryReader`。它是 LlamaIndex 官方的通用目录加载器，默认显式支持的类型见下方[支持的文件类型](#支持的文件类型)一节；不在列表里的纯文本文件通常也能按文本读取。

`rebuild_index()` 每次都是"整库重建"：先调用 `reset_index_storage()` 删掉 pgvector 的数据表，再重新扫描 `data/uploads/` 整个目录、重新生成全部向量。这是当前版本的设计选择（简单、可控、便于调试），不支持增量更新或单文档重建。

### 2. 分块（Chunking）策略（`CHUNK_MODE`）

由 `RagService._build_node_parser()` / `_parse_documents_to_nodes()` 按 `CHUNK_MODE` 四选一实现：

| 取值 | 使用的 LlamaIndex 组件 | 行为 |
|---|---|---|
| `sentence`（默认） | `SentenceSplitter()` | 按句子边界切分，官方默认参数 `chunk_size=1024`、`chunk_overlap=200`（token 数），本项目未覆盖这两个参数 |
| `semantic` | `SemanticSplitterNodeParser` | 基于 embedding 的语义相似度动态确定分块边界，而非固定长度；参数 `SEMANTIC_BUFFER_SIZE`（默认 `1`，每次比较时前后各扩展的句子数）、`SEMANTIC_BREAKPOINT_THRESHOLD`（默认 `95.0`，语义断点的百分位阈值，越高切得越少） |
| `sentence_window` | `SentenceWindowNodeParser.from_defaults()` | 按单句切分成小 Node 存入索引，但每个 Node 的 metadata 里保留 `window`（前后 `SENTENCE_WINDOW_SIZE` 句，默认 `3`）和 `original_text`；检索定位到单句、生成时可用完整窗口，兼顾检索精度与上下文完整性 |
| `layout_aware` | `HierarchicalNodeParser.from_defaults(chunk_sizes=HIERARCHICAL_CHUNK_SIZES)` | 真正的三层父子结构分块（root → mid → leaf，字符数从粗到细，默认 `[2048, 512, 128]`），见下方专门说明 |

`layout_aware` 与其余三种模式不同：其余三种是"扁平"分块，每个 `Document` 切出一批同层级的 Node；`layout_aware` 对每个 `Document` 递归切出一棵三层树，父子关系（`NodeRelationship.PARENT`/`CHILD`）由 `HierarchicalNodeParser` 自动建立。

**`layout_aware` 的检索行为**：`_parse_documents_to_nodes()` 返回 `(leaf_nodes, all_nodes)`。只有 `leaf_nodes`（最细粒度）会被 embedding、写入 pgvector、参与 BM25 索引；`all_nodes`（三层全部节点）整体写入 Postgres 落地的 docstore（`DOCSTORE_TABLE`，默认 `rag_docstore`，经由 `PostgresDocumentStore`）。检索时 `_build_retriever()` 会用 LlamaIndex 官方 `AutoMergingRetriever` 包一层：命中的 leaf Node 若同一父节点下的兄弟节点命中比例够高（`simple_ratio_thresh`，默认阈值），就从 docstore 按 `parent_id` 取出父节点、合并替换掉那些零散的 leaf 结果，从而在保留细粒度检索精度的同时，让上下文能自动"长回"更完整的父级片段。即使很短的文档也一定会切出三层（每层至少一个节点），保证父子链路总是完整的。

**`PDF_PARSER=docling` 时的例外**：docling 处理的 PDF/Office 文件不受 `CHUNK_MODE` 影响，而是固定使用 docling 自带的 `DoclingNodeParser`（默认 `HierarchicalChunker`，按文档结构切分，不暴露 `chunk_size`/`overlap` 等配置项），产出的 Node 本身已经足够细，不再额外生成父层，直接作为 leaf 处理（`leaf_nodes == all_nodes`）。因为 `DoclingNodeParser` 会用 docling 自己的 chunk metadata（如 `heading`）整体替换 `node.metadata`，代码里额外从 `node.relationships[NodeRelationship.SOURCE].metadata` 用 `setdefault` 把 `file_name`/`file_path` 补回去，保证引用展示不受影响。docling 无法处理的其余文件类型仍然受 `CHUNK_MODE`（含 `layout_aware`）控制，两批 Node 合并进同一批 leaf/all 结果。

### 3. Embedding 策略

- 固定使用本地 `HuggingFaceEmbedding`（`llama-index-embeddings-huggingface`）加载 `BAAI/bge-m3`，向量维度 `EMBED_DIM = 1024`（写死在 `app/rag.py`，需与 `PGVectorStore` 建表时的 `embed_dim` 保持一致）
- 优先尝试本地模型目录 `EMBED_MODEL_PATH`；路径不存在则回退到 `EMBED_MODEL_NAME`（默认 `BAAI/bge-m3`），由 Hugging Face 库按模型名解析（可能触发下载）
- 全程不依赖远程 embedding API；`semantic` 分块和评估系统的 embedding 也复用同一个 `LlamaSettings.embed_model` 实例，不额外走网络请求

### 4. 向量存储：pgvector（而非 LlamaIndex 内置向量库）

这是本项目对"尽量用 LlamaIndex 官方组件"原则的一处主动例外：向量持久化没有用 LlamaIndex 自带的内存/本地向量索引，而是用自管的 PostgreSQL + pgvector 扩展，访问层仍然是 LlamaIndex 官方的 `PGVectorStore`（`llama-index-vector-stores-postgres`）：

- `RagService._vector_store()` 用 `PGVectorStore.from_params(..., perform_setup=True, use_jsonb=True)` 自动建表/建索引
- `_ensure_index()` 用 `VectorStoreIndex.from_vector_store()` 在服务重启后无需重新 embedding 即可复用已有向量
- 重建索引时 `reset_index_storage()` 直接 `DROP TABLE IF EXISTS "data_<PG_TABLE>"`（`PGVectorStore` 实际建表名会加 `data_` 前缀），保证每次重建都是干净状态；同时也会 `DROP TABLE IF EXISTS "data_<DOCSTORE_TABLE>"`，清掉 `layout_aware` 模式下落地的 Postgres docstore（`PostgresDocumentStore`，同样是 `data_` 前缀建表规则）
- 选择自管 pgvector 而非托管向量数据库，是为了让向量数据留在本地、可用 SQL 直接检查，并为后续 HNSW/IVFFlat 等索引优化留出空间（当前 `docker/postgres/init.sql` 只启用了 `vector` 扩展，未建 ANN 索引，检索是精确 KNN）

### 5. 检索策略

`RagService._build_retriever()` 组装检索器，`_retrieve_nodes()` 按 `QUERY_TRANSFORM_MODE` 包一层查询变换：

- **基础检索**：`index.as_retriever(similarity_top_k=fetch_k)`，`fetch_k` 由 `_fetch_k()` 决定 —— 未启用 reranker 时等于 `TOP_K`（默认 `5`），启用 reranker 时先按更大的 `RETRIEVAL_TOP_K`（默认 `20`）召回，再由 reranker 精选到 `TOP_K`
- **混合检索**（`HYBRID_SEARCH_ENABLED=true`）：用 LlamaIndex 官方 `BM25Retriever`（内存态，构建自最近一次 `rebuild_index()` 产出的全部 leaf Node）与向量检索器一起交给 `QueryFusionRetriever`，`mode="reciprocal_rerank"`（RRF 融合），`num_queries=1` 表示只用原始 query、不做多查询改写；重建索引后 BM25 索引才会更新，代码里注释也标注了这一点。BM25/混合检索只作用于 leaf Node，不感知 `layout_aware` 的父子层级
- **自动合并**（`CHUNK_MODE=layout_aware` 时固定启用）：上面组装出的检索器（向量或混合）最外层再包一层 LlamaIndex 官方 `AutoMergingRetriever`，命中的 leaf Node 若同一父节点下命中的兄弟节点比例够高，会自动从 Postgres docstore 取出父节点合并替换，详见上方[分块策略](#2-分块chunking策略chunk_mode)一节
- **查询变换**（`QUERY_TRANSFORM_MODE`）：
  - `none`（默认）：直接用原始问题检索
  - `multi_query`：把（可能是混合检索的）retriever 再包一层 `QueryFusionRetriever`，让 LLM 生成 `NUM_QUERIES - 1`（默认共 `4` 个）个改写查询，与原始查询一起检索后 RRF 融合去重
  - `hyde`：用 LlamaIndex 官方 `HyDEQueryTransform(include_original=True)` 先让 LLM 生成一段假设性答案，用它的 embedding 去检索，同时保留原始查询的检索结果一起返回
  - 只要 `QUERY_TRANSFORM_MODE != "none"`，`LlamaSettings.llm` 才会指向真实的 LLM（`OpenAILike` 包装的 Gemini/DeepSeek）；否则用 `MockLLM()` 占位，避免不需要时的额外 API 调用和更慢的启动

### 6. 重排序（Reranking）

`RERANKER_ENABLED=true` 时，`_build_reranker()` 用 LlamaIndex 官方 `SentenceTransformerRerank`（`llama-index-postprocessor-sentence-transformer-rerank`）加载交叉编码器模型（默认 `BAAI/bge-reranker-v2-m3`，同样优先本地路径 `RERANKER_MODEL_PATH`），对检索到的 `RETRIEVAL_TOP_K` 个候选重新打分，取 `top_n=TOP_K`。执行顺序是：`_retrieve_nodes()` 完成（混合检索 + 查询变换，二者是同一步内的检索器组装）→ reranker 对结果重新排序精选 → 拼装上下文。

### 7. 上下文拼装与生成

- `_build_context()`：把最终选定的每个 Node 的**完整原文**（`node.get_content()`，只做空白规整，不截断）拼进 Prompt，并标注 `[序号] 文件名 (page X)`，供 LLM 引用编号使用
- `_build_snippet()`：另外生成一个截断到 `SNIPPET_LENGTH=220` 字符的短摘要，仅用于前端引用展示（UI），不会影响 LLM 实际看到的上下文——这两条路径在代码里是分开的，避免早期版本"LLM 上下文被摘要截断"的问题
- 生成阶段按 `LLM_PROVIDER` 走 Gemini（`google-genai` 官方客户端）或 DeepSeek（`openai` 客户端 + DeepSeek 的 OpenAI 兼容 API），两者的 System Prompt 都来自 `SYSTEM_PROMPT` 配置项
- 返回给前端的引用信息（`Citation`）和评估系统用的检索上下文（`RetrievedContext`）在同一次 `evaluate_query()` 调用里一起构建，避免重复检索

## 已实现功能

### 1. 文档上传

- 接口：`POST /api/documents/upload`
- 前端支持多文件上传
- 文件会直接落盘到 `data/uploads/`
- 当前行为是保留上传后的文件，后续重建索引会读取整个上传目录

### 2. 索引重建

- 接口：`POST /api/index/rebuild`
- 会重新扫描 `data/uploads/`
- 会清空当前 pgvector 表并重建索引
- 适合当前这个“先最小可用”的版本

### 3. RAG 问答

- 接口：`POST /api/chat`
- 输入：`{"message": "你的问题"}`
- 流程：检索 -> 交给当前启用的 LLM 生成回答 -> 返回答案和引用

### 4. 引用展示

每条回答下方会展示引用列表，当前包含：

- `file_name`：来源文件名
- `file_path`：来源文件路径
- `page_label`：页码，适用于 PDF 等有分页信息的文档
- `score`：检索相似度分数
- `snippet`：命中的文本片段

## 支持的文件类型

项目使用的是 LlamaIndex 官方 `SimpleDirectoryReader`。根据官方文档，它默认显式支持这些类型：

- `.csv`
- `.docx`
- `.epub`
- `.hwp`
- `.ipynb`
- `.jpeg`, `.jpg`
- `.mbox`
- `.md`
- `.mp3`, `.mp4`
- `.pdf`
- `.png`
- `.ppt`, `.pptm`, `.pptx`

除此之外，`SimpleDirectoryReader` 也会尝试把普通文本文件按文本读取，所以常见的纯文本文件通常也能用。更完整说明见官方文档：

- https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/

## 目录结构

```text
LlamaindexRAG/
├── .env.example         # 主应用环境变量模板
├── .env.eval.example    # 离线评测环境变量模板
├── app/
│   ├── main.py          # FastAPI 入口和 API 路由
│   ├── rag.py           # LlamaIndex / Gemini / DeepSeek / pgvector 组装逻辑
│   ├── schemas.py       # 请求和响应模型
│   ├── config.py        # 环境变量配置
│   └── static/          # 前端页面、样式、脚本
├── data/
│   └── uploads/         # 上传文档目录
├── docker/
│   └── postgres/
│       └── init.sql     # pgvector 扩展初始化
├── eval/                # 独立的离线 RAG 评测系统
│   ├── cli.py           # `python3 -m eval.cli` 命令入口
│   ├── config.py        # `.env.eval` 配置读取与校验
│   ├── dataset.py       # 数据集加载与校验
│   ├── prepare.py       # HotpotQA 数据准备
│   ├── runner.py        # RAGas 评测运行器
│   ├── datasets/        # 准备后的共享语料、generation 集与 manifest
│   └── reports/         # 每次评测生成的报告与样本结果
├── docker-compose.yml   # 本地 pgvector 服务
├── requirements.txt
└── tests/               # 应用和评测系统测试
```

## 环境要求

- Linux / macOS 优先
- `python3` 可用
- Docker 和 Docker Compose 可用，用于启动 PostgreSQL 17 + pgvector
- PostgreSQL 必须安装 `vector` 扩展；项目自带的 Docker 服务会自动完成这一步
- 本机具备本地 `BAAI/bge-m3` 模型目录，或者允许后续自行下载
- 可访问所选的生成模型 API；运行评测时还需要可访问 Ragas Judge API

当前 Docker 配置默认使用：

- Python `3.12.3`
- PostgreSQL 17 + pgvector，宿主机端口映射为 `5434`

之所以用 `5434`，是因为开发过程中发现本机 `5432` 已被占用，避免和你现有数据库冲突。

## 配置说明

主应用复制配置模板：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```env
LLM_PROVIDER=gemini

GEMINI_API_KEY=your-key
GOOGLE_GEMINI_BASE_URL=https://api.aicodemirror.com/api/gemini
GEMINI_MODEL=gemini-3-flash-preview

DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash

EMBED_MODEL_NAME=BAAI/bge-m3
EMBED_MODEL_PATH=/home/tony/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181

PG_HOST=127.0.0.1
PG_PORT=5434
PG_DATABASE=llama_rag
PG_USER=postgres
PG_PASSWORD=postgres
PG_TABLE=rag_documents

UPLOAD_DIR=/home/tony/Code/LLM_App/LlamaindexRAG/data/uploads
TOP_K=5
SYSTEM_PROMPT=You are a helpful RAG assistant. Use the retrieved context when it is relevant, and say when the answer is not grounded in the uploaded documents.
```

重点说明：

- `LLM_PROVIDER`：可选 `gemini` 或 `deepseek`
- `GEMINI_API_KEY`：当 `LLM_PROVIDER=gemini` 时必须填写
- `GOOGLE_GEMINI_BASE_URL`：Gemini 网关地址，当前默认走 `https://api.aicodemirror.com/api/gemini`
- `DEEPSEEK_API_KEY`：当 `LLM_PROVIDER=deepseek` 时必须填写
- `DEEPSEEK_BASE_URL`：DeepSeek API 地址，默认 `https://api.deepseek.com/v1`
- `EMBED_MODEL_PATH`：优先使用本地模型目录
- `TOP_K`：每次检索时召回的候选片段数

以上是最小可用配置。`PDF_PARSER`、`CHUNK_MODE`、`RERANKER_ENABLED`、`HYBRID_SEARCH_ENABLED`、`QUERY_TRANSFORM_MODE` 等策略开关默认关闭，具体取值和行为见上方[技术细节与策略](#技术细节与策略)一节，字段定义见 `app/config.py`，可选项注释见 `.env.example`。

评测系统使用单独的 `.env.eval` 文件，避免 Judge 模型配置与主应用生成模型配置相互影响：

```bash
cp .env.eval.example .env.eval
```

`.env.eval` 至少需要配置一种 Judge。默认的 OpenAI 兼容模式需要以下变量：

```env
EVAL_JUDGE_PROVIDER=openai_compatible
EVAL_JUDGE_API_KEY=your-judge-api-key
EVAL_JUDGE_BASE_URL=https://third-party.example/v1
EVAL_JUDGE_MODEL=judge-model-name
```

也可将 `EVAL_JUDGE_PROVIDER` 设为 `gemini` 或 `deepseek`，并在 `.env.eval` 中填写对应的 `GEMINI_*` 或 `DEEPSEEK_*` 变量。评测 RAG 本身仍会读取 `.env` 的 PostgreSQL、embedding 和主链路生成模型配置，因此运行评测前必须同时完成 `.env` 与 `.env.eval` 配置。

## 安装步骤

### 1. 创建虚拟环境

```bash
python3 -m venv .venv
. .venv/bin/activate
```

### 2. 安装 PyTorch CPU 版本

项目默认按 CPU 版本安装 `torch`，避免无意中拉取超大的 CUDA 包：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
cp .env.eval.example .env.eval
```

然后根据 `LLM_PROVIDER` 填入主应用 API Key，并在 `.env.eval` 填入评测 Judge API Key。仅使用网页应用时，`.env.eval` 可以暂不配置；执行任何 `eval` 命令前必须完成两份配置。

## 启动方式

### 1. 启动 pgvector

```bash
docker compose up -d
```

### 2. 启动 FastAPI

```bash
. .venv/bin/activate
uvicorn app.main:app --reload
```

### 3. 打开页面

浏览器访问：

```text
http://127.0.0.1:8000
```

### 4. 运行离线评测

评测不需要启动 FastAPI，但必须先启动 pgvector，并激活已安装依赖的虚拟环境。首次运行按以下顺序准备固定的 HotpotQA 数据集、建立隔离的评测索引、运行 generation 评测并生成 Markdown 报告：

```bash
. .venv/bin/activate
python3 -m eval.cli data fetch --dataset hotpotqa-distractor
python3 -m eval.cli data prepare --dataset hotpotqa-distractor --seed 20260812
python3 -m eval.cli index rebuild --dataset hotpotqa-distractor
python3 -m eval.cli run generation --dataset hotpotqa-distractor
python3 -m eval.cli report --run eval/reports/<run_id>
```

其中 `fetch` 下载原始数据到 `data/eval_raw/`，`prepare` 生成共享 corpus、200 条 retrieval 集和确定性抽取的 20 条 generation 集，`index rebuild` 写入独立的 `eval_hotpotqa_distractor` 表。评测报告保存在 `eval/reports/<run_id>/`；该流程不会修改主应用上传目录或主向量表。

## 使用流程

第一次使用建议按这个顺序：

1. 启动数据库和应用
2. 打开网页
3. 点击“上传文档”
4. 选择一个或多个文件
5. 上传完成后点击“重建索引”
6. 等索引完成
7. 在聊天框里提问
8. 查看回答下方的引用信息

如果你后来又上传了新文件，记得再次点击“重建索引”，否则新文件不会进入当前索引。

## API 说明

### `GET /health`

健康检查。

返回示例：

```json
{
  "status": "ok"
}
```

### `POST /api/documents/upload`

上传一个或多个文件。

返回示例：

```json
{
  "message": "已上传 2 个文件",
  "count": 2
}
```

### `POST /api/index/rebuild`

重建整个上传目录的索引。

返回示例：

```json
{
  "message": "索引完成，读取 12 个文档",
  "count": 12
}
```

### `POST /api/chat`

请求体：

```json
{
  "message": "Liu Guangzhi 最近一份工作是什么？"
}
```

返回体示例：

```json
{
  "answer": "Liu Guangzhi most recently worked as a Full Stack Developer at China Post Construction Technology Co., Ltd.",
  "citations": [
    {
      "file_name": "2_Liu Guangzhi-6 years-.NET software engineer.pdf",
      "file_path": "/home/tony/Code/LLM_App/LlamaindexRAG/data/uploads/2_Liu Guangzhi-6 years-.NET software engineer.pdf",
      "page_label": "2",
      "score": 0.4884,
      "snippet": "2023.09-2025.1 China Post Construction Technology Co., Ltd. Full Stack Developer (.NET/Java)..."
    }
  ]
}
```

## 常见问题

### 1. 上传完文件后为什么问不到新内容？

因为当前实现不是增量索引。你上传新文件后，需要点击一次“重建索引”。

### 2. 为什么引用里会有多个同一个文件？

因为命中的可能是同一个文件里的不同页、不同片段。当前是按命中片段返回，不是按文件去重合并。

### 3. 为什么回答有时比较泛？

可能原因：

- 检索召回不够准（可尝试开启 `HYBRID_SEARCH_ENABLED` 或 `QUERY_TRANSFORM_MODE`）
- 文档内容本身结构不清晰（可尝试 `CHUNK_MODE=semantic` 或 `PDF_PARSER=docling`）
- `TOP_K` 太小或太大
- 未开启 reranker（`RERANKER_ENABLED=true` 通常能明显提升精度）

### 4. 为什么索引重建会覆盖旧索引？

这是当前版本的设计选择：简单、可控、便于调试。现在是“整库重建”，不是“增量更新”。

### 5. 如果本地模型路径不存在怎么办？

当前代码会优先尝试 `EMBED_MODEL_PATH`，如果不存在，会退回 `EMBED_MODEL_NAME=BAAI/bge-m3`。如果本地和远程都不可用，embedding 初始化会失败。

## 停止服务

停止 FastAPI：

```bash
Ctrl+C
```

停止 pgvector：

```bash
docker compose down
```

## 下一步适合扩展什么

如果你继续往 Agent 方向做，比较自然的下一步是：

1. 增加会话历史
2. 增加流式输出
3. 增加文档删除和增量索引
4. 把当前 RAG query engine 包成 Agent 的一个工具
5. 再接网页搜索、SQL 查询、知识库路由等工具

## 当前状态总结

这个项目现在已经是一个能直接使用的基础版 RAG Web 应用，而不是单纯的脚手架。你可以：

- 上传资料
- 重建向量索引
- 发起问答
- 查看引用依据
- 继续在这套结构上往 Agent 演进
