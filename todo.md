# RAG 优化 TODO

> 基于当前代码（`app/rag.py`、`app/config.py`、`eval/`）的分析，整理以下可优化方向。
> 每项标注 **优先级**（🔴高 / 🟡中 / 🟢低）和 **预期收益**。

---

## 一、检索质量

### ✅ ~~1. 添加 Reranker（交叉编码器重排）~~

**现状**：检索只依赖向量相似度 `similarity_top_k=5`，无任何重排步骤。

**方案**：
- 在 `retriever.retrieve()` 之后接入 cross-encoder 重排，如：
  - `BAAI/bge-reranker-v2-m3`（与现有 bge-m3 embedding 配套）
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`（轻量）
- LlamaIndex 提供 `SentenceTransformerRerank` 节点后处理器，接入方式简单
- 建议先用大 `top_k=20` 召回，再 reranker 精选 `top_k=5`

**预期收益**：context_precision 和 faithfulness 指标显著提升。

---

### ✅ ~~2. 修复 SNIPPET_LENGTH 截断问题~~

**现状**：`_build_context()` 调用 `_build_snippet()` 将每个 chunk 截断至 **220 字符**，LLM 实际看到的上下文极短，严重影响答案质量。

**方案**：
- `_build_context()` 直接使用 `node.node.get_content()` 全文（无截断）
- `_build_snippet()` 仅用于前端展示 citation 预览，保留现有逻辑
- 将两个路径分开：`context_for_llm` vs `snippet_for_ui`

**预期收益**：这是当前最大的隐性 bug，修复后回答准确度会有明显跳升。

---

### ✅ ~~3. 混合检索（Hybrid Search）~~

**现状**：纯向量检索，无关键词/稀疏检索。

**方案**：
- 在 pgvector 中启用 BM25 / tsvector 全文索引，与向量检索结果用 **RRF（Reciprocal Rank Fusion）** 融合
- LlamaIndex 的 `QueryFusionRetriever` 可直接编排多路 retriever

**预期收益**：对专有名词、数字、精确短语的召回率大幅提升。

---

### ✅ ~~4. 查询变换（Query Transformation）~~

**现状**：用户问题直接送入 embedding，无任何变换。

**方案（任选其一或组合）**：
- **HyDE**（Hypothetical Document Embeddings）：让 LLM 先生成一段假设答案，再用该答案 embedding 检索
- **Multi-Query**：用 LLM 将一个问题改写为 3-5 个角度的子问题并行检索，再去重融合
- **Step-Back Prompting**：先抽象为更宽泛问题检索背景知识，再检索具体问题

**预期收益**：对模糊、复杂、多跳问题（如 HotpotQA）的召回率提升。

---

## 二、Chunking 策略

### ✅ ~~5. 改用语义分块（Semantic Chunking）~~

**已实现**：
- `app/config.py` 新增 `CHUNK_MODE=semantic`、`SEMANTIC_BUFFER_SIZE`、`SEMANTIC_BREAKPOINT_THRESHOLD`
- `app/rag.py` 的 `_build_node_parser()` 已接入 `SemanticSplitterNodeParser`
- `rebuild_index()` 已按配置切换分块器重建索引

**方案**：
- `SemanticSplitterNodeParser`：基于句子嵌入相似度动态确定分块边界，语义完整性更强
- 调整参数：`buffer_size=1`、`breakpoint_percentile_threshold=95`

**预期收益**：减少语义跨块分割，提升 context_precision。

---

### 🟡 6. 分层检索 / 父子 Chunk（Hierarchical Chunking）

**现状**：只有单一粒度的 chunk，粗细无法兼顾。

**方案**：
- **SentenceWindowNodeParser**：存储小 chunk（1-3 句），检索命中后扩展为周围窗口的完整段落送入 LLM
- **AutoMergingRetriever / 父子 Chunk**：小 chunk 用于检索，命中阈值后自动合并返回父节点（更完整的上下文）

**预期收益**：检索精准度与 LLM 上下文完整性同时兼顾。

---

### ✅ ~~7. 针对 PDF 的专项解析~~

**已实现**：
- `app/config.py` 新增 `PDF_PARSER` 配置，支持 `default` / `pymupdf4llm`
- `app/pdf_utils.py` 新增 PDF 预处理逻辑，可先将 PDF 转为 Markdown
- `app/rag.py` 的 `rebuild_index()` 已支持在索引前走 `pymupdf4llm` 预处理流程，并在完成后清理临时目录

**方案**：
- 接入 `llama-parse`（LlamaIndex 官方云服务，支持表格、图文混排）
- 或使用 `pdfplumber` / `pymupdf4llm` 做预处理，输出结构化 Markdown 再送入 chunker
- 过滤页眉/页脚噪声节点

**预期收益**：文档（如 `buyhouse.pdf`）的解析质量提升，减少噪声 chunk。

---

## 三、生成质量

### 🟡 8. 优化 Prompt 模板

**现状**：`_generate_answer()` 中 prompt 较简单，未引导 LLM 引用来源编号。

**方案**：
- 要求 LLM 在回答中用 `[1]`、`[2]` 等标注引用来源（与 citation 联动）
- 添加链式思考（CoT）指引，尤其对复杂多跳问题
- 针对"无法回答"情形给出更明确的拒绝模板

---

### 🟡 9. 对话记忆（Conversation History）

**现状**：每次 `/api/chat` 请求独立处理，无上下文记忆，多轮对话中代词消解失效。

**方案**：
- 在 `RagService` 中维护会话级 `chat_history: list[dict]`
- 检索前先用 LLM 将多轮上下文压缩为独立问题（Query Condensation）
- 或直接使用 LlamaIndex 的 `CondensePlusContextChatEngine`

---

## 四、评估体系

### 🟡 10. 补充评估指标

**现状**：`eval/metrics.py` 只有 3 个指标（context_precision、response_relevancy、faithfulness），且 multimodal 指标被跳过。

**方案**：
- 添加 `ContextRecall`（需要 reference_contexts，当前数据集已有此字段）
- 添加 `AnswerCorrectness`（与 reference 答案对比）
- 添加 `AnswerSimilarity`
- 考虑 `NoiseSensitivity`（对噪声 chunk 的鲁棒性）

---

### 🟢 11. 评估并发加速

**现状**：`eval_max_workers=1`，评估串行执行，速度很慢。

**方案**：
- 提升 `EVAL_MAX_WORKERS` 至 4-8（受 API 并发限制）
- `run_evaluation` 中 RAG 推理阶段也可改为线程池并发

---

## 五、工程与性能

### 🟡 12. 语义缓存（Semantic Cache）

**现状**：相同或相似问题每次都重新检索+生成，无任何缓存。

**方案**：
- 使用 `GPTCache` 或 LlamaIndex 内置缓存，对语义相似的查询命中缓存
- 简单方案：将 `(query_embedding, answer)` 对存入 Redis，新查询先做向量相似度检查

---

### 🟡 13. 异步检索与生成解耦

**现状**：`RagService.chat()` 是同步阻塞，通过 `run_in_threadpool` 包装。

**方案**：
- 将 `evaluate_query` 改为 `async def`，embedding 推理和 LLM 生成可用 `asyncio` 并发
- 支持 **Streaming** 输出（SSE），提升前端感知响应速度

---

### 🟢 14. 向量索引优化

**现状**：pgvector 默认使用精确 KNN（无 ANN 索引）。

**方案**：
- 在 `docker/postgres/init.sql` 中为向量列创建 `HNSW` 或 `IVFFlat` 索引
- 适当调整 `ef_search` / `probes` 参数，在速度与精度间平衡
- 大规模数据时效果更显著

---

### 🟢 15. 健壮性与可观测性

**现状**：缺少结构化日志、指标暴露和健康检查细节。

**方案**：
- 在 `RagService` 方法中添加耗时打点（检索耗时、生成耗时分开记录）
- 暴露 `/metrics` Prometheus 端点（FastAPI + `prometheus-fastapi-instrumentator`）
- `rebuild_index` 加进度回调，支持大文件集的增量索引

---

## 优先级汇总

| # | 项目 | 优先级 | 难度 |
|---|------|--------|------|
| ~~2~~ | ~~修复 SNIPPET 截断 bug~~ ✅ | 🔴 | 低 |
| ~~1~~ | ~~添加 Reranker~~ ✅ | 🔴 | 中 |
| ~~5~~ | ~~语义分块~~ ✅ | 🔴 | 中 |
| ~~3~~ | ~~混合检索~~ ✅ | 🟡 | 高 |
| ~~4~~ | ~~查询变换（Multi-Query/HyDE）~~ ✅ | 🟡 | 中 |
| 6 | 分层 Chunk | 🟡 | 中 |
| 8 | Prompt 优化 + 引用标注 | 🟡 | 低 |
| 9 | 对话记忆 | 🟡 | 中 |
| 10 | 补充评估指标 | 🟡 | 低 |
| 12 | 语义缓存 | 🟡 | 中 |
| 13 | 异步 + Streaming | 🟡 | 中 |
| ~~7~~ | ~~PDF 专项解析~~ ✅ | 🟢 | 中 |
| 11 | 评估并发加速 | 🟢 | 低 |
| 14 | HNSW 向量索引 | 🟢 | 低 |
| 15 | 可观测性 | 🟢 | 低 |
