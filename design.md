# Design: Docling 文档解析集成 (PDF_PARSER=docling)

## 0. Context

- Requirement: 引入 [docling](https://github.com/docling-project/docling) 作为可选的文档解析后端，把 PDF/Office 文档转换成结构化内容后再交给 LlamaIndex 处理，替代当前 `pymupdf4llm` 路径在版面复杂文档（表格、多栏、扫描件）上的不足。
- 约束（用户已确认）：
  1. 尽量只用 LlamaIndex 官方组件（`Document`/`Node` 等），不引入自研解析代码，除非确有必要。
  2. DeepSeek 走 LlamaIndex 支持的方式——**现状已满足**，`app/rag.py::_build_llamaindex_llm` 已用官方 `OpenAILike` 包装 DeepSeek 的 OpenAI 兼容 API，本次不改动。
  3. 有疑问先问用户——已在此前对话中确认以下决策点。
  4. 追加需求：docling 处理不到的文件（用户口径：一般是 `.md`/`.json`）的 chunking 方式要升级为 LlamaIndex/docling 支持的 **Layout-Aware** 和 **Semantic Chunking** 切法；node 的父子关系、元数据由 LlamaIndex 的 `NodeParser` 自动处理，实现只需要管"怎么切"。

## 1. 已确认的决策

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 接入方式 | 新增 `PDF_PARSER=docling`，与现有 `default` / `pymupdf4llm` 三选一并列，可随时切换回退 |
| 2 | docling 自身 chunk 策略 | docling 处理的 PDF/Office 文件使用 `DoclingNodeParser` 默认的 `HierarchicalChunker`（按文档结构切分），不暴露 `chunk_size`/`overlap` 等配置项 |
| 3 | 文件范围 | docling 处理 PDF + Office（DOCX/PPTX/XLSX）；docling 不支持的其他文件类型仍走 `SimpleDirectoryReader` 原逻辑兜底，两批节点合并入索引 |
| 4 | 耗时提示 | docling 转换耗时明显长于 pymupdf4llm（版面分析+可选 OCR），在 `/api/index/rebuild` 响应及日志中提示 |
| 5 | 非 docling 文件的 chunk 策略 | 在现有 `CHUNK_MODE` 基础上新增 `layout_aware` 选项（而不是绕开 CHUNK_MODE 单独做扩展名分流），已有的 `sentence`/`semantic`/`sentence_window` 三个值行为不变 |
| 6 | embedding 资源 | 用户已确认本地 embedding 模型资源足够（可用 docker 起），**仅确认资源，本次不改 embedding 相关代码**——`_configure_llama_index` 里现有的本地 `HuggingFaceEmbedding` 加载路径保持不变，`semantic`/`layout_aware` 都复用同一个 `Settings.embed_model` 实例 |

## 2. 依赖变更

新增（均为 LlamaIndex 官方维护包，已核实与当前锁定的 `llama-index-core 0.11.x` 兼容，**不需要**升级任何现有 llama-index 系列包）：

```
llama-index-readers-docling>=0.2.1,<0.3.0
llama-index-node-parser-docling>=0.2.0,<0.3.0
```

这两个包会传递引入 `docling`（含 `torch`、`torchvision`、`docling-ibm-models` 版面分析模型、`easyocr`、`python-docx`、`python-pptx` 等）。已知代价，用户已确认可接受。首次运行会触发模型下载，属预期行为，不额外处理缓存策略。

`docling`/`docling-core` 不直接写进 `requirements.txt`（作为上面两个包的传递依赖自动装入），保持 requirements.txt 只声明直接依赖的现有风格。

## 3. 架构改动

### 3.1 `app/config.py`

```python
# --- PDF parsing ---
# "default"     – SimpleDirectoryReader built-in PDF parsing
# "pymupdf4llm" – convert PDF to Markdown via pymupdf4llm before indexing
# "docling"     – convert PDF/DOCX/PPTX/XLSX via docling (layout-aware,
#                 hierarchical chunking); slower, better on tables/complex layout
pdf_parser: Literal["default", "pymupdf4llm", "docling"] = Field(
    default="default", alias="PDF_PARSER"
)

# --- Chunking ---
# "sentence"        – SentenceSplitter (existing default)
# "semantic"        – SemanticSplitterNodeParser (embedding 语义边界切分)
# "sentence_window" – SentenceWindowNodeParser (逐句切分 + 前后窗口)
# "layout_aware"    – 按文件结构切分：.md 用 MarkdownNodeParser（按标题层级切），
#                      .json 用 JSONNodeParser（按 JSON 结构切），其余扩展名回退
#                      SentenceSplitter；PDF_PARSER=docling 时 PDF/Office 文件不
#                      受此项影响（走 docling 自己的 HierarchicalChunker，见决策 #2）
chunk_mode: Literal["sentence", "semantic", "sentence_window", "layout_aware"] = Field(
    default="sentence", alias="CHUNK_MODE"
)
```

不新增其他配置字段——`layout_aware` 各扩展名对应的 parser 都用 LlamaIndex 默认参数，不做 chunk_size 等可调项（与决策 #2 的"先用默认值"保持一致的极简原则）。

### 3.2 `app/rag.py` — 新增 `_parse_documents_to_nodes`，`rebuild_index()` 改分支

当前结构（简化）：

```python
def rebuild_index(self) -> int:
    if pdf_parser == "pymupdf4llm":
        load_dir = convert_pdfs_to_markdown_temp(upload_dir)   # temp dir
    else:
        load_dir = upload_dir

    documents = SimpleDirectoryReader(load_dir).load_data()
    ...
    node_parser = self._build_node_parser()          # CHUNK_MODE
    nodes = node_parser.get_nodes_from_documents(documents)
    self._index = VectorStoreIndex(nodes, ...)
```

`CHUNK_MODE=layout_aware` 需要按文件扩展名分流到不同 `NodeParser`，而不是像现有三个模式那样对所有 `Document` 统一调用同一个 parser。因此把"documents → nodes"这一步从 `_build_node_parser()` 单实例调用，升级为一个新的分流方法 `_parse_documents_to_nodes()`，供 `default`/`pymupdf4llm`/docling-fallback **三个分支共用**（不止 docling 分支，保证 `layout_aware` 在任何 `PDF_PARSER` 下行为一致）：

```python
def _parse_documents_to_nodes(self, documents: list[Document]) -> list[BaseNode]:
    """把 Document 列表切成 Node 列表，按 CHUNK_MODE 分派。

    layout_aware 按扩展名分流；LlamaIndex 的 NodeParser 自动处理 node 的
    父子关系（relationships）和 metadata 继承，这里只需要选对 parser：
      .md   -> MarkdownNodeParser  （按标题层级切）
      .json -> JSONNodeParser      （按 JSON 结构切）
      其他  -> SentenceSplitter    （安全兜底，等价于 CHUNK_MODE=sentence 默认值）
    其余三个 CHUNK_MODE 值保持原有行为：整批 documents 用同一个 parser。
    """
    if self.settings.chunk_mode != "layout_aware":
        return self._build_node_parser().get_nodes_from_documents(documents)

    from llama_index.core.node_parser import MarkdownNodeParser, JSONNodeParser

    groups: dict[str, list[Document]] = {"md": [], "json": [], "other": []}
    for doc in documents:
        suffix = Path(doc.metadata.get("file_name", "")).suffix.lower()
        key = "md" if suffix == ".md" else "json" if suffix == ".json" else "other"
        groups[key].append(doc)

    nodes: list[BaseNode] = []
    if groups["md"]:
        nodes.extend(MarkdownNodeParser().get_nodes_from_documents(groups["md"]))
    if groups["json"]:
        nodes.extend(JSONNodeParser().get_nodes_from_documents(groups["json"]))
    if groups["other"]:
        nodes.extend(SentenceSplitter().get_nodes_from_documents(groups["other"]))
    return nodes
```

docling 分支仍然不复用 `SimpleDirectoryReader`（因为 docling 直接从原始二进制产出结构化 chunk，中间不经过 `Document.text` 这一层），但按文件类型分流后，docling 处理不到的文件现在统一交给 `_parse_documents_to_nodes()`：

```python
DOCLING_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}

def rebuild_index(self) -> int:
    if self.settings.pdf_parser == "docling":
        nodes = self._build_docling_nodes()          # new helper
        document_count = self._docling_doc_count      # tracked in helper
    elif self.settings.pdf_parser == "pymupdf4llm":
        load_dir = convert_pdfs_to_markdown_temp(upload_dir)
        documents = SimpleDirectoryReader(load_dir).load_data()
        document_count = len(documents)
        nodes = self._parse_documents_to_nodes(documents)   # was: node_parser.get_nodes_from_documents(documents)
    else:
        documents = SimpleDirectoryReader(upload_dir).load_data()
        document_count = len(documents)
        nodes = self._parse_documents_to_nodes(documents)   # was: node_parser.get_nodes_from_documents(documents)

    self.reset_index_storage()
    if not nodes:
        self._index = None
        self._bm25_nodes = None
        return 0
    self._index = VectorStoreIndex(nodes, storage_context=self._storage_context())
    self._bm25_nodes = nodes
    return document_count
```

新增私有方法 `_build_docling_nodes()`（放在 `_build_node_parser` 附近，遵循"私有辅助方法前缀 `_`"的项目约定）：

```python
def _build_docling_nodes(self) -> list[BaseNode]:
    """Docling 分支：PDF/Office 走 docling 结构化解析 + HierarchicalChunker，
    其余文件类型（txt/md/json 等）回退到 SimpleDirectoryReader +
    _parse_documents_to_nodes()（走 CHUNK_MODE，含 layout_aware 分流），
    两批 nodes 合并后一起建索引。
    """
    from llama_index.readers.docling import DoclingReader
    from llama_index.node_parser.docling import DoclingNodeParser

    upload_dir = self.settings.upload_dir
    docling_paths = [p for p in upload_dir.rglob("*") if p.is_file() and p.suffix.lower() in DOCLING_SUFFIXES]
    other_paths = [p for p in upload_dir.rglob("*") if p.is_file() and p.suffix.lower() not in DOCLING_SUFFIXES]

    nodes: list[BaseNode] = []
    doc_count = 0

    if docling_paths:
        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        docling_documents = list(reader.lazy_load_data(file_path=[str(p) for p in docling_paths]))
        doc_count += len(docling_documents)
        nodes.extend(DoclingNodeParser().get_nodes_from_documents(docling_documents))

    if other_paths:
        fallback_documents = SimpleDirectoryReader(input_files=[str(p) for p in other_paths]).load_data()
        doc_count += len(fallback_documents)
        nodes.extend(self._parse_documents_to_nodes(fallback_documents))   # was: self._build_node_parser()...

    self._docling_doc_count = doc_count
    return nodes
```

说明：
- `DoclingReader` 按用户决策 #1 只处理 `upload_dir` 下匹配的 PDF/Office 文件；其余文件交给 `_parse_documents_to_nodes()`，保证行为不回归（比如已上传的 .txt/.md/.json 文件仍然可索引，且现在能吃到 `layout_aware`）。
- `document_count` 的返回口径与现有分支保持一致（"读取了多少个文档"），docling 分支下用 docling 产出的文档数 + fallback 文档数相加。
- `_parse_documents_to_nodes` 里不手写 node 的 relationships/metadata 逻辑——`MarkdownNodeParser`/`JSONNodeParser`/`SentenceSplitter` 都是 LlamaIndex 官方 `NodeParser`，父子关系（`PREV`/`NEXT`/`SOURCE`）和 metadata 继承由基类 `NodeParser.get_nodes_from_documents()` 统一处理，符合用户"你只要管怎么切就行"的要求。
- `DoclingReader` 产出的 `LIDocument` 默认不像 `SimpleDirectoryReader` 那样自动填充 `file_name` metadata；这个点记录在 §4 限制里，code-impl 阶段需要手动在 `extra_info` 里补上，避免引用/来源展示功能回归。

### 3.3 `app/main.py::rebuild_index` 路由

按用户决策 #4，docling 模式下追加耗时提示：

```python
@app.post("/api/index/rebuild", response_model=ActionResponse)
async def rebuild_index(
    settings: Settings = Depends(get_settings),
    rag: RagService = Depends(get_rag_service),
) -> ActionResponse:
    try:
        count = await run_in_threadpool(rag.rebuild_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc
    message = f"索引完成，读取 {count} 个文档"
    if settings.pdf_parser == "docling":
        message += "（docling 解析耗时较长，属预期行为）"
    return ActionResponse(message=message, count=count)
```

（提示放在完成后的消息里，而不是提前发一个"正在处理"的通知——现有 API 是同步阻塞式 REST 调用，没有进度推送机制，符合最小改动原则。）

### 3.4 `app/pdf_utils.py`

不改动。docling 分支不复用这个文件里的 `convert_pdfs_to_markdown_temp`（那是 pymupdf4llm 专用的临时目录转换逻辑），docling 走的是内存中的 `DoclingReader` 直接读取，无需生成临时文件。

## 4. 边界与已知限制（需写进代码注释，避免后续误判为 bug）

1. **`PDF_PARSER=docling` 时，PDF/Office 文件不受 `CHUNK_MODE` 影响**——这些文件走 docling 自己的 `HierarchicalChunker`（决策 #2），`CHUNK_MODE`（含新增的 `layout_aware`）只对 docling 处理不到的文件生效。需要在 `_build_docling_nodes()` 的 docstring 和 `config.py` 的 `PDF_PARSER`/`CHUNK_MODE` 字段注释里都说明，防止未来排查"改了 CHUNK_MODE 怎么没用"时踩坑。
2. **`CHUNK_MODE=layout_aware` 对非 `.md`/`.json` 文件回退到 `SentenceSplitter` 默认参数**——不是"不支持"，只是没有对应的结构化 parser 时用安全兜底，行为等价于 `CHUNK_MODE=sentence`。用户口径里"其他类型文件一般是 md/json"，兜底分支预期命中率低。
3. **`DoclingReader` 不像 `SimpleDirectoryReader` 那样自动填充 `file_name` metadata**——code-impl 时需要在读取后手动补 `extra_info`/`metadata["file_name"]`，否则引用来源展示会回归（现有 `_build_snippet`/`_build_context` 依赖这个字段）。
4. **`reranker`/`hybrid_search`/`query_transform` 等下游检索特性不受影响**——它们操作的是 `VectorStoreIndex` 里的 nodes，不关心 nodes 是怎么切出来的，`layout_aware`/docling 分支产出的 `TextNode` 与其他分支完全同构，无需改动 `_build_retriever`/`_retrieve_nodes`。
5. **首次使用 `PDF_PARSER=docling` 会触发 docling 版面模型下载**，网络受限环境需要预先准备好模型缓存；这不在本次改动范围内处理（不做类似 `EMBED_MODEL_PATH` 那样的本地路径配置），先用默认在线下载，如后续需要离线部署再单独提需求。
6. **性能**：docling 转换比 pymupdf4llm 慢，`rebuild_index()` 仍是同步全量重建（`main.py` 已用 `run_in_threadpool`，不阻塞事件循环，但请求耗时会显著变长）。不在本次改动中处理增量索引问题（那是另一项已识别但未排期的优化点）。
7. **embedding 资源不变**——`layout_aware` 分流本身不需要 embedding（`MarkdownNodeParser`/`JSONNodeParser`/`SentenceSplitter` 都是结构/规则切分，不调用 embed_model）；已有的 `CHUNK_MODE=semantic` 仍复用 `_configure_llama_index` 里加载的本地 `HuggingFaceEmbedding`，本次不改 embedding 相关代码（用户决策 #6）。

## 5. 测试计划

新增 `tests/test_rag_docling_parser.py`，风格对齐 `tests/test_rag_opt_01_04.py`：

- **Config 层**（不依赖 docling 包本身）：
  - `PDF_PARSER=docling` 能被 `Settings` 正确解析
  - `CHUNK_MODE=layout_aware` 能被 `Settings` 正确解析
  - 非法值仍然报错（Literal 校验）
- **`RagService._build_docling_nodes` 逻辑**（mock `DoclingReader`/`DoclingNodeParser`/`SimpleDirectoryReader`，不触发真实模型加载）：
  - 目录里全是 PDF → 只调用 docling 路径，不调用 `SimpleDirectoryReader`
  - 目录里混合 PDF 和 .txt → 两条路径都被调用，nodes 合并
  - 目录为空 → 返回空列表，`rebuild_index()` 返回 0 且不建索引（复用现有空目录处理逻辑）
- **`RagService._parse_documents_to_nodes` 逻辑**（`CHUNK_MODE=layout_aware`，mock 或用真实的 `MarkdownNodeParser`/`JSONNodeParser`/`SentenceSplitter`——这三个都是纯规则切分，不依赖 embedding，可以不 mock）：
  - 传入 `.md` 文档 → 走 `MarkdownNodeParser`，产出的 node 带正确的 metadata（如 `file_name`）
  - 传入 `.json` 文档 → 走 `JSONNodeParser`
  - 传入未知扩展名文档 → 走 `SentenceSplitter` 兜底
  - 混合 `.md` + `.json` + 其他 → 三组分别切分后合并，总数等于各组产出之和
  - `CHUNK_MODE` 为其他三个值（`sentence`/`semantic`/`sentence_window`）时 → 仍走 `_build_node_parser()` 单实例路径（回归验证，不改变现有行为）
- 不新增依赖真实 Postgres/embedding 模型的集成测试（与项目现有测试策略一致，见 `flow-engine/.bootstrap.md` 的"Test command"约定）。

## 6. 不做的事（明确排除，避免范围蔓延）

- 不做 docling 的 OCR/图片专项支持（`export_type=JSON` 已含表格结构，图片默认走 `image_placeholder=""` 占位，不单独处理）
- 不做增量索引（已识别的独立优化项，不在本次范围）
- 不改 DeepSeek 接入方式（已用官方组件，无需改动）
- 不给 docling chunk 暴露 `chunk_size`/`overlap` 等配置项（用户决策 #2：先用默认值）
- 不给 `layout_aware` 的 `MarkdownNodeParser`/`JSONNodeParser`/`SentenceSplitter` 暴露额外配置项（同样先用默认值，保持最小改动）
- 不改 embedding 基础设施（用户决策 #6：本地模型资源已确认足够，代码不动）
- 不处理 BM25 索引持久化等其他既有问题（超出本次范围）

## 7. 影响文件清单

- `requirements.txt`（新增 2 行：docling 相关包）
- `.env.example`（`PDF_PARSER` 注释补充 docling 选项；`CHUNK_MODE` 注释补充 `layout_aware` 选项）
- `app/config.py`（`pdf_parser` 字段 Literal 扩展；`chunk_mode` 字段 Literal 扩展，均加注释）
- `app/rag.py`（新增 `_build_docling_nodes`、`_parse_documents_to_nodes`，`DOCLING_SUFFIXES` 常量，`rebuild_index` 三个分支都改用 `_parse_documents_to_nodes`/`_build_docling_nodes`）
- `app/main.py`（`rebuild_index` 路由注入 `settings` 依赖，耗时提示文案）
- `tests/test_rag_docling_parser.py`（新增，含 docling 分支 + layout_aware 分流两部分用例）
