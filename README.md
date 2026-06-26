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
- reranker 重排
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

### 运行方式

安装依赖后可直接执行：

```bash
. .venv/bin/activate
python3 -m eval.cli run --dataset eval/datasets/rag_eval_v1.jsonl
```

也支持限制标签或样本数：

```bash
python3 -m eval.cli run --dataset eval/datasets/rag_eval_v1.jsonl --tag resume --limit 5
```

如果你想先用公开数据集做一个小规模基线，可以先导出 HotpotQA 的前 50 条：

```bash
. .venv/bin/activate
python3 -m eval.hotpotqa --output eval/datasets/hotpotqa_50.jsonl --limit 50
python3 -m eval.cli run --dataset eval/datasets/hotpotqa_50.jsonl --limit 50
```

默认使用：

- dataset: `hotpotqa/hotpot_qa`
- config: `distractor`
- split: `validation`

如果你已经把 `fullwiki/validation` 之类的原始样本保存成了本地 `json` 或 `jsonl`，现在也可以直接一条命令完成这几步：

- 提取 `question` / `answer` / `supporting_facts`
- 把每条样本的 `context` 写成可 ingest 的文本文件
- 用隔离的语料目录和 pgvector 表重建索引
- 调当前 RAG 链路生成回答
- 运行 `ragas` 并落盘报告

命令示例：

```bash
. .venv/bin/activate
python3 -m eval.cli run-hotpotqa-local \
  --input eval/datasets/hotpotqa_fullwiki_validation_15.json \
  --limit 15
```

默认产物：

- 评测集：`eval/datasets/<输入文件名>.jsonl`
- 语料目录：`data/eval_uploads/<输入文件名>/`
- 报告目录：`eval/reports/<run_id>/`

这个命令默认使用隔离的 pgvector 表，不会复用你主应用正在使用的上传目录和向量表。

### 输出结果

每次运行都会在 `eval/reports/<run_id>/` 下生成：

- `manifest.json`
- `summary.json`
- `summary.md`
- `samples.jsonl`
- `failures.json`

评估 judge 默认也支持 provider 切换，配置放在 `.env.eval`：

```env
EVAL_JUDGE_PROVIDER=deepseek
EVAL_JUDGE_MODEL=deepseek-v4-flash
```

如果不显式配置 `EVAL_JUDGE_MODEL`，评估会按 `EVAL_JUDGE_PROVIDER` 自动回退到 `.env.eval` 中对应的模型配置。评估 embeddings 直接复用本地 `bge-m3`，不会额外调用远端 embedding API。评估 judge 与应用主链路的 `LLM_PROVIDER` 是独立可配的。

## 系统结构

请求链路如下：

1. 用户在前端上传文件
2. 文件保存到 `data/uploads/`
3. 点击“重建索引”后，`SimpleDirectoryReader` 递归读取目录
4. 本地 `bge-m3` 生成 embedding
5. 向量写入 PostgreSQL 的 pgvector 表
6. 用户提问时，先从 pgvector 检索相关片段
7. 检索结果交给当前启用的 LLM 生成最终回答
8. 返回答案和引用信息给前端展示

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
├── docker-compose.yml   # 本地 pgvector 服务
├── requirements.txt
└── .env.example
```

## 环境要求

- Linux / macOS 优先
- `python3` 可用
- Docker 和 Docker Compose 可用
- 本机具备本地 `BAAI/bge-m3` 模型目录，或者允许后续自行下载

当前开发环境默认使用：

- Python `3.12.3`
- PostgreSQL 端口映射 `5434`

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
```

然后根据 `LLM_PROVIDER` 填入对应的 API Key。

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

- 检索召回不够准
- 文档内容本身结构不清晰
- `TOP_K` 太小或太大
- 当前还没有加 reranker

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
4. 增加 reranker
5. 把当前 RAG query engine 包成 Agent 的一个工具
6. 再接网页搜索、SQL 查询、知识库路由等工具

## 当前状态总结

这个项目现在已经是一个能直接使用的基础版 RAG Web 应用，而不是单纯的脚手架。你可以：

- 上传资料
- 重建向量索引
- 发起问答
- 查看引用依据
- 继续在这套结构上往 Agent 演进
