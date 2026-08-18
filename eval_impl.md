# RAG Eval 可执行实施指引

## 0. 本文用途与边界

本文是 `计划.md` 的实施拆解，供实现者按阶段完成离线 RAG 评测。它是执行约束，不是新需求：与 `计划.md` 冲突时，以 `计划.md` 为准；实现过程中不得自行扩大范围。

目标是建立一个可复现的、只评测生成质量且能完整回溯检索过程的离线流程：

- 固定 HotpotQA distractor validation 的 200 条 retrieval 集与其中确定性选取的 20 条 generation 集。
- 仅对 20 条 generation 集执行完整 RAG 和 Ragas Judge。
- 200 条 retrieval 集当前只准备、保存和建共享索引，不运行 RAG、Ragas 或任何检索指标。
- 保存真实 pipeline 的输入、检索轨迹、最终上下文、回答、逐样本 Judge 状态和可比较的运行产物。
- 只在明确指定 baseline 时输出 comparison；不自动更新 baseline，也不做门禁阻断。

### 0.1 当前代码事实（实现必须遵守）

| 责任 | 当前位置 | 实施要求 |
| --- | --- | --- |
| RAG pipeline | `app/rag.py` 的 `RagService.evaluate_query()` | 在这个真实调用路径上采集 trace；不能让 eval 重新做一次检索。 |
| 生成上下文 | `RagService._build_context()` 与 `_generate_answer()` | trace 中的 `serialized_context` 必须来自同一份上下文构造结果。 |
| Eval 适配层 | `eval/adapter.py` | 负责把结构化 RAG 结果转换为 eval 行，不承担检索逻辑。 |
| Runner | `eval/runner.py` | 负责 preflight、样本执行、Judge、聚合和 artifacts。 |
| 报告与落盘 | `eval/reporting.py` | 扩展为原子 run 写入，不能让 runner 各自拼装文件系统逻辑。 |
| 数据集模型 | `eval/dataset.py` | 保留 source sample ID；新增字段必须向后兼容已有 JSONL。 |
| HotpotQA 转换 | `eval/hotpotqa.py`、`eval/prepare.py` | 改为共享 corpus，不能按问题写独占语料。 |
| CLI | `eval/cli.py` | 收敛为本文第 4 阶段规定的命令，删除或弃用会泄漏的旧快捷流程。 |

### 0.2 明确禁止（防止过度设计）

- 不实现 MRR、HitRate、Recall、MAP、nDCG、`RetrieverEvaluator` 或 `pytrec_eval`。
- 不创建在线服务、Web UI、任务队列、数据库 schema migration、后台 worker 或新的评测框架。
- 不把 `reference`、`reference_contexts`、supporting facts 传给生成 LLM，也不为单个 query 构造专属语料库。
- 不将 API key、Authorization header 或完整 `.env.eval` 写进 artifact、异常文本、日志或报告。
- 不做自动 baseline 选择/覆盖、分数阈值或 CI 阻断。
- 不尝试记录 provider 不返回的 token usage；字段可为空并明确标为 unavailable。
- 不为了兼容所有历史 Ragas 版本写复杂反射层；仅针对锁定依赖版本确认 `factual_correctness` 是否可用，并把不可用原因记录为 skipped。
- 不修改公开 benchmark 的原始样本。下载文件和派生文件均不提交，除非已确认 license 允许。

### 0.3 统一实现规则

1. 每个阶段先补该阶段的失败测试，再实现最小代码；完成阶段后运行指定测试。
2. 运行测试不得调用真实 LLM、Judge、HuggingFace 下载或 PostgreSQL；使用 fake `RagService`、mock Judge 或临时文件夹。
3. 所有时间以 UTC ISO 8601 存储；持续时间用毫秒数。
4. JSON artifact 使用 UTF-8；ID、列表顺序、哈希输入与配置序列化必须确定性。
5. 任何会改动 PostgreSQL 表的命令都必须先校验表名严格匹配 `eval_[a-z0-9_]+`；eval 重建必须同时为 vector、docstore、file-index 三张表设置独立名称并逐个校验，绝不能继承 app 的 `DOCSTORE_TABLE` 或 `FILE_INDEX_TABLE`。
6. 每个阶段完成后运行 `pytest -q`；只在实际集成验证时运行本地服务，并在验证完成后立即关闭。

## 1. 目标契约

### 1.1 数据集目录和身份

以 `<dataset_root>/<dataset_name>/` 表示一个准备完成的数据集，最少包含：

```text
dataset_manifest.json
corpus.jsonl
generation.jsonl
retrieval.jsonl
```

`generation.jsonl` 中恰有 20 条，`retrieval.jsonl` 中恰有 200 条，且 generation 的 `source_sample_id` 集合是 retrieval 的子集。测试允许构造缩小 fixture，但生产命令不得接受任意数量作为成功数据集。

每个 query 行至少为：

```json
{
  "id": "hotpotqa_<source_id>",
  "source_sample_id": "<原始 HotpotQA id>",
  "user_input": "...",
  "reference": "...",
  "reference_contexts": ["..."],
  "tags": ["hotpotqa", "distractor"],
  "difficulty": "easy|medium|hard",
  "question_type": "bridge|comparison"
}
```

`corpus.jsonl` 的段落 identity 必须是 `title + paragraph_index` 的稳定组合，并保存原始段落文本、标题、段落索引与来源 sample IDs。去重只按稳定 identity 进行；若同一 identity 内容不同，prepare 必须失败而不是静默选择一个版本。

`dataset_manifest.json` 至少记录来源 URL、license、版本、下载 UTC 日期、原始文件 SHA-256、转换版本、抽样 seed、200/20 的 source IDs、corpus 去重规则、corpus SHA-256 和各文件 SHA-256。

### 1.2 单样本 pipeline 输出

新增内部 dataclass（建议置于 `app/rag.py`，或仅放在与它同级的轻量模型模块）`RAGPipelineResult`。`evaluate_query()` 可以保留当前返回值以保护 Web 调用；新增一个供 adapter 使用的公共方法，例如 `evaluate_query_with_trace(message)`，返回该结构化结果。

该结果至少包含 answer、citations、final contexts、`retrieval_trace` 和 generation request 元数据。不要在 adapter 中反推 trace。

`retrieval_trace` 的最小 JSON 形状：

```json
{
  "query": "原始问题",
  "transformed_queries": [],
  "retrieval_mode": "vector|hybrid|fusion|hyde",
  "top_k": 5,
  "fetch_k": 20,
  "candidate_count": 20,
  "reranker_enabled": true,
  "rerank_input_count": 20,
  "rerank_output_count": 5,
  "retrieved_nodes": [{"rank": 1, "node_id": "...", "score": 0.0, "file_name": "...", "file_path": "...", "page_label": "...", "text": "完整文本"}],
  "final_contexts": [{"rank": 1, "node_id": "...", "score": 0.0, "file_name": "...", "file_path": "...", "page_label": "...", "text": "完整文本"}],
  "generation_input": {
    "system_prompt_template_id": "app.system_prompt",
    "system_prompt_hash": "sha256:...",
    "user_prompt_template_id": "app.answer_prompt.v1",
    "user_prompt_hash": "sha256:...",
    "serialized_context": "实际送入 LLM 的 context",
    "context_token_count": null,
    "request_token_count": null,
    "output_token_count": null,
    "context_operations": []
  },
  "timings_ms": {"query_transform": 0.0, "retrieve": 0.0, "rerank": 0.0, "generation": 0.0}
}
```

约束：

- `retrieved_nodes` 是 rerank 前的完整初始候选；`final_contexts` 是 rerank 后最终顺序。未开启 reranker 时二者相同，但均需写入。
- `node_id` 用 LlamaIndex node ID；不可用时为 `null`，不得伪造。
- `transformed_queries` 必须记录实际送入 retriever 的 query：multi-query 必须复用同一次生成的 query 列表，HyDE 必须记录实际 embedding 文本；不得为了记录而额外发起 LLM 调用，也不得只写原始 query。
- `serialized_context` 必须严格等于 `_build_context()` 返回值。最终 prompt 可由 system prompt、固定 user-template 和该字段重新构造。
- 当前 `_build_context()` 只有规范化、排序和拼接行为时，`context_operations` 如实写入；没有去重/截断时不能写这些操作。
- 将 `_generate_answer()` 重构为先构造一个 generation request 对象、再调用 provider。两条 provider 路径必须接收同一 system/user prompt 内容。
- `reference`、`reference_contexts` 绝不传入 `RagService` 的 generation request。

### 1.3 运行 artifact 契约

成功 run 的最终目录为 `eval/reports/<run_id>/`，包含且只要求以下核心文件：

```text
manifest.json
samples.jsonl
retrieval_traces.jsonl
summary.json
summary.md
failures.json
comparison.json              # 仅传入 --baseline 时
```

- `samples.jsonl`：一行一个 generation 样本，保存 source ID、回答、references、final contexts、citations、sample status、每个 metric 的状态/分数/错误。
- `retrieval_traces.jsonl`：一行一个 pipeline 成功且 generation input 未泄漏的样本；包含 `id`、`source_sample_id` 和完整 trace。
- `failures.json`：逐项失败记录以及可按 failure type 聚合的统计；不得只留错误字符串。
- `manifest.json`：必须包含 `status: "completed"`、run ID、UTC 时间、git SHA（不可用则 `null`）、dataset absolute path/hash、source IDs hash、corpus hash、实际 app/pipeline 配置、judge 配置、embedding 标识和哈希、prompt hash。不得保存 key。
- `summary.json`：只保存聚合；`summary.md` 从 `summary.json` 和 manifest 重建，不触发 RAG/Judge。

Run 原子语义：先创建同级唯一临时目录 `<run_id>.tmp-<uuid>`，写齐 artifacts、校验 JSONL 行数和 summary 后原子 rename 为正式 run 目录，最后的 manifest 才写 `completed`。任何 preflight 失败不创建正式 run；执行中失败则将临时目录改名为明确的 `.failed-<uuid>` 并写 `manifest.status: "failed"` 或 `"partial"`。failed/partial run 不可 report、comparison 或 baseline。

### 1.4 样本状态与指标状态

样本 status 只能为：

- `scored`：pipeline 成功，至少一个 Judge 指标得到有限数值。
- `judge_failed`：pipeline 成功但所有尝试指标均失败。
- `invalid`：发现 reference/reference context 泄漏到 generation input，不调用 Judge。
- `pipeline_failed`：检索、rerank、生成或 trace 构造失败。

每个 metric 独立写 `status`：`scored`、`failed`、`skipped`、`nan`。仅 `scored` 且为有限数值的样本可参与该 metric 的均值、分位数和 delta；失败与 NaN 绝不能转为 0。

## 2. 实施阶段

每个阶段完成后提交前均运行本阶段列出的测试和 `pytest -q`。不得在未通过前推进下一阶段。

### 阶段 1：固定数据模型与可重复工具

**目的**：先确立数据 identity 与确定性工具，避免后续 trace/baseline 依赖临时格式。

**修改范围**：`eval/dataset.py`，新增小型纯函数模块（可放 `eval/artifacts.py`），对应测试。

**实现**：

1. 为 `EvalSample` 新增必填 `source_sample_id`，并在仅为旧本地 fixture 兼容的情况下允许它默认回退为 `id`；新 prepare 产物必须显式写入。
2. 提供纯函数：规范化 JSON、计算 SHA-256、计算文件 SHA-256、计算有序 source ID 集合 hash、获取 git SHA（失败返回 `None`）。
3. 添加只允许 `eval_[a-z0-9_]+` 的 PostgreSQL 表名验证函数；拒绝空值、引号、点、连字符和大小写。
4. 扩展加载校验：缺文件、空 JSONL、重复 `source_sample_id`、generation 非 retrieval 子集必须能被调用方明确区分，不能静默按空集成功。

**测试**：新增 `tests/test_eval_artifacts.py` 和扩展 `tests/test_eval_dataset.py`。覆盖哈希稳定性、git SHA 缺失、合法/非法表名、重复 source ID、空/缺失 dataset。

**验收**：纯函数无需 LLM、数据库或网络即可完成；所有 hash 对相同输入稳定，敏感配置键永远不进入序列化快照。

### 阶段 2：RAG 真实 pipeline trace

**目的**：让一次真实 RAG 调用同时产出回答和可审计 trace，保持现有 Web API 不变。

**修改范围**：`app/rag.py`、必要时 `app/schemas.py`、`eval/adapter.py`、新增/扩展 RAG 与 adapter 测试。

**实现**：

1. 添加 dataclass/模型表示节点快照、generation input、trace 和 pipeline result。只使用 Python 标准库 dataclass 或现有 Pydantic，不引入新依赖。
2. 从 `_retrieve_nodes()` 返回或在其旁边采集 retriever 输出，保证 rerank 前节点列表不会被覆盖。
3. 在 rerank 前后分别记录数量、节点属性和耗时；无 reranker 时填写真实的相同列表与计数。
4. 抽出单一的 context/prompt 构造路径，使 generation 与 trace 共享 `serialized_context`、prompt hash 和模板 ID。
5. 添加 `evaluate_query_with_trace()`；保持 `evaluate_query()` 与 `chat()` 的外部返回类型和行为完全不变，它们可委托新方法。
6. Adapter 改用新方法，并把 trace 原样带入 `RAGRunResult`。
7. 添加 generation input 泄漏检查函数：检查完整 reference 与每条非空 reference context 是否是实际 system/user prompt 或 serialized context 的子串。命中时返回 invalid 原因。它只在 eval runner 调用，不能改变正常 chat。

**测试**：

- 不启用 reranker：初始与最终节点相同、`fetch_k == top_k`、context 与 prompt 完全一致。
- 启用 fake reranker：记录 rerank 前 20 个、后 5 个以及正确顺序。
- 每个节点含完整 text、metadata 和 score；citation snippet 不作为 trace text。
- `_build_context()` 的输出等于 trace `serialized_context`。
- `evaluate_query()`、`chat()` 现有测试不回归。
- 泄漏检查命中 reference 和 reference context 各一例，且普通问题不误报。

**验收**：fake RAG 环境中 adapter 一次执行只触发一次检索和一次生成；artifact 所需的 trace 字段均可直接取得，无需第二次调用 RAG。

### 阶段 3：安全、原子的运行产物

**目的**：生成可复现且不会被半成品污染的 run 目录；本阶段仍可用 fake Judge。

**修改范围**：`eval/reporting.py`、`eval/runner.py`、测试。

**实现**：

1. 以 run writer/context manager 替换 `create_run_dir()` 直接创建正式目录的方式。writer 负责临时目录、写 JSON/JSONL、完成校验、rename 和失败状态。
2. run ID 必须含 UTC 时间和足够的随机/递增成分，防止同一秒冲突。
3. manifest 写入显式 `status`、数据集和 corpus identity、实际 app/eval 配置的安全快照。base URL 可根据配置脱敏，但绝不可写 key。
4. 每个 pipeline 成功样本立即积累独立 sample row 与 trace row；pipeline 失败样本仍写 sample row 和 failure record，但不写 trace row。
5. 生成 `failures.json` 中的结构化字段：`source_sample_id`、`stage`、`error_type`、安全的 message、metric key（适用时）。禁止把异常对象 repr 或环境变量整体串行化。
6. 仅在完成校验后写最终 summary、summary.md 和 completed manifest；若抛出异常或 writer 失败，生成可诊断 failed/partial 临时目录，正式 reports 下无 completed run。

**测试**：

- 成功 run：所有核心文件存在，manifest 为 completed，JSONL 行数与 summary 相符。
- preflight 前失败：没有正式 run。
- pipeline 中途失败：正式目录不存在，保留 failed/partial 目录且 manifest 状态正确。
- 同秒两次创建不冲突。
- 设置假 API key 后扫描所有 artifact 字节，断言不含该值。

**验收**：删除进程或模拟 write error 后，不会留下 `status: completed` 的不完整目录；`summary.md` 可通过保存的 JSON 重建。

### 阶段 4：HotpotQA 下载、准备、独立索引和 CLI

**目的**：把当前按样本生成文档的本地快捷流程，替换为固定共享 corpus 的数据集流程。

**修改范围**：`eval/hotpotqa.py`、`eval/prepare.py`、`eval/cli.py`、README 的命令段落、测试。

**实现**：

1. 提供 `eval data fetch --dataset hotpotqa-distractor`：从代码中固定的官方 URL/Hub 标识下载指定 validation 版本到代码目录外的原始数据目录。写下载元数据；实现前先确认 license 并在 manifest 记录。
2. 提供 `eval data prepare --dataset hotpotqa-distractor --seed <int>`：稳定地选 200 条；稳定地从这 200 条选 20 条 generation。生产命令不暴露任意 `--limit` 成功路径。
3. 遍历 200 条的所有 HotpotQA `context` 段落，按 `title + paragraph_index` 生成共享 `corpus.jsonl`。每条段落必须记录它出现在哪些 source sample 中；不能按 query 输出一个 `.txt` 文件。
4. 写 query 文件与 dataset manifest，校验 200/20 数量、子集关系、ID 唯一性、哈希和段落内容冲突。
5. 提供 `eval index rebuild --dataset <name>`：读取 manifest/corpus，验证 vector、docstore、file-index 的全部 target table 名，仅清空并重建对应 `eval_*` 表。将三张表、corpus hash、文档数、chunk 数、索引配置 hash、UTC 时间写入 index state 文件。
6. `eval run generation --dataset <name>` 是唯一完整评测入口；默认且只读取 generation 20 条。`eval run retrieval` 当前不注册为可执行命令。
7. `eval report --run <run_id_or_path>` 只读取 artifacts 并重建 `summary.md`，不构造 `RagService`、不调用 Judge。
8. 删除 `run-hotpotqa-local` 或标为非公开、不可用于评测的开发工具；它不能再作为 README 推荐路径，因为它会按样本写语料。

**测试**：使用 2-4 条精简 HotpotQA fixture 测试转换算法；参数化测试可允许 fixture 规模，但生产函数的最终集完整性校验必须单独测试 200/20。测试共享段落去重、冲突检测、generation 子集、seed 决定性、表名拒绝、index state 写入以及 report 不调用 RAG/Judge。

**验收**：准备出的任意 generation query 可查询同一个 corpus；corpus 内容不是只来自该 query 的 supporting facts；重建索引只能接触 manifest 指定的三张 `eval_*` 表，并有测试断言 app 默认表绝不出现在 eval settings 中。

### 阶段 5：Preflight 与 Judge 集成

**目的**：在任何外部调用前阻止不可复现或不完整的评测；只评测回答质量。

**修改范围**：`eval/config.py`、`eval/metrics.py`、`eval/runner.py`、`.env.eval.example`、测试。

**实现**：

1. 将 Judge 配置收敛为 OpenAI-compatible 的 provider/key/base URL/model/temperature/timeout/retry/workers。支持当前已使用的 provider 值只要它们映射到同一显式配置模型；不得从 app 主 LLM 配置偷读 Judge credential。
2. 增加 `EVAL_JUDGE_TEMPERATURE`，默认 0；manifest 记录实际值。key 仅传给 client。
3. embeddings 复用 `app.config.get_settings()` 指向的本地 embedding 模型配置；manifest 保存模型标识、存在时的版本/路径哈希与配置 hash，不保存无关 credential。
4. Ragas 仅启用 `faithfulness`、`response_relevancy`，以及经锁定依赖验证可用的 `factual_correctness`。移除 `context_precision/context_recall` 作为本阶段 active metric。
5. Runner 在创建 writer 之前完成 preflight：dataset manifest 完整、generation 正好 20 条、corpus hash 匹配、index state 存在且 table/corpus hash 匹配、本地 embedding 可用、Judge config 完整、metric API 可构造。失败返回非零、写安全错误到 stderr/调用方，不创建完成 run。
6. 先顺序运行每个样本 RAG，做泄漏检查，之后仅对 pipeline 成功且非 invalid 的行调用 Judge。初始实现可保留 Ragas 批量调用，但必须能把每个 metric 的成功、失败、NaN 映射回 source sample ID；无法做到时改为逐样本评测，优先正确性而非并发。
7. 对 `NaN` 使用 `math.isfinite()` 判定；以每个 metric 的 attempted/scored/failed/skipped/nan 计数和有限数值集合生成 mean、p50、p95。测试必须断言失败/NaN 不进入任何统计值。

**测试**：

- 每一个 preflight 条件失败均不调用 fake RAG、fake Judge，也不创建正式 run。
- fake Judge：一个 metric 成功、一个超时、一个 NaN，断言状态与聚合完全正确。
- pipeline failure 与 invalid 不调用 Judge。
- `factual_correctness` 不可用时为 skipped，且在 manifest/summary 有明确原因。
- metrics 中不出现 `context_precision`、`context_recall` 或检索指标。

**验收**：20 条之外、旧 corpus 索引、缺 Judge key、缺 embedding、指标 API 不兼容均在外部调用前失败；一项指标失败不影响同一样本其他指标有效分数。

### 阶段 6：Baseline 比较、报告与人工核验

**目的**：提供严格但不阻断的回归观察能力。

**修改范围**：`eval/runner.py`、`eval/reporting.py`、可新增 `eval/comparison.py`、CLI、测试。

**实现**：

1. `eval run generation --dataset <name> --baseline <run_id_or_path>` 仅接受 completed baseline。解析路径必须限制在 configured reports/baselines 根目录或用户明确给出的现有目录，禁止猜测最新 run。
2. aggregate delta 的前置条件：current/baseline dataset hash 相同、corpus hash 相同、generation source IDs 的有序集合 hash 相同、样本数为 20、metric key 集合相同。任一不一致时写 `comparable: false` 和理由，不计算 aggregate delta。
3. 当可比较时，按 `source_sample_id` 和 metric key 对齐；只对两边均为 finite scored 的样本算 delta。单独统计 current 新失败、恢复成功、缺失、NaN 和不可比较项。
4. 比较 pipeline/Judge/embedding/prompt 的安全配置快照，生成结构化 config diff；这些差异不阻止 comparison，因为它们正是实验变量。
5. summary/report 显示各 metric 的有效样本数、均值、p50、p95、状态计数、tag/difficulty/question_type 分桶、失败列表和对应 `retrieval_traces.jsonl` 引用、comparison 结果。不要在报告中设置 pass/fail 门禁。
6. 提供一次人工核验清单（文档即可）：抽查低分/失败样本，核对 trace 初始候选、rerank 结果、final contexts、serialized context、prompt hash、回答与 Judge 状态；确认 corpus source IDs 不等于单 query 专属内容。

**测试**：

- dataset hash、source ID、指标集不一致各自导致 `comparable: false`。
- 同一 source ID 的 finite scored 对产生正确 delta。
- failed/NaN/missing 不被作为 0，不进入均值或 delta。
- baseline 为 failed/partial 时拒绝。
- report 重新生成不调用 RAG/Judge。

**验收**：comparison 可解释哪些样本参与了每个指标 delta，且实验配置改变会显示 diff；默认无 baseline 时不会生成 comparison 或修改任何已有 run。

## 3. 交付顺序和依赖

严格按以下顺序合入，避免后续接口反复返工：

1. 阶段 1：identity、hash、表名校验。
2. 阶段 2：真实 RAG trace 与 adapter。
3. 阶段 3：原子 artifacts 与安全失败记录。
4. 阶段 4：固定 HotpotQA 数据集、共享 corpus、索引与 CLI。
5. 阶段 5：preflight、Ragas Judge、指标状态与聚合。
6. 阶段 6：baseline、报告、人工核验。

阶段 4 的 prepare/index 可先使用临时 fake index state 做单测，但在阶段 5 之前必须接到真实 index state。不要先实现 baseline 或报告格式，再倒推 artifact schema。

## 4. 命令契约

最终公开命令：

```bash
python3 -m eval.cli data fetch --dataset hotpotqa-distractor
python3 -m eval.cli data prepare --dataset hotpotqa-distractor --seed 20260812
python3 -m eval.cli index rebuild --dataset hotpotqa-distractor
python3 -m eval.cli run generation --dataset hotpotqa-distractor
python3 -m eval.cli run generation --dataset hotpotqa-distractor --baseline <run_id_or_path>
python3 -m eval.cli report --run <run_id_or_path>
```

命令返回：成功为 0；输入、preflight、baseline 不可比以外的执行前错误为非 0；Judge 或单样本 pipeline 部分失败可产生 completed run，但 summary 必须如实报告失败。所有命令都不得启动 Web 服务。

`.env.eval` 最小配置：

```dotenv
EVAL_JUDGE_PROVIDER=openai_compatible
EVAL_JUDGE_API_KEY=...
EVAL_JUDGE_BASE_URL=https://third-party.example/v1
EVAL_JUDGE_MODEL=...
EVAL_JUDGE_TEMPERATURE=0
EVAL_TIMEOUT_SECONDS=120
EVAL_MAX_RETRIES=2
EVAL_MAX_WORKERS=1
```

允许为向后兼容映射旧变量，但 `.env.eval.example`、README 和 manifest 必须以以上统一变量为准。

## 5. 最终验收

实施完成后，依次执行并记录结果：

```bash
pytest -q
python3 -m eval.cli data prepare --dataset hotpotqa-distractor --seed 20260812
python3 -m eval.cli index rebuild --dataset hotpotqa-distractor
python3 -m eval.cli run generation --dataset hotpotqa-distractor
python3 -m eval.cli report --run <上一步 run_id>
```

在具备有效 `.env.eval`、本地 embedding、PostgreSQL 和已确认的数据集 license 的环境中，最终验收需全部满足：

- generation run 只执行固定 20 条；retrieval 200 条仅被准备和索引，不运行 RAG/Judge。
- 每个 pipeline 成功样本能从 trace 看见原始及实际 transformed query、实际配置、rerank 前后节点、完整 final context、真实 serialized context、prompt hash、context/request token 数、耗时和回答。
- final context 与真实 generation prompt 的 context 部分逐字一致；reference/reference contexts 未出现在 generation input。
- artifacts 完整、原子，manifest 为 completed；API key 不出现在任一 artifact；manifest 包含完整 pipeline/chunk/parser/prompt/embedding 快照和绝对 dataset path。
- `faithfulness`、`response_relevancy` 和可用时 `factual_correctness` 都有逐样本状态和正确聚合；没有检索质量指标或伪造 0 分。
- 旧 corpus hash、非法 eval 表、缺配置、空集、不是 20 条的 generation 集、无 index state 都会在任何 RAG/Judge 调用前失败。
- 指定可比较 baseline 时有逐样本 delta、聚合 delta、失败变化和配置 diff；不满足可比条件时明确拒绝 aggregate delta。
- `pytest -q` 全绿，且所有测试与 report 重建均不产生真实外部调用。

## 6. 实施过程中需要停下确认的情况

以下情况不是自行设计的授权，必须先向需求方确认：

1. HotpotQA 的具体下载版本、license 或允许用途无法确认。
2. 当前锁定的 Ragas 版本中 `factual_correctness` 的 API 与计划不兼容，且没有不升级依赖的直接实现方式。
3. LlamaIndex 无法从真实 multi-query/HyDE 调用中暴露 transformed query，且需要侵入第三方库或额外发起 LLM 调用才能取得。
4. PostgreSQL 当前 schema 无法以独立 `eval_*` 表建索引，且修复需要改变开发/生产表或数据库权限。
5. 实际 provider 无法返回/可靠估算 token usage，且需求方要求该字段必须为数值而非 unavailable。
