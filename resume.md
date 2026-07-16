# 项目经历

## LlamaIndex RAG 知识库问答与评估系统

基于 Python 搭建离线优先的 RAG 问答系统，使用 LlamaIndex 完成文档加载、向量索引与检索增强生成，并补齐 `ragas` 评估链路，形成从知识入库、检索问答到效果评测的完整闭环。

- 负责设计并实现 RAG 主链路，支持本地文档导入、索引重建、相似度检索、引用返回和基于检索上下文的答案生成。
- 使用 LlamaIndex 对接文档读取与向量索引能力，结合 PostgreSQL + pgvector 持久化向量数据，避免纯内存方案带来的重复构建成本。
- 接入本地 embedding 模型 `BAAI/bge-m3`，在不依赖远程 embedding API 的前提下完成向量化与检索，提升系统可控性并降低调用成本。
- 搭建 `ragas` 离线评估流程，将真实问答样本转换为评测数据集，评估检索召回、答案相关性与事实一致性等指标，输出结构化报告用于迭代优化。
- 将应用主链路与评估链路解耦，支持独立配置评测模型与批处理参数，便于后续做不同模型、不同检索策略的横向对比。

技术栈：Python、LlamaIndex、ragas、PostgreSQL/pgvector、本地 embedding 模型 `BAAI/bge-m3`。 
