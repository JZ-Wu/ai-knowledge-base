# 外部增强与 Agent

LLM 本身只会"生成文本"。要让它真正有用，需要外挂各种外部能力：

| 能力 | 方式 | 类比 |
|------|------|------|
| **知识** | RAG（检索增强生成） | 给 LLM 一本参考书 |
| **技能** | Tool Calling / MCP | 给 LLM 一套工具箱 |
| **自主性** | Agent（规划+循环执行） | 给 LLM 大脑+手脚 |

三者不互斥——Agent 经常同时使用 RAG 和 Tool Calling。

## 知识点索引

- [RAG详解](RAG详解.md) — 完整流程、分块策略、Embedding、向量数据库、Reranker、混合检索、高级技巧
- [工具调用与MCP](工具调用与MCP.md) — Function Calling、Tool Use 范式、MCP 协议详解
- [Agent框架详解](Agent框架详解.md) — ReAct、规划与记忆、Coding Agent、2025-2026 Agent 爆发的原因

---
[返回上级](../README.md) | [返回总目录](../../README.md)
