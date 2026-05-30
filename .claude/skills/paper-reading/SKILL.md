---
name: paper-reading
description: 对学术论文做批判精读 + 研究缺口分析。区分作者声称 vs 真实成立 vs 未明说，并从用户研究背景出发找可切入的研究空间。当用户上传或粘贴一篇论文（PDF / arXiv 链接 / 纯文本），或要求"精读 / 讲解 / 总结 / limitation 在哪 / 这篇怎么样"时使用。同方法论同时被右侧 AI 助手用于论文卡片页对话——canonical 方法论存在 `idea-research/methods/paper-critical-reading.md`，本 skill 是 Claude Code 入口。
---

# Paper Reading（论文批判精读）

**canonical 方法论文档**：[`idea-research/methods/paper-critical-reading.md`](/idea-research/methods/paper-critical-reading.md)

## 触发时该做什么

1. **先 Read** canonical 文档：`idea-research/methods/paper-critical-reading.md`
2. 按文档里的方法论应用到当前论文 / 用户问题：
   - §0 用户研究背景（缺口分析锚点）
   - §1 总体工作流（每节定位→讲解→批判→串联→缺口）
   - §2 逐节针对性打法（摘要 / Related Work / Overview / 方法 / 实验 / Limitation）
   - §3 声称 vs 真实 vs 未明说（最重要的纪律）
   - §4 未明说局限清单（逐条排查）
   - §5 研究缺口分析（结尾的"我的机会"）
   - §6 语气与格式
   - §7 完整精读节奏示例

## 和 paper-card 的分工

| skill | 产物 | 内容 |
|-------|------|------|
| **paper-card** | markdown 文件 `wiki/papers/<paper>.md` | **纯事实理解**（任务 / 方法 / 公式 / 实验数字） |
| **paper-reading**（本 skill） | 对话回复，可选沉淀到 `wiki/gaps/<topic>.md` | **批判 + 缺口**（声称-真实-未明说、对用户课题的研究空间） |

**两者绝不混写**：
- 写 paper-card 时不写批判
- 做 paper-reading 时不污染已有的 paper-card

## 为什么是个 thin 指针

- canonical 方法论统一在 `idea-research/methods/paper-critical-reading.md`，让 **Claude Code 和右侧 AI 助手共用同一份方法**（AI 助手在论文卡片页能直接读到这个文件作上下文）。
- 改方法只改一个地方，避免两份不一致。
- 本 skill 保留 trigger keywords，Claude Code 仍会在用户说"精读 / 讲解 / limitation"时自动激活。

## 历史变更

- 2026-05-29：从 195 行的完整方法论改为 thin 指针，正文挪到 `idea-research/methods/paper-critical-reading.md` —— 让 AI 助手也能用同一份。
