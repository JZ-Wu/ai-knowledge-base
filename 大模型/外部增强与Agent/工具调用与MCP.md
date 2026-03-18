# 工具调用与 MCP (Model Context Protocol)

让 LLM 不只是"聊天"，而是能**调用外部工具**——查天气、执行代码、操作数据库、发邮件。

## 一、为什么需要工具调用？

LLM 的能力边界：

| 问题 | 说明 |
|------|------|
| **不能执行动作** | 只能生成文字，不能真的发邮件、写文件 |
| **不擅长精确计算** | 问 "1234567 × 7654321" 经常算错 |
| **不知道实时信息** | 不知道现在几点、今天股价多少 |
| **不能访问外部系统** | 不能查数据库、调 API |

工具调用的思路：**让 LLM 决定"该用什么工具"，系统负责执行，结果返回给 LLM 继续推理**。

## 二、工具调用的基本流程

```
用户: "北京今天多少度？"
  │
  ▼
LLM 推理: 这个问题需要实时天气数据，我应该调用天气 API
  │
  ▼
LLM 输出: { "tool": "get_weather", "args": {"city": "北京"} }
  │
  ▼
系统执行: 调用天气 API → 返回 "15°C, 晴"
  │
  ▼
LLM 继续: "北京今天 15°C，晴天。"
```

关键点：**LLM 不直接执行工具**，它只是"说"要用什么工具、传什么参数，由外部系统执行。

## 三、Function Calling 的演进

### 3.1 早期：Prompt Engineering 时代（2023 初）

把工具描述写在 prompt 里，让模型按格式输出：

```
你可以使用以下工具：
- search(query): 搜索网页
- calculator(expression): 计算数学表达式

请用 <tool>工具名(参数)</tool> 格式调用工具。

用户: 123 * 456 等于多少？
助手: <tool>calculator("123 * 456")</tool>
```

问题：格式不稳定，模型经常不按规矩输出。

### 3.2 原生 Function Calling（2023 年中）

OpenAI 率先在 GPT-3.5/4 中加入原生 function calling 支持：

```json
// 定义工具
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取指定城市的天气",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "城市名"}
        },
        "required": ["city"]
      }
    }
  }]
}

// 模型返回
{
  "tool_calls": [{
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"北京\"}"
    }
  }]
}
```

优势：
- **结构化输出**：模型原生支持 JSON 格式的工具调用，不再需要靠 prompt 引导
- **多工具选择**：模型可以从多个工具中选择合适的
- **并行调用**：一次可以调用多个工具

### 3.3 Tool Use 范式（2024）

Anthropic（Claude）、Google（Gemini）等跟进，形成行业标准范式：

```
系统提供工具定义 → 模型决定是否调用 → 系统执行 → 结果返回模型 → 模型继续推理
```

核心设计原则：

| 原则 | 说明 |
|------|------|
| **模型只做决策** | 选工具、填参数，不执行 |
| **工具描述是关键** | 好的工具描述 = 好的调用准确率 |
| **Schema 约束** | JSON Schema 定义参数类型，减少格式错误 |
| **多轮交互** | 工具结果返回后，模型可以继续调用其他工具 |

## 四、MCP：Model Context Protocol

### 4.1 MCP 解决什么问题？

2024 年之前，每个 AI 应用要接入外部工具，都需要自己写适配代码：

```
之前（碎片化）：
  Claude App ←→ [自写适配] ←→ GitHub API
  Claude App ←→ [自写适配] ←→ Slack API
  Claude App ←→ [自写适配] ←→ 数据库
  GPT App    ←→ [自写适配] ←→ GitHub API  ← 又要写一遍！

MCP 之后（标准化）：
  Claude App ←→ [MCP 协议] ←→ GitHub MCP Server
  GPT App   ←→ [MCP 协议] ←→ GitHub MCP Server  ← 同一个 Server！
  任何 App  ←→ [MCP 协议] ←→ 任何 MCP Server
```

**MCP 就像 USB-C**：统一了 LLM 应用与外部工具之间的连接协议。

### 4.2 MCP 架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  MCP Host    │     │ MCP Client  │     │ MCP Server  │
│ (AI 应用)    │ ──→ │ (协议客户端) │ ──→ │ (工具提供方) │
│ Claude Code  │     │             │     │ GitHub/Slack │
│ Cursor       │     │             │     │ 文件系统     │
│ 自建应用     │     │             │     │ 数据库       │
└─────────────┘     └─────────────┘     └─────────────┘
```

三个角色：
- **Host**：AI 应用本身（Claude Desktop、Claude Code、Cursor 等）
- **Client**：Host 内部的 MCP 客户端，负责与 Server 通信
- **Server**：工具提供方，暴露工具/资源/提示词给 Client

### 4.3 MCP Server 提供的三种能力

| 能力 | 说明 | 示例 |
|------|------|------|
| **Tools（工具）** | 可调用的函数 | `search_issues`、`create_file`、`run_query` |
| **Resources（资源）** | 可读取的数据 | 文件内容、数据库记录、API 返回 |
| **Prompts（提示词模板）** | 预定义的 prompt 模板 | 代码审查模板、SQL 生成模板 |

### 4.4 MCP 通信协议

基于 **JSON-RPC 2.0**，支持两种传输方式：

| 传输方式 | 说明 | 适用场景 |
|---------|------|---------|
| **stdio** | 通过标准输入/输出通信 | 本地 Server（最常用） |
| **SSE (HTTP)** | 通过 HTTP Server-Sent Events | 远程 Server |

```json
// Client → Server: 调用工具
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_issues",
    "arguments": {"query": "bug", "repo": "my-project"}
  }
}

// Server → Client: 返回结果
{
  "jsonrpc": "2.0",
  "result": {
    "content": [{"type": "text", "text": "Found 3 issues: ..."}]
  }
}
```

### 4.5 MCP 的意义

| 之前 | 之后 |
|------|------|
| 每个 AI 应用自己写工具适配 | 写一次 MCP Server，所有应用都能用 |
| 工具格式不统一 | JSON-RPC 标准协议 |
| 难以组合多个工具 | Server 可以自由组合 |
| 工具生态碎片化 | 统一的 MCP 生态系统 |

目前（2026.03）已有大量社区 MCP Server：GitHub、Slack、PostgreSQL、文件系统、浏览器控制、Puppeteer 等。

## 五、Skill（技能）：工具调用的高层封装

### 5.1 什么是 Skill？

**Skill = 预定义的 prompt 模板 + 工具组合 + 执行逻辑**，用来完成一个特定任务。

如果说 Tool 是"一把螺丝刀"，那 Skill 就是"组装一台电脑的操作手册"——手册里告诉你什么时候用螺丝刀、什么时候用扳手、按什么顺序操作。

```
层级关系：

Tool（工具）     → 单个函数调用，如 read_file、search_code
    ↓ 组合
Skill（技能）    → 预定义的工具编排 + prompt 模板，完成特定任务
    ↓ 自主规划
Agent（智能体）  → 动态规划、自主决策调用哪些 tool/skill
```

### 5.2 Skill 长什么样？

以 Claude Code 的 `/commit` 技能为例：

```
Skill 定义:
  名称: commit
  触发: 用户输入 /commit

  展开后的 prompt:
    "1. 运行 git status 查看变更
     2. 运行 git diff 查看具体改动
     3. 分析变更内容，生成 commit message
     4. 执行 git add + git commit"

  可用工具: [Bash, Read, Edit]
```

用户只需输入 `/commit`，系统自动展开为一整套 prompt 指令，模型按照指令依次调用工具完成任务。

### 5.3 Skill vs Tool vs Agent

| | Tool | Skill | Agent |
|---|---|---|---|
| **粒度** | 单个函数 | 预定义的工作流 | 自主规划的任务 |
| **谁决定步骤** | 模型 | **Skill 预定义** | 模型自主规划 |
| **灵活性** | 高（模型自由调用） | 中（流程固定，细节灵活） | 最高（完全自主） |
| **可靠性** | 取决于模型判断 | **高**（流程经过验证） | 较低（可能跑偏） |
| **示例** | `get_weather()` | `/commit`、`/review-pr` | "帮我重构这个模块" |

### 5.4 Skill 的本质

Skill 本质上做了两件事：

1. **把经过验证的 prompt 模板固化下来**：不用每次让模型从零开始规划，而是直接给它一套验证过的"操作手册"
2. **限定工具范围和执行顺序**：减少模型"选错工具"或"步骤遗漏"的概率

这和前面讲的 MCP 的 **Prompts（提示词模板）** 能力是同一个思路——MCP Server 可以暴露预定义的 prompt 模板，本质上就是一种 Skill。

> 一句话：**Tool 是积木，Skill 是拼装说明书，Agent 是能自己设计图纸的工程师**。

## 六、Tool Use 的关键设计问题

### 6.1 工具描述的质量

工具描述直接影响模型的调用准确率：

```json
// ❌ 差的描述
{ "name": "search", "description": "搜索" }

// ✅ 好的描述
{
  "name": "search_code",
  "description": "在代码仓库中搜索匹配指定正则表达式的代码片段。适用于查找函数定义、变量引用、import 语句等。不适合搜索文件名（请用 find_file）。",
  "parameters": { ... }
}
```

### 6.2 工具数量与选择

- 工具太多 → 模型选择困难，容易选错
- 工具太少 → 能力受限
- 实践中：**10-30 个工具**是比较好的范围
- 工具特别多时，可以分组/分层：先选类别，再选具体工具

### 6.3 安全考量

| 风险 | 说明 | 缓解 |
|------|------|------|
| **Prompt 注入** | 恶意输入让模型调用危险工具 | 权限控制、确认机制 |
| **过度授权** | 模型有权限但不该用 | 最小权限原则 |
| **数据泄露** | 工具返回敏感数据被模型引用 | 输出过滤 |

## 七、模型如何学会工具调用？

工具调用能力**不是天生的**，需要在后训练阶段专门训练。

### 7.1 SFT 阶段：学格式

SFT 数据中混入大量工具调用样本，教模型学会输出结构化调用格式：

```
训练样本示例:
  用户: "北京今天多少度？"
  模型: {"tool": "get_weather", "arguments": {"city": "北京"}}
  工具返回: {"temperature": 15, "condition": "晴"}
  模型: "北京今天 15°C，晴天。"
```

SFT 阶段的训练数据是**多种能力混合**在一起训练的，不是分开多次训：

```python
sft_data = (
    对话数据 ~50%
    + 工具调用数据 ~15%    # ← 学会工具调用格式
    + 代码数据 ~15%
    + 数学推理数据 ~10%
    + 安全/拒绝数据 ~5%
)
```

### 7.2 RL 阶段：学判断

SFT 让模型"会调"，RL 让模型"调得好"：

| 阶段 | 学到什么 | 示例 |
|------|---------|------|
| **SFT** | 输出格式、基本调用能力 | 看到天气问题 → 输出 get_weather 调用 |
| **RL** | 判断力——该不该调、调哪个 | "你好" → 不需要工具；"今天天气" → 需要 get_weather 而非 search |

RL 阶段通过奖励信号（Reward Model 打分或规则奖励）优化调用决策的准确性。

### 7.3 与 RAG 的训练需求对比

| 能力 | 需要专门训练吗 | 说明 |
|------|--------------|------|
| **RAG（利用检索结果回答）** | 不需要 | RAG 是工程侧把上下文塞进 prompt，模型只需"会读上下文回答"——SFT 自然学会 |
| **工具调用格式** | 需要（SFT） | 模型要学会输出 JSON 格式的调用指令 |
| **工具调用判断** | 需要（RL） | 模型要学会何时调用、调用哪个工具 |
| **推理/CoT** | 需要（RL） | 如 DeepSeek-R1 用 GRPO 训练长链推理能力 |

> 一句话：**RAG 不动模型，工具调用要训模型**。RAG 是"开卷考试给参考书"，工具调用是"教模型学会使用计算器"。

## 八、Tool Calling vs RAG

| | RAG | Tool Calling |
|---|---|---|
| **目的** | 获取知识 | 执行动作或获取实时数据 |
| **数据流** | 检索 → 读取 → 生成 | 决策 → 执行 → 观察 → 继续 |
| **是否改变外部状态** | 不改变（只读） | 可以改变（读写） |
| **典型场景** | 知识问答 | 发邮件、写代码、查实时数据 |

---

**相关文档**：
- [RAG详解](RAG详解.md) — 检索增强生成完整流程
- [Agent框架详解](Agent框架详解.md) — Agent 如何编排工具调用

[返回上级](README.md) | [返回总目录](../../README.md)
