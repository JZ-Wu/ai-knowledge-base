# AI 知识库

<div id="kb-stats" class="kb-stats"></div>

> 选中文字 → 问 AI → AI 直接改源文件。

<!-- 在这里放一张截图或 GIF：展示选中文字后 AI 侧边栏弹出、AI 回答并编辑文件的完整流程 -->
<!-- ![demo](docs/assets/demo.gif) -->

一个 **AI 原生**的 Markdown 知识库。不是在笔记里嵌一个聊天框——是 AI 真的能**读你的笔记、改你的笔记**。

## 它能做什么

### 1. 选中提问，带上下文

选中页面上任意一段文字，点击浮动按钮，AI 自动带着**当前页面内容 + 你选中的文字**回答你的问题。不需要复制粘贴，不需要切换窗口。

<!-- ![选中提问](docs/assets/select-ask.png) -->

### 2. AI 直接编辑源文件

对 AI 说"把这段公式推导补充完整"或"帮我加一个对比表格"，它不是给你一段文本让你自己粘——它**直接修改对应的 .md 文件**。刷新页面就能看到更新。

<!-- ![AI 编辑](docs/assets/ai-edit.png) -->

### 3. 内置源码编辑器

`Ctrl+Shift+E` 打开 CodeMirror 编辑器，自动定位到你正在阅读的位置。Markdown 语法高亮，`Ctrl+S` 保存，所见即所得。

<!-- ![编辑器](docs/assets/editor.png) -->

### 4. 数学公式原生支持

KaTeX 渲染，行内 `$...$` 和行间 `$$...$$` 都支持。写满公式推导的技术笔记也能完美显示。

### 5. 模型切换 + Thinking 模式

Opus / Sonnet / Haiku 随时切换。开启 Thinking 模式让 AI 先"想一想"再回答，适合复杂推导和代码生成。

## 技术栈

| 层 | 技术 | 作用 |
|----|------|------|
| 渲染 | [Docsify](https://docsify.js.org/) | 零构建，改 `.md` 刷新即生效 |
| AI | [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) | 子进程调用，直接操作本地文件 |
| 后端 | FastAPI | SSE 流式对话 + 文件读写 API |
| 编辑器 | CodeMirror 5 | Markdown 语法高亮 + 实时编辑 |
| 公式 | KaTeX | LaTeX 数学公式渲染 |

**不需要 API Key**，通过 Claude CLI 使用你的 Claude 订阅认证。

## 快速开始

```bash
npm install -g @anthropic-ai/claude-code && claude   # 安装 Claude CLI 并登录
git clone https://github.com/JZ-Wu/ai-knowledge-base.git && cd ai-knowledge-base
pip install -r server/requirements.txt
python run.py                                         # 打开 http://localhost:8000
```

详细安装说明、项目结构、架构图 → [INSTALL.md](INSTALL.md)

## 内置知识库内容

本项目自带一套完整的 **AI/ML 面试知识体系**（全中文），涵盖：

| 分支 | 内容 |
|------|------|
| **[大模型](大模型/README.md)** | Transformer、Tokenizer、MoE、Scaling Laws、长上下文、SFT/RLHF/DPO/GRPO、推理优化（KV Cache/FlashAttention/量化/vLLM/投机解码）、RAG、VLM、评估与对齐、主流模型全景对比 |
| **[机器学习基础](机器学习基础/README.md)** | 概率统计（8 章）、线性代数（5 篇）、IML 课程（12 讲）、深度学习基础、KL 散度 |
| **[强化学习](强化学习/README.md)** | MDP/Bellman 方程、Q-Learning/DQN、策略梯度/PPO/SAC、Model-Based RL/MuZero、Offline RL/Decision Transformer、MARL |
| **[视觉](视觉/README.md)** | 对比学习与 CLIP、DINO、生成模型（VAE/Diffusion）、3D 稀疏卷积 |
| **[面试手撕](面试手撕/README.md)** | 大模型组件手撕（MHA/RoPE/SwiGLU/RMSNorm/LoRA...）+ 算法手撕 |
| **[论文阅读](论文阅读/README.md)** | 2D/3D 检测分割、生成模型（DDPM/ControlNet）、Point Transformer |

你也可以**清空内容目录**，用它来搭建任何主题的知识库 → [如何用作自己的知识库](INSTALL.md#用作自己的知识库)

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Shift+A` | 打开 AI 侧边栏 |
| `Ctrl+Shift+E` | 打开源码编辑器 |
| `Ctrl+S` | 保存编辑 |

## License

MIT
