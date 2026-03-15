# AI 知识库

> AI 原生的个人知识库。没有多余功能 — 选中文字问 AI，AI 直接帮你改。

一堆 Markdown 文件 + 一个 AI 侧边栏 + 一个源码编辑器，没了。

基于 [Docsify](https://docsify.js.org/) 渲染，无需构建步骤，改完 `.md` 刷新即生效。AI 通过 [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) 驱动，可以直接读取和编辑知识库文件。

## 功能

- **AI 对话**：右上角 AI 按钮或 `Ctrl+Shift+A` 打开侧边栏，与 Claude 对话
- **选中提问**：选中页面任意文字，点击浮动按钮直接带上下文提问
- **AI 编辑**：要求 AI 修改内容时，它会直接编辑对应的 Markdown 文件
- **源码编辑**：右上角 Edit 按钮或 `Ctrl+Shift+E`，CodeMirror 编辑器自动定位到当前阅读位置
- **模型切换**：支持 Opus / Sonnet / Haiku，可开关 Thinking 模式
- **图片输入**：对话框支持粘贴截图或上传图片
- **数学公式**：KaTeX 渲染，支持 `$...$` 和 `$$...$$`
- **全文搜索**：Docsify 内置搜索
- **对话历史**：同标签页内刷新不丢失（sessionStorage）

## 安装

### 环境要求

- Python 3.11+
- Node.js 18+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code)（需要 Claude Pro/Max 订阅）

### 1. 安装 Claude CLI

```bash
npm install -g @anthropic-ai/claude-code
```

安装后运行 `claude` 完成认证登录。

### 2. 克隆项目

```bash
git clone <your-repo-url> 知识库
cd 知识库
```

### 3. 安装 Python 依赖

```bash
pip install -r server/requirements.txt
```

### 4. 启动

```bash
python run.py
```

浏览器打开 http://localhost:8000 即可。

启动参数：

```bash
python run.py --port 8080        # 指定端口
python run.py --reload           # 开发模式，代码修改自动重启
```

## 项目结构

```
知识库/
├── index.html                  # Docsify 入口 + AI 侧边栏 + 编辑器 HTML
├── _sidebar.md                 # Docsify 侧边栏导航
├── run.py                      # 启动脚本
├── server/                     # FastAPI 后端
│   ├── main.py                 # 应用入口，API 路由 + 静态文件
│   ├── config.py               # 配置（Claude CLI 路径、知识库根目录）
│   ├── models.py               # 请求/响应模型
│   ├── routes/
│   │   ├── chat.py             # POST /api/chat — SSE 流式对话
│   │   ├── edit.py             # POST /api/apply-edit — 写入文件
│   │   └── source.py           # GET /api/page-source — 读取源码
│   └── services/
│       ├── claude_service.py   # Claude CLI 子进程封装
│       └── file_service.py     # 文件读写 + 路径安全校验
├── docs/
│   ├── js/ai-sidebar.js        # AI 侧边栏前端逻辑
│   ├── js/editor.js            # CodeMirror 源码编辑器
│   └── css/ai-sidebar.css      # 样式
└── 大模型/、机器学习基础/...      # 知识库内容（Markdown 文件）
```

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Shift+A` | 打开/关闭 AI 侧边栏 |
| `Ctrl+Shift+E` | 打开/关闭源码编辑器 |
| `Ctrl+S`（编辑器内） | 保存文件 |
| `Escape` | 关闭当前面板 |
| `Ctrl+F`（编辑器内） | 搜索 |

## 架构

```
浏览器 (localhost:8000)
  ├── Docsify 前端渲染 (index.html + .md 文件)
  ├── AI 侧边栏 (选中文字 → 对话 → AI 编辑文件)
  ├── 源码编辑器 (CodeMirror, 实时编辑 Markdown)
  └── fetch → /api/*

FastAPI 后端 (同一端口)
  ├── /api/chat         POST  SSE 流式对话（调用 Claude CLI 子进程）
  ├── /api/page-source  GET   获取页面 Markdown 源码
  ├── /api/apply-edit   POST  写入修改（自动创建 .bak 备份）
  └── /*                静态文件服务
```

AI 后端通过 Claude CLI 子进程调用，工作目录设为知识库根目录，因此 Claude 可以直接使用 Read/Edit/Write 等工具操作知识库文件。不需要 API Key，使用 Claude 订阅认证。

## 用作自己的知识库

1. 删除现有的知识库内容目录（`大模型/`、`机器学习基础/` 等）
2. 创建自己的目录和 Markdown 文件
3. 编辑 `_sidebar.md` 配置导航
4. 修改 `index.html` 中的 `name: 'AI 知识库'` 为你的名称
5. 启动即可

## 知识树

| 分支 | 说明 |
|------|------|
| [大模型](大模型/README.md) | 基础理论、训练微调、推理优化、RAG、多模态、评估与对齐 |
| [机器学习基础](机器学习基础/README.md) | 数学基础、经典 ML 算法、深度学习基础、PyTorch |
| [视觉](视觉/README.md) | 表示学习、3D视觉、三维重建、SLAM |
| [具身智能](具身智能/README.md) | Embodied AI、机器人操控、仿真环境、VLA 模型 |
| [CUDA编程](CUDA编程/README.md) | GPU 架构、CUDA 编程、内存层次、性能优化、Triton |
| [分布式训练](分布式训练/README.md) | 数据并行、模型并行、混合精度、DeepSpeed/Megatron |
| [面试手撕](面试手撕/README.md) | 大模型组件手撕 + LeetCode 算法 |
| [论文阅读](论文阅读/README.md) | 按方向分类的详细阅读笔记 |
| [经典论文](经典论文/README.md) | 经典论文索引 |
| [行业动态](行业动态/README.md) | 重要论文解读、产品发布、技术趋势 |

## 维护规范

1. **新增内容**：在对应目录下创建 `.md` 文件，并更新该目录的 `README.md` 索引
2. **结构调整**：当某个目录下文件超过 15 个时，拆分为子目录
3. **交叉引用**：使用相对路径链接
4. **版本标注**：涉及特定模型版本或 API 的内容，标注版本和日期
