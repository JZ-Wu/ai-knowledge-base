# 安装与配置

## 环境要求

- Python 3.11+
- Node.js 18+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code)（需要 Claude Pro/Max 订阅）

## 安装步骤

```bash
# 1. 安装 Claude CLI 并登录
npm install -g @anthropic-ai/claude-code
claude  # 首次运行完成认证

# 2. 克隆项目
git clone https://github.com/JZ-Wu/ai-knowledge-base.git
cd ai-knowledge-base

# 3. 安装 Python 依赖
pip install -r server/requirements.txt

# 4. 启动
python run.py
```

浏览器打开 http://localhost:8000 即可。

### 启动参数

```bash
python run.py --port 8080        # 指定端口
python run.py --reload           # 开发模式，代码修改自动重启
```

## 项目结构

```
ai-knowledge-base/
├── index.html                  # Docsify 入口 + AI 侧边栏 + 编辑器 HTML
├── _sidebar.md                 # Docsify 侧边栏导航
├── run.py                      # 启动脚本
├── server/                     # FastAPI 后端
│   ├── main.py                 # 应用入口，挂载 API 路由 + 静态文件
│   ├── config.py               # 配置（Claude CLI 路径、知识库根目录）
│   ├── models.py               # Pydantic 请求/响应模型
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

1. Fork 或 clone 本项目
2. 删除现有的知识库内容目录（`大模型/`、`机器学习基础/` 等）
3. 创建自己的目录和 Markdown 文件
4. 编辑 `_sidebar.md` 配置导航
5. 修改 `index.html` 中的 `name: 'AI 知识库'` 为你的名称
6. `python run.py` 启动即可

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Shift+A` | 打开/关闭 AI 侧边栏 |
| `Ctrl+Shift+E` | 打开/关闭源码编辑器 |
| `Ctrl+S`（编辑器内） | 保存文件 |
| `Escape` | 关闭当前面板 |
| `Ctrl+F`（编辑器内） | 搜索 |
