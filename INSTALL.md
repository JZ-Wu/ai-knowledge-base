# 安装与配置

## 环境要求

- Python 3.11+
- Node.js 18+（用于 Astro 前端）
- （推荐）[Claude CLI](https://docs.anthropic.com/en/docs/claude-code) —— 用 Claude 订阅，无需 API Key
- 或者：任意 OpenAI 兼容 API 的 API Key（OpenAI / DeepSeek 等）

## 安装步骤

```bash
# 1. （推荐）装 Claude CLI 并登录
npm install -g @anthropic-ai/claude-code
claude  # 首次运行完成认证

# 2. 克隆项目
git clone https://github.com/JZ-Wu/ai-knowledge-base.git
cd ai-knowledge-base

# 3. 装 Python 依赖
pip install -r server/requirements.txt

# 4. 装前端依赖
cd web
npm install
cd ..
```

## 启动方式

### 开发模式（推荐改代码时使用）

开两个终端：

```bash
# 终端 1：FastAPI 后端 API
python run.py

# 终端 2：Astro 前端
cd web
npm run dev
```

浏览器打开 http://localhost:4321/。

开发模式下：

- `4321` 是 Astro 前端。
- `8001` 是 FastAPI 后端 API。
- 前端会直接调用 `http://localhost:8001/api/*`；后端已配置 CORS。
- 不要用 `http://localhost:8001/` 看页面，除非你已经执行过生产构建。

### 生产 / 单端口模式

先构建前端：

```bash
cd web
npm run build
cd ..
```

构建完成后会生成 `web/dist/index.html`。然后启动后端：

```bash
python run.py
```

浏览器打开 http://localhost:8001/。

`server/main.py` 会自动检测 `web/dist`：存在则挂载 Astro 构建产物，不存在则无法在 `8001 /` 提供新版前端首页。

### 启动参数

```bash
python run.py --port 8080        # 指定端口
python run.py --reload           # 开发模式，代码修改自动重启
```

### 首次启动流程

1. 服务起来后会自动探测 `claude` 命令是否可用：
   - 可用 → backend 默认为 `claude_cli`，**直接能用**，不用碰任何配置
   - 不可用 → backend 默认为 `openai_api`，但 API key 为空，顶部会弹横幅
2. 点横幅或访问设置页：
   - 开发模式：`http://localhost:4321/docs/tools/settings.html`
   - 生产模式：`http://localhost:8001/docs/tools/settings.html`
3. 选 backend、填 API key / 改默认模型、（可选）设访问密码 → 保存
4. 回到首页即可使用

设置存到 `server/.settings.json`（已 gitignored），重启后保留。

## 项目结构

```
ai-knowledge-base/
├── run.py                      # 启动脚本
├── web/                        # Astro 前端
│   ├── src/                    # 页面、布局、组件、全局样式
│   ├── public/                 # 静态资源；npm run sync/build 时会镜像 docs/
│   ├── scripts/
│   │   └── sync-content.mjs    # 把 knowledge_bases/ 同步到 Astro content collection
│   ├── astro.config.mjs
│   └── package.json
├── server/
│   ├── main.py                 # FastAPI 应用，挂中间件 + 路由
│   ├── auth.py                 # 安全中间件：路径防护 + 速率限制 + 密码认证
│   ├── config.py               # 真常量（路径、字符上限）
│   ├── models.py               # Pydantic 请求模型
│   ├── routes/
│   │   ├── kb.py               # /api/kbs CRUD + /_sidebar.md 动态生成 + /kb/<slug>/...
│   │   ├── chat.py             # /api/chat SSE 流式对话
│   │   ├── source.py           # /api/page-source 读源码
│   │   ├── edit.py             # /api/suggest-edit + /api/apply-edit
│   │   ├── stats.py            # /api/stats + /api/models + /api/rate-limits
│   │   └── settings.py         # /api/settings + /api/login
│   ├── services/
│   │   ├── kb_service.py       # 多 KB CRUD + 路径解析 + sidebar 生成 + 上传
│   │   └── settings_service.py # .settings.json IO + Claude CLI 自动探测
│   └── backends/
│       ├── __init__.py         # Backend 抽象 + 工厂
│       ├── claude_cli.py       # Claude CLI 子进程实现
│       └── openai_api.py       # OpenAI 兼容 API 实现（带服务器侧工具调用）
├── docs/
│   ├── js/                     # ai-sidebar / editor / settings 前端
│   ├── css/                    # 样式
│   ├── tools/
│   │   ├── pdf-reader.html     # PDF 阅读器
│   │   └── settings.html       # 设置页
│   └── vendor/                 # KaTeX / CodeMirror / PDF.js 等本地资源
├── knowledge_bases/            # 多知识库内容目录
│   └── ai-ml-interview/
│       ├── .kb/meta.json       # KB 元数据（名称、时间戳）
│       ├── README.md
│       └── 大模型/、机器学习基础/...
└── _trash/                     # 内部"回收站"（gitignored，删除的 KB / 旧文件都进这里）
```

## 架构

```
浏览器
  ├── 开发模式: http://localhost:4321/  (Astro dev)
  ├── 生产模式: http://localhost:8001/  (FastAPI serve web/dist)
  ├── Astro 渲染 /kb/<slug>/<path> 页面
  ├── AI 侧边栏（选中 → 对话 → 编辑文件）
  ├── KB 顶栏入口 → 设置页管理 KB（新建/改名/删/上传 KB）
  ├── 编辑器（CodeMirror，Ctrl+Shift+E）
  └── fetch → /api/*

FastAPI 后端（默认 8001）
  ├── 中间件 SecurityMiddleware：路径防护 + 速率限制 + ACCESS_PASSWORD 认证
  ├── CORS：允许 Astro dev(:4321) 调 /api/*
  ├── /api/settings        GET/PUT 运行时设置（脱敏后暴露给前端）
  ├── /api/settings/check  GET     未登录也能访问，用于首次启动检测
  ├── /api/login           POST    登录
  ├── /api/kbs             GET/POST/PATCH/DELETE 多 KB CRUD
  ├── /api/kbs/<slug>/upload POST  上传文件/文件夹
  ├── /api/chat            POST    SSE 流式对话 → 走当前 backend
  ├── /api/page-source     GET     读 markdown 源码
  ├── /api/apply-edit      POST    保存源码编辑器修改
  ├── /api/stats           GET     总文件数 + 总字数
  ├── /api/models          GET     当前 backend 的模型列表
  ├── /_sidebar.md         GET     动态生成 Docsify sidebar
  ├── /kb/<slug>/...       GET     静态文件（限制在该 KB 目录内）
  └── /*                   生产模式静态资源（web/dist，catch-all）
```

### Backend 切换

两个 backend 都实现同一个 Protocol（`server/backends/__init__.py`）：

```python
class Backend(Protocol):
    name: str
    def list_models(self) -> list[dict]: ...
    def status(self) -> dict: ...
    def stream_chat(self, *, page_path, page_content, selected_text, messages,
                    model="", thinking=False, images=None, session_id="") -> Generator: ...
    def suggest_edit(self, *, page_content, instruction, chat_context="") -> str: ...
```

`get_active_backend()` 按 `settings_service.active_backend_name()` 返回 `ClaudeCLIBackend()` 或 `OpenAIAPIBackend()`，路由层只调 backend 接口。

`claude_cli`：工作目录切到具体 KB 根，CLI 工具白名单只放 `Read,Edit,Write,Glob,Grep`，图片上传走 `.tmp_images/` 中转。

`openai_api`：服务器侧实现 `read_markdown / replace_markdown / write_markdown / search_markdown / list_markdown_files`，路径强制限制在 `knowledge_bases/<slug>/` 内、仅 `.md` 后缀、文件名不含 secret/key/token 等字段。

## 用作自己的知识库

1. Fork 或 clone 本项目
2. KB 按钮 → 新建一个空知识库（或保留 `ai-ml-interview` 当 demo）
3. 拖文件夹到 KB Manager 上传
4. 开发模式用 `python run.py` + `cd web && npm run dev`；生产模式先 `cd web && npm run build` 再 `python run.py`

如果想清空内置的 `ai-ml-interview` 改成自己的：直接在 KB 管理面板点"删"，整个目录会被搬到 `_trash/`（gitignored，不会推到远端）。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Shift+A` | 打开/关闭 AI 侧边栏 |
| `Ctrl+Shift+E` | 打开/关闭源码编辑器 |
| `Ctrl+S`（编辑器内） | 保存文件 |
| `Escape` | 关闭当前面板 |
| `Ctrl+F`（编辑器内） | 搜索 |

## 部署给别人用

1. 设访问密码：设置页底部填密码 → 保存。本机访问仍直通，外网访问会跳登录页。
2. 先在 `web/` 执行 `npm run build`，再启动 `python run.py`。
3. 反向代理（nginx 等）走 8001 端口即可，**不要**把 `/server/`、`/knowledge_bases/`、`/_trash/`、`/_backup/` 暴露给外网（中间件已经拦了，但反向代理层也建议加白名单）。
4. 不要把 `server/.settings.json` 和 `server/.auth_secret` 推到 git（已在 `.gitignore`）。
