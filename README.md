# AI 知识库

<div id="kb-stats" class="kb-stats"></div>

一个 Markdown 知识库，但 AI 能直接读和改你的 `.md` 文件。

不是聊天框里返回一段文本让你自己粘回去 —— AI 拿到的工具是 Read / Edit / Write / Glob / Grep（同一套 Claude Code 工具），它会在原位修改你选中的段落，写入磁盘，刷新页面就看到。

<!-- ![demo](docs/assets/demo.gif) -->

## 安装

```bash
git clone https://github.com/JZ-Wu/ai-knowledge-base
cd ai-knowledge-base
pip install -r server/requirements.txt
python run.py            # http://localhost:8001
```

启动时自动探测 `claude` 命令是否在 PATH 上：

- **找到了** → 默认用 Claude CLI 后端，开箱即用（享受 Claude 订阅，不需要 API key）
- **没找到** → 顶栏弹横幅，引导到 `/docs/tools/settings.html` 设置页填任意 OpenAI 兼容 API 的 key（OpenAI、DeepSeek、Qwen 等）

想用 Claude CLI 但还没装：`npm i -g @anthropic-ai/claude-code && claude` 一次性登录就行。

## 几个常见用法

### 让 AI 在原文里加内容

打开 `大模型/MoE.md`，正文有这句：

> MoE 通过门控网络把 token 路由到不同专家，常见做法是 top-k 选择。

选中它，点屏幕右下角浮动按钮，对 AI 说：

> 在这句后面加 3 个具体例子：Switch Transformer / Mixtral / DeepSeekMoE，每个写清楚 k 值和总专家数。

AI 会先 Read 这个文件定位你的句子，然后用 Edit 工具把这一段扩成你要的格式。右下角会弹 `Edit: 大模型/MoE.md` 提示，刷新即可。

### 让 AI 帮你重写 / 翻译

选中一段，对 AI 说：

```
帮我把这段改得更适合初学者读
把这章翻译成英文
检查这段公式有没有错，错了直接改
```

AI 会调 Edit / Write 工具替换原文（写之前自动生成 `.bak` 备份，写完了告诉你改了哪些位置）。

### 让 AI 看截图

直接 Ctrl+V 把截图粘到 AI 输入框，或者拖图进去。AI 收到的是真图片，不是 base64 字符串塞 prompt 里 —— 因此能正确理解你画的架构图、表格、错误截图。

### 读 PDF 时问 AI

`docs/tools/pdf-reader.html` 是内置的 PDF.js 阅读器。选中 PDF 里的段落，AI 侧边栏的上下文会自动带上你选的文字，可以问 "这段公式什么意思"、"这跟我笔记里的 X 概念有什么联系"。

### 源码编辑

`Ctrl+Shift+E` 打开 CodeMirror 源码编辑器，自动定位到你正在读的位置，Markdown 高亮，`Ctrl+S` 保存，`Ctrl+F` 搜索。适合自己改格式 / 公式这种 AI 帮不上的小活。

## 多知识库

`knowledge_bases/<slug>/` 下每个子目录是一个独立 KB，对应 URL `/kb/<slug>/`，sidebar 自动分组生成。比如：

```
knowledge_bases/
├── ai-ml-interview/          → /kb/ai-ml-interview/
│   ├── 大模型/
│   ├── 强化学习/
│   └── ...
└── research-notes/           → /kb/research-notes/
    └── ...
```

顶栏 `KB ▾` 一键切换。设置页里：

- **新建**：填名字和 URL slug（可选，留空自动生成）
- **改名**：只改显示名，slug 不变（URL 不破坏）
- **删除**：整个目录搬到 `_trash/`（gitignored，本地保留，远端不会留）
- **上传**：选文件或文件夹，文件夹会保留目录结构

自带 `ai-ml-interview` 是一套 AI/ML 面试知识体系（166 篇 / 121 万字），覆盖大模型、机器学习基础、强化学习、视觉、具身智能、CUDA 编程、分布式训练、行业动态、面试手撕。想清空换成自己的内容：删了它，建个新 KB，拖文件夹上传，三步搞定。

## 后端配置

设置页（`/docs/tools/settings.html`）一个页面管所有：

- **后端切换**：Claude CLI ↔ OpenAI 兼容 API
- **Claude CLI**：路径（留空自动从 PATH 找）、默认模型（Opus 4.7 / Sonnet 4.6 / Haiku 4.5）
- **OpenAI 兼容 API**：可加多个模型 profile（名字 / 模型 ID / base URL / API key / 上下文窗口）、设默认模型、是否允许前端切换、是否开工具调用
- **访问密码**：填明文，落盘前自动 scrypt 哈希；本机访问免登录，外网走 `/api/login` 限速

设置存到 `server/.settings.json`（已 gitignore）。修改即时生效，不用重启。

## 部署给别人用

打算挂局域网或公网给同事 / 同学看时：

1. 设置页底部填访问密码 → 保存。本机访问仍然直通，外网开 5 次 / 5 分钟登录限速
2. 反向代理（nginx 等）走 server 端口，**不要**把 `/server/`、`/knowledge_bases/`、`/_trash/`、`/_backup/`、`/.git/` 等暴露给外网 —— 中间件已经 403 拦截，反向代理那层也加一道更稳
3. `server/.settings.json` 和 `server/.auth_secret` **不要**进 git（默认 `.gitignore` 已包含）

工具调用受路径白名单约束：AI 只能操作 `.md` 文件、只能在 KB 目录内、不能 `../` 上溯、含 `secret/key/token/credential/password` 字样的文件名直接拒。

## 项目结构

```
ai-knowledge-base/
├── index.html                 Docsify 入口 + AI 侧边栏 + 顶栏按钮
├── server/
│   ├── main.py                FastAPI app
│   ├── auth.py                SecurityMiddleware（路径白名单 + 限速 + scrypt 登录）
│   ├── config.py              真常量（DOCS_ROOT、KB_ROOT、EXTERNAL_MOUNTS）
│   ├── routes/                /api/chat、/api/kbs、/api/settings、/_sidebar.md 等
│   ├── services/              kb_service（多 KB CRUD）、settings_service（.settings.json IO）
│   └── backends/              Backend 抽象 + claude_cli / openai_api 两个实现
├── docs/                      前端：JS / CSS / 第三方库 / 设置页 / PDF 阅读器
└── knowledge_bases/<slug>/    内容
```

详细架构、EXTERNAL_MOUNTS 用法、备份策略等见 [INSTALL.md](INSTALL.md)。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Shift+A` | 打开 / 关闭 AI 侧边栏 |
| `Ctrl+Shift+E` | 打开 / 关闭源码编辑器 |
| `Ctrl+S` | 保存（编辑器内） |
| `Ctrl+F` | 搜索（编辑器内） |
| `Esc` | 关闭当前面板 |

## 技术栈

Docsify · FastAPI · Claude CLI / OpenAI 兼容 API · CodeMirror 5 · KaTeX · PDF.js · Python stdlib scrypt

## License

MIT
