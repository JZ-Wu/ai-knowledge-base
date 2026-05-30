# Claude Code 项目规则

## 项目说明

这是一个 **Markdown 多知识库** 项目。内容在 `knowledge_bases/<slug>/`；
前端是 **Astro 静态站**（`web/`，构建产物 `web/dist/`），由 **FastAPI（8001）** 同源托管。
前后端 / Claude skills 框架代码与内容分离（见下「框架共享」）。

### 操作权限

- 允许所有文件的读取、编辑、创建、搜索
- 允许使用 Glob、Grep、Read、Edit、Write 工具，无需额外确认
- 允许使用 Bash 执行 ls、mkdir、git mv 等文件管理命令
- **禁止** `rm -rf` 或任何删除文件/目录的命令；要"删"就 `git mv` 到 `_trash/`

### KB 布局约定

- **所有 KB 内容放在 `knowledge_bases/<slug>/`**，slug 命名 `[A-Za-z0-9_-]{1,64}`。
- 每个 KB 自带 `README.md` + `_sidebar.md`；新建 KB 必须同步创建这两个文件。
- URL 模式：`/kb/<slug>/<path>`（Astro 路由，`.md` → 目录式 `/`）。
- 站名取自根 `README.md` 的 `# 一级标题`，**不写死在框架里**。

### 内容规范

- 知识库 .md 全部用中文，Markdown 格式
- 新增 .md 后更新所在目录的 `README.md` 索引
- 交叉引用用**绝对路径** `/kb/<slug>/...`（Astro `trailingSlash:'always'`，相对路径易 404）
- 侧栏由 Astro **构建时**解析各 KB 的 `knowledge_bases/<slug>/_sidebar.md` 生成；
  没有 `_sidebar.md` 时按文件夹自动建树。新增 paper card / 文档 → 同步更新对应 `_sidebar.md` 和相关 README。
- 单目录文件超过 15 个时拆子目录
- 涉及版本/时间敏感内容标注日期

### 后端架构（修改后端代码时务必读懂）

- 两个 backend 实现 `server/backends/__init__.py` 里的 `Backend` Protocol，路由层只调接口
- 运行时设置（API key / 默认模型 / thinking 强度 / ACCESS_PASSWORD）走 `server/services/settings_service.py` 的 `.settings.json`，**不要再加 env var 配置项**
- 多 KB CRUD / 路径解析都在 `server/services/kb_service.py`
- 路径安全永远在 `server/auth.py` 的中间件里做（已拦 `/server/`、`/knowledge_bases/`、`/_trash/` 等）

### 框架共享（kb-core）

本 KB 的**前端（`web/`）/ 后端（`server/`）/ Claude skills / `docs/{js,css,vendor,tools}`** 来自 sibling repo
`kb-core`（GitHub `JZ-Wu/kb-core`，本地 `../kb-core`）。

**改框架代码时**：先在 `../kb-core` 改、commit + push，再回本 KB 跑 `python scripts/sync_core.py`，然后 commit。
**不要直接改**本 KB 里的 `web/src/`、`web/scripts/`、`docs/js/`、`docs/css/`、`docs/vendor/`、`docs/tools/`、`server/`、`.claude/skills/`、`scripts/sync_core.py` —— 下次 sync 会覆盖。

改 KB 自己的内容（`knowledge_bases/`、根 `README.md`、各 KB 的 `_sidebar.md`、`CLAUDE.md`）则直接改即可。
`python scripts/sync_core.py --dry-run` 预览同步变更。

构建前端：`cd web && npm install && npm run build`（产物 `web/dist/` 已 gitignore；`web/scripts/sync-content.mjs` 在 build 前把 `knowledge_bases/` 同步进 `web/src/content/docs/kb/`）。

### Git 规范

- 每完成一组修改后自动 `git add` + `git commit` + `git push`，不等用户要求
- commit message 用英文，简洁描述改动
- 每次修改主动告知改了哪些文件
