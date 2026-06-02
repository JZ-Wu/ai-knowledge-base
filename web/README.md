# Web Frontend

这是 AI 知识库的 Astro 前端。FastAPI 后端仍在项目根目录的 `server/`，默认端口 `8001`；前端开发服务器默认端口 `4321`。

## 开发模式

从项目根目录开两个终端：

```bash
# 终端 1：后端 API
python run.py

# 终端 2：前端
cd web
npm install
npm run dev
```

访问：

```text
http://localhost:4321/
```

开发模式下，前端会调用 `http://localhost:8001/api/*`。后端已经配置 CORS 允许 `localhost:4321`。

## 生产构建

```bash
cd web
npm install
npm run build
cd ..
python run.py
```

访问：

```text
http://localhost:8001/
```

`npm run build` 会先执行 `scripts/sync-content.mjs`，把 `knowledge_bases/` 同步到 `web/src/content/docs/`，并把根目录 `docs/` 镜像到 `web/public/docs/`。构建完成后生成 `web/dist/`，FastAPI 会自动挂载它作为单端口前端。

如果 `http://localhost:8001/` 返回 404，通常是因为还没有生成 `web/dist/index.html`。

## 常用命令

| Command | Action |
| --- | --- |
| `npm install` | 安装前端依赖 |
| `npm run sync` | 同步知识库内容和公共静态资源 |
| `npm run dev` | 同步内容后启动 Astro dev server |
| `npm run build` | 同步内容并构建到 `web/dist/` |
| `npm run preview` | 预览已构建的前端产物 |

## 目录

```text
web/
├── src/
│   ├── pages/          # Astro 页面
│   ├── layouts/        # 页面布局
│   ├── components/     # Topbar / Sidebar / TOC 等组件
│   ├── styles/         # 全局样式
│   └── content/        # sync-content 生成的知识库内容
├── public/             # sync-content 镜像的 docs/ 静态资源
├── scripts/
│   └── sync-content.mjs
├── astro.config.mjs
└── package.json
```
