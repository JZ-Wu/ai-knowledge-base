from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.auth import SecurityMiddleware
from server.config import DOCS_ROOT, EXTERNAL_MOUNTS
from server.routes import chat, edit, kb, settings, source, stats

app = FastAPI(
    title="AI Knowledge Base Server",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# 单一安全中间件：路径防护 + 速率限制 + 认证（顺序敏感）。
app.add_middleware(SecurityMiddleware)

# CORS：让 Astro dev(:4321) 能跨源调 :8001 的 /api/*（带 cookie）。在 SecurityMiddleware
# 之后注册 → CORS 处于更外层，OPTIONS 预检与被 SecurityMiddleware 拒绝的 401/403 也能带上
# CORS 头，浏览器才不会把它们当成网络错误。生产同源(:8001)时无副作用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321", "http://127.0.0.1:4321",
        "http://localhost:8001", "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(settings.router, prefix="/api")  # 含 /api/login + /api/settings/*
app.include_router(source.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(edit.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(kb.router)                       # 含 /api/kbs/* + /_sidebar.md + /kb/<slug>/...

# 静态前端根：有 web/dist（Astro 生产构建）就 serve 它（新版 UI），否则回退 Docsify DOCS_ROOT。
_web_dist = DOCS_ROOT / "web" / "dist"
_astro_mode = _web_dist.exists()
_frontend_root = _web_dist if _astro_mode else DOCS_ROOT

# EXTERNAL_MOUNTS：只在 Docsify 模式挂载原始文件夹（Docsify 客户端拉原始 .md 渲染）。
# Astro 模式下外部内容已被 sync-content 预渲染进 web/dist（含图片/PDF），原始文件夹挂载
# 反而会 shadow 掉渲染好的目录式路由（请求 /external-reports/.../x/ → 原始文件夹只有
# x.md、没有 x/ 目录 → 404）。所以 Astro 模式跳过，交给下面的 catch-all serve web/dist。
if not _astro_mode:
    # 在 catch-all 之前注册，让 URL 前缀优先匹配。
    for _prefix, _fs_path in EXTERNAL_MOUNTS.items():
        app.mount(
            f"/{_prefix}",
            StaticFiles(directory=str(_fs_path), html=True),
            name=f"ext-{_prefix}",
        )

# 静态文件（catch-all）。
app.mount("/", StaticFiles(directory=str(_frontend_root), html=True), name="static")
