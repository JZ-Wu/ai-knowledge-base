import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from server.config import DOCS_ROOT
from server.auth import AuthMiddleware
from server.routes import source, chat, edit, stats, auth

app = FastAPI(title="AI Knowledge Base Server")

# Rate limiting
_rate_limit: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/chat"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        if client_ip not in _rate_limit:
            _rate_limit[client_ip] = []
        _rate_limit[client_ip] = [t for t in _rate_limit[client_ip] if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_limit[client_ip]) >= RATE_LIMIT_MAX:
            return JSONResponse({"error": "Rate limit exceeded. Try again later."}, status_code=429)
        _rate_limit[client_ip].append(now)
    return await call_next(request)

# 认证中间件（本地免密，局域网需密码）
app.add_middleware(AuthMiddleware)

# API 路由
app.include_router(auth.router, prefix="/api")
app.include_router(source.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(edit.router, prefix="/api")
app.include_router(stats.router, prefix="/api")

# 静态文件 (知识库目录) — 必须放在最后，作为 catch-all
app.mount("/", StaticFiles(directory=str(DOCS_ROOT), html=True), name="static")
