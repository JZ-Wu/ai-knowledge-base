import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from server.config import DOCS_ROOT
from server.auth import AuthMiddleware
from server.routes import source, chat, edit, stats, auth

app = FastAPI(title="AI Knowledge Base Server")

# Rate limiting（按 IP，分别限制不同端点）
_rate_limit: dict[str, list[float]] = {}
_login_rate_limit: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30
LOGIN_RATE_LIMIT_WINDOW = 300  # 5 分钟窗口
LOGIN_RATE_LIMIT_MAX = 5       # 最多 5 次尝试

# 静态文件服务禁止访问的路径前缀（防止泄露敏感文件）
# 注意：比较时已统一转小写，应对 Windows 大小写不敏感
_BLOCKED_PREFIXES = ("/server/", "/.claude/", "/.git/", "/.github/", "/.env",
                     "/_cli_sandbox/", "/_backup/", "/__pycache__/",
                     "/.auth_secret")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Windows 文件系统大小写不敏感，统一转小写防止 /Server/.env 绕过
    path = request.url.path.lower()

    # 阻止访问敏感路径
    if any(path.startswith(p) or path == p.rstrip("/") for p in _BLOCKED_PREFIXES):
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # 登录端点限速（防暴力破解）
    if path == "/api/login":
        if client_ip not in _login_rate_limit:
            _login_rate_limit[client_ip] = []
        _login_rate_limit[client_ip] = [
            t for t in _login_rate_limit[client_ip] if now - t < LOGIN_RATE_LIMIT_WINDOW
        ]
        if len(_login_rate_limit[client_ip]) >= LOGIN_RATE_LIMIT_MAX:
            return JSONResponse(
                {"error": "Too many login attempts. Try again later."}, status_code=429
            )
        _login_rate_limit[client_ip].append(now)

    # Chat 端点限速
    elif path.startswith("/api/chat"):
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
