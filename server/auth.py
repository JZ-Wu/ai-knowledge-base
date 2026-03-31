"""统一安全中间件：路径防护 + 速率限制 + 密码认证，合并为单一中间件保证执行顺序。"""

import hashlib
import hmac
import os
import pathlib
import re
import secrets
import time
import urllib.parse

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse

from server.config import ACCESS_PASSWORD

# ── Cookie 签名密钥（持久化，重启后 cookie 仍有效）──
_SECRET_FILE = pathlib.Path(__file__).parent / ".auth_secret"


def _load_or_create_secret() -> str:
    if _SECRET_FILE.exists():
        secret = _SECRET_FILE.read_text().strip()
        if len(secret) >= 32:
            return secret
    secret = secrets.token_hex(32)  # 256-bit
    _SECRET_FILE.write_text(secret)
    try:
        os.chmod(str(_SECRET_FILE), 0o600)
    except OSError:
        pass
    return secret


_SECRET = _load_or_create_secret()

# 只信任直连 socket IP，绝不信任 X-Forwarded-For
_LOCAL_IPS = frozenset({"127.0.0.1", "::1"})

# ── 路径安全 ──
_BLOCKED_PREFIXES = (
    "/server/", "/.claude/", "/.git/", "/.github/", "/.env",
    "/_cli_sandbox/", "/_backup/", "/__pycache__/", "/.auth_secret",
    "/.venv/", "/venv/", "/.vscode/", "/.idea/",
)
_DANGEROUS_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _normalize_path(raw: str) -> str | None:
    """全面规范化 URL 路径。返回小写路径，含危险内容返回 None。"""
    if _DANGEROUS_CHARS.search(raw):
        return None
    # 双重 URL 解码（防 %252F 等双重编码）
    decoded = urllib.parse.unquote(urllib.parse.unquote(raw))
    if _DANGEROUS_CHARS.search(decoded):
        return None
    # 反斜杠 → 正斜杠（Windows）
    decoded = decoded.replace("\\", "/")
    # 折叠连续斜杠
    while "//" in decoded:
        decoded = decoded.replace("//", "/")
    # 解析 . 和 ..
    parts = decoded.split("/")
    resolved: list[str] = []
    for p in parts:
        if p in ("", "."):
            continue
        elif p == "..":
            if resolved:
                resolved.pop()
        else:
            resolved.append(p)
    normalized = "/" + "/".join(resolved)
    return normalized.lower()


def _is_path_blocked(path: str) -> bool:
    """检查规范化后的路径是否被禁止。"""
    for prefix in _BLOCKED_PREFIXES:
        if path.startswith(prefix) or path == prefix.rstrip("/"):
            return True
    # 阻止 .py/.pyc 文件
    if path.endswith(".py") or path.endswith(".pyc"):
        return True
    # 阻止所有隐藏文件/目录（/. 开头的段）
    if "/." in path:
        return True
    return False


# ── Token 生成与验证（时序安全）──
def _make_token(password: str) -> str:
    """基于密码和密钥生成认证 token（完整 HMAC-SHA256）。"""
    return hmac.new(_SECRET.encode(), password.encode(), hashlib.sha256).hexdigest()


def verify_token(token: str) -> bool:
    """常量时间验证 token，防止时序攻击。"""
    if not token or not ACCESS_PASSWORD:
        return False
    return hmac.compare_digest(token, _make_token(ACCESS_PASSWORD))


def verify_password(candidate: str) -> bool:
    """常量时间验证密码，防止时序攻击。"""
    if not candidate or not ACCESS_PASSWORD:
        return False
    return hmac.compare_digest(candidate, ACCESS_PASSWORD)


# ── 速率限制 ──
_chat_rate: dict[str, list[float]] = {}
_login_rate: dict[str, list[float]] = {}
_MAX_TRACKED_IPS = 10000

CHAT_RATE_WINDOW = 60
CHAT_RATE_MAX = 30
LOGIN_RATE_WINDOW = 300   # 5 分钟
LOGIN_RATE_MAX = 5        # 最多 5 次


def _check_rate(store: dict, ip: str, window: int, limit: int) -> bool:
    """返回 True 表示被限制。"""
    now = time.time()
    # 防内存泄漏
    if len(store) > _MAX_TRACKED_IPS:
        expired = [k for k, v in store.items() if not v or now - v[-1] > window]
        for k in expired:
            del store[k]
    if ip not in store:
        store[ip] = []
    store[ip] = [t for t in store[ip] if now - t < window]
    if len(store[ip]) >= limit:
        return True
    store[ip].append(now)
    return False


# ── 本地判断 ──
def _is_local(request: Request) -> bool:
    """只看直连 socket IP，绝不信任代理头。"""
    client = request.client
    if not client:
        return False
    return client.host in _LOCAL_IPS


# ── 登录页 ──
LOGIN_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>登录 - AI 知识库</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; background: #f0f2f5; font-family: -apple-system, sans-serif;
  }
  .login-card {
    background: #fff; border-radius: 12px; padding: 40px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1); width: 360px; text-align: center;
  }
  .login-card h1 { font-size: 22px; color: #333; margin-bottom: 8px; }
  .login-card p { font-size: 14px; color: #888; margin-bottom: 24px; }
  .login-card input {
    width: 100%; padding: 12px 16px; border: 1px solid #ddd; border-radius: 8px;
    font-size: 15px; outline: none; transition: border-color 0.2s;
  }
  .login-card input:focus { border-color: #3F51B5; }
  .login-card button {
    width: 100%; padding: 12px; margin-top: 16px; border: none; border-radius: 8px;
    background: #3F51B5; color: #fff; font-size: 15px; cursor: pointer;
    transition: background 0.2s;
  }
  .login-card button:hover { background: #303F9F; }
  .error { color: #e53935; font-size: 13px; margin-top: 10px; display: none; }
</style>
</head>
<body>
<div class="login-card">
  <h1>AI 知识库</h1>
  <p>请输入访问密码</p>
  <form id="f">
    <input type="password" id="pw" placeholder="密码" autofocus>
    <button type="submit">登录</button>
    <div class="error" id="err">密码错误</div>
  </form>
</div>
<script>
document.getElementById('f').onsubmit = function(e) {
  e.preventDefault();
  fetch('/api/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({password: document.getElementById('pw').value})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) { location.reload(); }
    else { document.getElementById('err').style.display = 'block'; }
  });
};
</script>
</body>
</html>"""


# ── 统一安全中间件 ──
class SecurityMiddleware(BaseHTTPMiddleware):
    """合并路径防护 + 速率限制 + 认证，保证执行顺序。"""

    async def dispatch(self, request: Request, call_next):
        # 1. 路径规范化与安全检查
        normalized = _normalize_path(request.url.path)
        if normalized is None:
            return JSONResponse({"error": "Bad request"}, status_code=400)
        if _is_path_blocked(normalized):
            return JSONResponse({"error": "Forbidden"}, status_code=403)

        # 2. 速率限制
        client_ip = request.client.host if request.client else "unknown"
        if normalized == "/api/login":
            if _check_rate(_login_rate, client_ip, LOGIN_RATE_WINDOW, LOGIN_RATE_MAX):
                return JSONResponse(
                    {"error": "Too many login attempts. Try again later."},
                    status_code=429,
                    headers={"Retry-After": str(LOGIN_RATE_WINDOW)},
                )
        elif normalized.startswith("/api/chat"):
            if _check_rate(_chat_rate, client_ip, CHAT_RATE_WINDOW, CHAT_RATE_MAX):
                return JSONResponse(
                    {"error": "Rate limit exceeded. Try again later."},
                    status_code=429,
                )

        # 3. 认证
        if not ACCESS_PASSWORD:
            return await call_next(request)
        if _is_local(request):
            return await call_next(request)
        if normalized == "/api/login":
            return await call_next(request)

        token = request.cookies.get("kb_auth", "")
        if verify_token(token):
            return await call_next(request)

        # 未认证
        if normalized.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return HTMLResponse(LOGIN_PAGE)
