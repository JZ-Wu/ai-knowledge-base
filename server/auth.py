"""简单的密码认证中间件：本地访问免密，局域网访问需要密码。"""

import hashlib
import hmac
import pathlib
import secrets

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse

from server.config import ACCESS_PASSWORD

# 用于签名 cookie 的密钥（持久化到文件，重启后 cookie 仍有效）
_SECRET_FILE = pathlib.Path(__file__).parent / ".auth_secret"


def _load_or_create_secret() -> str:
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    secret = secrets.token_hex(16)
    _SECRET_FILE.write_text(secret)
    return secret


_SECRET = _load_or_create_secret()

_LOCAL_IPS = {"127.0.0.1", "::1", "localhost"}

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


def _make_token(password: str) -> str:
    """基于密码和密钥生成认证 token。"""
    return hmac.new(_SECRET.encode(), password.encode(), hashlib.sha256).hexdigest()[:32]


def _is_local(request: Request) -> bool:
    """判断请求是否来自本机。"""
    client = request.client
    if not client:
        return False
    return client.host in _LOCAL_IPS


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 未设置密码 → 全部放行
        if not ACCESS_PASSWORD:
            return await call_next(request)

        # 本地访问 → 放行
        if _is_local(request):
            return await call_next(request)

        # 登录接口本身 → 放行
        if request.url.path == "/api/login":
            return await call_next(request)

        # 检查 cookie
        token = request.cookies.get("kb_auth", "")
        if token and token == _make_token(ACCESS_PASSWORD):
            return await call_next(request)

        # 未认证：API 返回 401，页面返回登录页
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return HTMLResponse(LOGIN_PAGE)
