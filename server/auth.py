"""统一安全中间件：路径防护 + 速率限制 + 密码认证，合并为单一中间件保证执行顺序。

settings.json 里的 `access_password` 字段存的是 scrypt 哈希（格式
`scrypt$N$r$p$salt_b64$hash_b64`），不是明文。第一次启动若发现是裸明文，
settings_service.write() 会在落盘前自动 rehash 升级。
"""

import base64
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

from server.services import settings_service

# ── Cookie 签名密钥（持久化，重启后 cookie 仍有效）──
_SECRET_FILE = pathlib.Path(__file__).parent / ".auth_secret"


def _load_or_create_secret() -> str:
    if _SECRET_FILE.exists():
        secret = _SECRET_FILE.read_text().strip()
        if len(secret) >= 32:
            return secret
    secret = secrets.token_hex(32)
    _SECRET_FILE.write_text(secret)
    try:
        os.chmod(str(_SECRET_FILE), 0o600)
    except OSError:
        pass
    return secret


_SECRET = _load_or_create_secret()

_LOCAL_IPS = frozenset({"127.0.0.1", "::1"})

# ── 路径安全 ──
_BLOCKED_PREFIXES = (
    "/server/", "/.claude/", "/.git/", "/.github/", "/.env",
    "/knowledge_bases/", "/_trash/", "/_cli_sandbox/", "/_backup/",
    "/__pycache__/", "/.auth_secret",
    "/.venv/", "/venv/", "/.vscode/", "/.idea/",
)
_DANGEROUS_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _normalize_path(raw: str) -> str | None:
    if _DANGEROUS_CHARS.search(raw):
        return None
    decoded = urllib.parse.unquote(urllib.parse.unquote(raw))
    if _DANGEROUS_CHARS.search(decoded):
        return None
    decoded = decoded.replace("\\", "/")
    while "//" in decoded:
        decoded = decoded.replace("//", "/")
    parts = decoded.split("/")
    resolved: list[str] = []
    for p in parts:
        if p in ("", "."):
            continue
        elif p == "..":
            if resolved:
                resolved.pop()
        else:
            resolved.append(p.rstrip(". "))
    normalized = "/" + "/".join(resolved)
    return normalized.lower()


def _is_path_blocked(path: str) -> bool:
    for prefix in _BLOCKED_PREFIXES:
        if path.startswith(prefix) or path == prefix.rstrip("/"):
            return True
    if path.endswith(".py") or path.endswith(".pyc"):
        return True
    if "/." in path:
        return True
    if "~" in path:
        return True
    return False


# ── 密码哈希（scrypt，stdlib，无新依赖）──
# 存储格式: scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>
_SCRYPT_N = 1 << 15        # 32768
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_SALT_BYTES = 16
# OpenSSL 默认 maxmem=32 MB，n=2^15 时 128*r*n ≈ 32 MB 刚好超出，必须显式抬高。
_SCRYPT_MAXMEM = 128 * 1024 * 1024  # 128 MB


def hash_password(password: str) -> str:
    """生成 scrypt 哈希字符串。"""
    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN, maxmem=_SCRYPT_MAXMEM,
    )
    return "scrypt${n}${r}${p}${salt}${hash}".format(
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        salt=base64.b64encode(salt).decode("ascii"),
        hash=base64.b64encode(derived).decode("ascii"),
    )


def is_hash(value: str) -> bool:
    """判断字符串是否是已哈希过的格式（用于升级旧明文）。"""
    return bool(value) and value.startswith("scrypt$") and value.count("$") == 5


def _verify_hash(candidate: str, stored: str) -> bool:
    """常量时间验证 scrypt 哈希。失败包括格式异常。"""
    try:
        algo, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        derived = hashlib.scrypt(
            candidate.encode("utf-8"),
            salt=salt, n=int(n), r=int(r), p=int(p),
            dklen=len(expected), maxmem=_SCRYPT_MAXMEM,
        )
        return hmac.compare_digest(derived, expected)
    except (ValueError, TypeError, base64.binascii.Error):
        return False


# ── Token 生成与验证（时序安全）──
def _token_material() -> str:
    """cookie 签名材料：哈希字符串本身。空 = 未启用密码。"""
    return settings_service.access_password_hash()


def _make_token(material: str | None = None) -> str:
    material = material if material is not None else _token_material()
    return hmac.new(_SECRET.encode(), material.encode(), hashlib.sha256).hexdigest()


def verify_token(token: str) -> bool:
    material = _token_material()
    if not token or not material:
        return False
    return hmac.compare_digest(token, _make_token(material))


def verify_password(candidate: str) -> bool:
    """常量时间验证密码。stored 永远是哈希；若发现裸明文，认为是误存，拒绝。"""
    if not candidate:
        return False
    stored = settings_service.access_password_hash()
    if not stored:
        return False
    if is_hash(stored):
        return _verify_hash(candidate, stored)
    # 极小概率走到这里：旧明文还没被升级。继续 constant-time 比对，但下次写盘会升级。
    return hmac.compare_digest(candidate, stored)


# ── 速率限制 ──
_chat_rate: dict[str, list[float]] = {}
_login_rate: dict[str, list[float]] = {}
_MAX_TRACKED_IPS = 10000

CHAT_RATE_WINDOW = 60
CHAT_RATE_MAX = 30
LOGIN_RATE_WINDOW = 300
LOGIN_RATE_MAX = 5


def _check_rate(store: dict, ip: str, window: int, limit: int) -> bool:
    now = time.time()
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


def _is_local(request: Request) -> bool:
    client = request.client
    if not client:
        return False
    return client.host in _LOCAL_IPS


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


_OPEN_PATHS = frozenset({"/api/login", "/api/settings/check"})


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        normalized = _normalize_path(request.url.path)
        if normalized is None:
            return JSONResponse({"error": "Bad request"}, status_code=400)
        if _is_path_blocked(normalized):
            return JSONResponse({"error": "Forbidden"}, status_code=403)

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

        if not _token_material():
            # 未设访问密码：仅允许本机访问；外网一律拒绝。避免无密码实例暴露在公网时
            # 被任意读写、甚至被远程抢先设密码。必须先在服务器本机设好密码再对外开放。
            if _is_local(request):
                return await call_next(request)
            if normalized.startswith("/api/"):
                return JSONResponse(
                    {"detail": "Access password not set; remote access disabled. Set a password from localhost first."},
                    status_code=403,
                )
            return HTMLResponse(
                "<!DOCTYPE html><meta charset=utf-8><title>访问被禁用</title>"
                "<div style='font-family:sans-serif;max-width:520px;margin:80px auto;text-align:center'>"
                "<h1>访问被禁用</h1><p>本实例尚未设置访问密码，仅允许本机访问。"
                "请在服务器本机的设置页设好访问密码后，再从外部访问。</p></div>",
                status_code=403,
            )
        if _is_local(request):
            return await call_next(request)
        if normalized in _OPEN_PATHS:
            return await call_next(request)

        token = request.cookies.get("kb_auth", "")
        if verify_token(token):
            return await call_next(request)

        if normalized.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return HTMLResponse(LOGIN_PAGE)
