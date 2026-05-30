"""运行时可变设置：API key、所选 backend、模型列表、ACCESS_PASSWORD、Claude CLI 路径。

持久化到 `server/.settings.json`（已 gitignore）。前端可通过 `/api/settings` 读写。

### 设计要点
- 文件不存在 → 首次启动：探测 PATH 上有没有 `claude`，有就把 backend 默认为 claude_cli，
  无就默认 openai_api（此时 API key 为空，前端会被引导去设置页）。
- 读取时做完整字段补全（向后兼容），不要求老配置带新字段。
- 写入时永远写入完整结构，避免少字段。
- 修改 ACCESS_PASSWORD 后中间件下一个请求会读到新值，无需重启。
"""

import json
import os
import shutil
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

_SETTINGS_FILE = Path(__file__).parent.parent / ".settings.json"
_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None


def _detect_claude_cli() -> str:
    """探测系统上的 claude CLI，找不到返回空字符串。"""
    found = shutil.which("claude")
    if found:
        return found
    # Windows npm global bin
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidate = Path(appdata) / "npm" / "claude.cmd"
        if candidate.exists():
            return str(candidate)
    return ""


def _default_settings() -> dict[str, Any]:
    detected = _detect_claude_cli()
    return {
        "backend": "claude_cli" if detected else "openai_api",
        "access_password": "",
        "claude_cli": {
            "cli_path": detected,  # 空字符串 = 每次调用前重新探测
            "models": [
                {"key": "opus-4-8",   "name": "Claude Opus 4.8",   "model": "claude-opus-4-8"},
                {"key": "opus-4-7",   "name": "Claude Opus 4.7",   "model": "claude-opus-4-7"},
                {"key": "sonnet-4-6", "name": "Claude Sonnet 4.6", "model": "claude-sonnet-4-6"},
                {"key": "haiku-4-5",  "name": "Claude Haiku 4.5",  "model": "claude-haiku-4-5-20251001"},
            ],
            "default_model_key": "opus-4-8",
            "enable_tools": True,
        },
        "openai_api": {
            "models": [
                {
                    "key": "gpt-4-1-mini",
                    "name": "GPT-4.1 mini",
                    "model": "gpt-4.1-mini",
                    "api_key": "",
                    "api_base_url": "https://api.openai.com/v1",
                    "context": 200000,
                }
            ],
            "default_model_key": "gpt-4-1-mini",
            "allow_client_model": True,
            "enable_tools": True,
            "include_usage": False,
            "request_timeout": 300,
            "max_tool_rounds": 5,
            "temperature": 0.2,
        },
        "chat_defaults": {
            "effort": "max",       # 默认思考强度："" / low / medium / high / max
            "enable_tools": True,  # 默认允许 AI 编辑 .md（工具调用）
            "with_page": True,     # 默认把当前页全文作为上下文喂给 AI
        },
    }


def _merge_defaults(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归把 override 合并进 base 的副本上（用于补全旧 settings 缺失字段）。"""
    merged = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_defaults(merged[k], v)
        else:
            merged[k] = v
    return merged


def _read_from_disk() -> dict[str, Any]:
    defaults = _default_settings()
    if not _SETTINGS_FILE.exists():
        return defaults
    try:
        raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return defaults
        return _merge_defaults(defaults, raw)
    except Exception:
        return defaults


def read() -> dict[str, Any]:
    """返回完整 settings dict（带补全）。线程安全。"""
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = _read_from_disk()
        return deepcopy(_CACHE)


def write(new_settings: dict[str, Any]) -> dict[str, Any]:
    """原子覆盖写入（先补全再落盘），更新缓存。

    若 access_password 是裸明文（不是 scrypt$... 格式），自动哈希升级后落盘。
    """
    global _CACHE
    # 延迟 import 避免循环：auth.py 也 import 本模块
    from server.auth import hash_password, is_hash

    defaults = _default_settings()
    merged = _merge_defaults(defaults, new_settings)
    pwd = (merged.get("access_password") or "").strip()
    if pwd and not is_hash(pwd):
        merged["access_password"] = hash_password(pwd)
    with _LOCK:
        tmp = _SETTINGS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_SETTINGS_FILE)
        try:
            os.chmod(str(_SETTINGS_FILE), 0o600)
        except OSError:
            pass
        _CACHE = merged
    return deepcopy(merged)


def invalidate_cache() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None


# ── 便捷访问 ──

def active_backend_name() -> str:
    return read().get("backend") or "claude_cli"


def access_password_hash() -> str:
    """返回 access_password 字段的当前值（哈希字符串），未设置时返回空串。"""
    return (read().get("access_password") or "").strip()


# 兼容别名，避免别处误以为还是明文
def access_password() -> str:
    return access_password_hash()


def is_configured() -> bool:
    """判断当前 backend 是否可用。"""
    s = read()
    if s["backend"] == "claude_cli":
        if s["claude_cli"].get("cli_path"):
            return True
        return bool(_detect_claude_cli())
    profiles = s["openai_api"].get("models") or []
    return any((p.get("api_key") or "").strip() for p in profiles)


def claude_cli_path() -> str:
    """返回当前应使用的 claude CLI 路径（settings 优先，没配则现场探测）。"""
    cfg = read()["claude_cli"]
    explicit = (cfg.get("cli_path") or "").strip()
    if explicit:
        return explicit
    return _detect_claude_cli() or "claude"


def public_view() -> dict[str, Any]:
    """脱敏后的 settings，安全地暴露给前端。API key 只返回是否设置。"""
    s = read()
    openai = deepcopy(s["openai_api"])
    for profile in openai.get("models") or []:
        profile["configured"] = bool((profile.get("api_key") or "").strip())
        profile.pop("api_key", None)
    # 注：claude_cli.cli_path 是本机路径，不脱敏——/api/settings 已由 SecurityMiddleware
    # 保护（未设密码=仅本机；设了密码=需 cookie），不会暴露给网络上的匿名用户。脱敏它
    # 反而会让设置页保存时把 cli_path 清空（write 合并的是 DEFAULTS，前端拿不到原值回填）。
    return {
        "backend": s["backend"],
        "access_password_set": bool((s.get("access_password") or "").strip()),
        "claude_cli": s["claude_cli"],
        "openai_api": openai,
        "chat_defaults": s.get("chat_defaults", {}),
        "is_configured": is_configured(),
        "claude_cli_available": bool(_detect_claude_cli()),
    }
