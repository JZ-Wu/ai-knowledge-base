"""读取 Claude CLI 的 OAuth 凭据，发一个最小请求，从响应头取实时配额百分比。

5h / 7d 用量百分比**不在** CLI 的 stream-json 里（rate_limit_event 只有 status +
resets_at），只能靠带订阅 OAuth token 调一次 Anthropic API、读
`anthropic-ratelimit-unified-*` 响应头拿到。30s 缓存，避免频繁打点。

（多 KB 重构拆 backends/ 时把这套丢了，导致配额条只剩 status；此处恢复。）
"""

import json
import logging
import time
from pathlib import Path

import httpx

_CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
_API_URL = "https://api.anthropic.com/v1/messages"

_cache: dict = {}
_cache_ts: float = 0
_CACHE_TTL = 30


def _read_token() -> str | None:
    try:
        data = json.loads(_CREDENTIALS_FILE.read_text(encoding="utf-8"))
        return data.get("claudeAiOauth", {}).get("accessToken")
    except Exception:
        return None


def check_quota() -> dict:
    """返回 {status, five_hour:{used_percentage,resets_at}, seven_day:{...}}；失败返回 {error}。"""
    global _cache, _cache_ts
    if time.time() - _cache_ts < _CACHE_TTL and _cache:
        return _cache
    token = _read_token()
    if not token:
        return {"error": "no credentials"}
    try:
        resp = httpx.post(
            _API_URL,
            headers={
                # 订阅 OAuth token（sk-ant-oat…）必须走 Bearer + oauth beta；旧的 x-api-key
                # 现在返回 401 invalid（Anthropic 改了鉴权），这正是 5h 用量曾失效的原因。
                "authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "q"}],
            },
            timeout=10,
        )
    except Exception as e:
        logging.error("Quota check failed: %s", e)
        return {"error": "Failed to check quota"}
    h = resp.headers
    result: dict = {"status": h.get("anthropic-ratelimit-unified-status", "unknown")}
    for key, abbrev in [("five_hour", "5h"), ("seven_day", "7d")]:
        util = h.get(f"anthropic-ratelimit-unified-{abbrev}-utilization")
        reset = h.get(f"anthropic-ratelimit-unified-{abbrev}-reset")
        if util is not None and reset is not None:
            result[key] = {
                "used_percentage": round(float(util) * 100, 1),
                "resets_at": int(reset),
            }
    _cache = result
    _cache_ts = time.time()
    return result
