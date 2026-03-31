"""读取 Claude CLI OAuth 凭据，发最小请求获取实时配额百分比。"""

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
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
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
        logging.error(f"Quota check failed: {e}")
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
