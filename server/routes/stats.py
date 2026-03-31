import time

from fastapi import APIRouter
from server.config import DOCS_ROOT

router = APIRouter()

_cache = {"data": None, "ts": 0}
_TTL = 60  # 缓存 60 秒


@router.get("/stats")
def get_stats():
    """统计知识库 Markdown 文件数量和总字数（带 60s 缓存）"""
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < _TTL:
        return _cache["data"]

    total_chars = 0
    total_files = 0
    for md in DOCS_ROOT.rglob("*.md"):
        parts = md.relative_to(DOCS_ROOT).parts
        if any(p.startswith(".") or p == "node_modules" for p in parts):
            continue
        try:
            total_chars += len(md.read_text(encoding="utf-8"))
            total_files += 1
        except Exception:
            pass

    result = {"files": total_files, "chars": total_chars}
    _cache["data"] = result
    _cache["ts"] = now
    return result


@router.get("/rate-limits")
async def rate_limits():
    from server.quota_check import check_quota
    rl = check_quota()
    return {"rate_limits": rl}
