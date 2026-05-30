"""读取 docsify 页面的 markdown 源码（给 CodeMirror 编辑器用）。"""

from fastapi import APIRouter, HTTPException, Query

from server.config import DOCS_ROOT
from server.services import kb_service

router = APIRouter()


@router.get("/page-source")
async def get_page_source(path: str = Query(..., description="Docsify page path")):
    try:
        resolved = kb_service.resolve_docsify_page(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    content = resolved.abs_path.read_text(encoding="utf-8")
    # file_path 必须能被 apply-edit 的 validate_path 无歧义还原回原文件。
    # KB 页 → 返回相对 DOCS_ROOT 的 "knowledge_bases/<slug>/<rel>"，命中 validate_path
    # 的 knowledge_bases/ 分支精确定位（裸 rel_path 不含 slug，会回退 DEFAULT_KB_SLUG → 400）。
    # abs_path 不在 DOCS_ROOT 下（外部挂载）→ relative_to 抛 ValueError，退回 rel_path。
    try:
        round_trip = str(resolved.abs_path.relative_to(DOCS_ROOT)).replace("\\", "/")
    except ValueError:
        round_trip = resolved.rel_path
    return {"source": content, "file_path": round_trip, "rel_path": resolved.rel_path}
