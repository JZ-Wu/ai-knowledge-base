"""读取 docsify 页面的 markdown 源码（给 CodeMirror 编辑器用）。"""

from fastapi import APIRouter, HTTPException, Query

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
    # 用 kb_service 已经算好的 rel_path——它正确处理了 DOCS_ROOT 与 EXTERNAL_MOUNTS
    # 两类来源，直接 relative_to(DOCS_ROOT) 会在外部 mount 路径上抛 ValueError → 500。
    return {"source": content, "file_path": resolved.rel_path}
