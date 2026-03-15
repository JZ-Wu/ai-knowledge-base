from fastapi import APIRouter, HTTPException, Query
from server.services import file_service

router = APIRouter()


@router.get("/page-source")
async def get_page_source(path: str = Query(..., description="Docsify page path")):
    """获取页面的 markdown 源码。"""
    try:
        resolved = file_service.resolve_docsify_path(path)
        content = resolved.read_text(encoding="utf-8")
        # 返回相对于 DOCS_ROOT 的路径
        rel_path = resolved.relative_to(file_service.DOCS_ROOT)
        return {"source": content, "file_path": str(rel_path)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
