import difflib
from fastapi import APIRouter, HTTPException
from server.models import SuggestEditRequest, ApplyEditRequest
from server.services import file_service, claude_service

router = APIRouter()


@router.post("/suggest-edit")
async def suggest_edit(request: SuggestEditRequest):
    """AI 生成修改建议，返回 diff。"""
    try:
        resolved = file_service.resolve_docsify_path(request.page_path)
        original = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    modified = claude_service.suggest_edit(
        page_content=original,
        instruction=request.instruction,
        chat_context=request.chat_context,
    )

    # 生成 unified diff
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="original",
        tofile="modified",
    )
    diff_text = "".join(diff)

    rel_path = str(resolved.relative_to(file_service.DOCS_ROOT))
    return {
        "original": original,
        "modified": modified,
        "diff": diff_text,
        "file_path": rel_path,
    }


@router.post("/apply-edit")
async def apply_edit(request: ApplyEditRequest):
    """确认后写入修改。"""
    try:
        file_service.write_source(request.file_path, request.modified_content)
        return {"success": True, "file_path": request.file_path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
