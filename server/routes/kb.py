"""多知识库 API + 动态 sidebar。

注：/kb/<slug>/... 静态文件**不再**由本路由处理——交给 main.py 的 StaticFiles
mount（指向 web/dist/，Astro 静态构建产物）。
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, field_validator

from server.config import DOCS_ROOT
from server.services import kb_service

router = APIRouter()

_TRASH_ROOT = DOCS_ROOT / "_trash" / "knowledge_bases"


class CreateKnowledgeBaseRequest(BaseModel):
    name: str
    slug: str | None = None

    @field_validator("name")
    @classmethod
    def name_length(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 80:
            raise ValueError("name must be 1-80 chars")
        return value

    @field_validator("slug")
    @classmethod
    def slug_format(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return kb_service.validate_slug(value.strip())


class RenameKnowledgeBaseRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_length(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 80:
            raise ValueError("name must be 1-80 chars")
        return value


@router.get("/api/kbs")
async def list_kbs():
    return {"items": kb_service.list_knowledge_bases()}


@router.post("/api/kbs")
async def create_kb(request: CreateKnowledgeBaseRequest):
    try:
        return kb_service.create_knowledge_base(request.name, request.slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/api/kbs/{slug}")
async def rename_kb(slug: str, request: RenameKnowledgeBaseRequest):
    try:
        return kb_service.rename_knowledge_base(slug, request.name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api/kbs/{slug}")
async def delete_kb(slug: str):
    try:
        return kb_service.delete_knowledge_base(slug, _TRASH_ROOT)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/kbs/{slug}/upload")
async def upload_to_kb(slug: str, request: Request):
    try:
        kb_service.validate_slug(slug)
        form = await request.form()
        return await kb_service.save_uploads(slug, form)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except AssertionError:
        raise HTTPException(
            status_code=500,
            detail="File upload support requires python-multipart. Run: pip install -r requirements.txt",
        )


@router.get("/api/sidebar")
async def api_sidebar():
    return PlainTextResponse(kb_service.build_sidebar_markdown(), media_type="text/markdown")


@router.get("/_sidebar.md")
async def sidebar():
    """根 sidebar：优先静态 DOCS_ROOT/_sidebar.md，缺失才回退到自动构建。

    设计意图：KB 维护者通常会手写一个简洁的 KB switcher 落地页 sidebar；
    auto-build 那种"把每个 KB 的每个 md 全列出来"的版本只在没人手写时兜底。
    """
    static_sidebar = DOCS_ROOT / "_sidebar.md"
    if static_sidebar.exists():
        return PlainTextResponse(
            static_sidebar.read_text(encoding="utf-8"),
            media_type="text/markdown",
        )
    return PlainTextResponse(kb_service.build_sidebar_markdown(), media_type="text/markdown")


# 注：原 /kb/{slug}/ 和 /kb/{slug}/{file_path:path} 两个 FileResponse 路由
# （直接返回 knowledge_bases/<slug>/... 下的 raw markdown）已删除。
# 它们是 Docsify 时代的产物——Docsify 客户端 fetch 原始 .md 自渲染。
# 现在前端是 Astro 静态构建（web/dist/kb/<slug>/<path>/index.html），
# 这些路由不应再拦截 URL，必须 fall through 到 main.py 的 StaticFiles。
# AI 侧边栏读原文用 /api/page-source（独立路由，仍走 knowledge_bases/）。
