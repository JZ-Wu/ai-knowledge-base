"""SSE 流式对话路由。把请求转给当前激活的 Backend。"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from server.backends import get_active_backend
from server.models import ChatRequest
from server.services import kb_service, settings_service

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        resolved = kb_service.resolve_docsify_page(request.page_path)
        page_content = resolved.abs_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        page_content = ""
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # 对话默认（设置页可配）：是否带页面全文 / 是否允许工具
    cd = settings_service.read().get("chat_defaults", {})
    if not cd.get("with_page", True):
        page_content = ""
    enable_tools = bool(cd.get("enable_tools", True))

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    images = [{"base64": img.base64, "media_type": img.media_type} for img in request.images]
    backend = get_active_backend()

    def event_generator():
        try:
            for event in backend.stream_chat(
                page_path=request.page_path,
                page_content=page_content,
                selected_text=request.selected_text,
                messages=messages,
                model=request.model,
                thinking=request.thinking,
                effort=request.effort,
                enable_tools=enable_tools,
                images=images,
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logging.error("Chat stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An internal error occurred.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
