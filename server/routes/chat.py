import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from server.models import ChatRequest, CompactRequest
from server.services import file_service, claude_service

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """流式对话，返回 SSE。支持文本回复和工具调用事件。"""
    try:
        resolved = file_service.resolve_docsify_path(request.page_path)
        page_content = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        page_content = ""
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    def event_generator():
        try:
            for event in claude_service.stream_chat(
                page_content=page_content,
                selected_text=request.selected_text,
                messages=messages,
                model=request.model,
                thinking=request.thinking,
                images=[{"base64": img.base64, "media_type": img.media_type} for img in request.images],
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logging.error("Chat stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An internal error occurred.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/compact")
async def compact(request: CompactRequest):
    """手动压缩会话上下文：通过 --resume 发送 /compact 命令。"""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="No active session to compact.")

    def event_generator():
        try:
            for event in claude_service.stream_chat(
                page_content="",
                selected_text="",
                messages=[{"role": "user", "content": "/compact"}],
                model=request.model,
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logging.error("Compact error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Compact failed.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
