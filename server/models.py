from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ImageData(BaseModel):
    base64: str
    media_type: str


class ChatRequest(BaseModel):
    page_path: str
    selected_text: str = ""
    messages: list[ChatMessage]
    model: str = ""  # 空串 = 用 backend 的默认模型
    thinking: bool = False
    effort: str = ""  # ""/"default" = 不指定；"low"/"medium"/"high"（仅 OpenAI 推理模型生效，Claude CLI 忽略）
    images: list[ImageData] = []
    session_id: str = ""  # 传入则复用 CLI 会话，无需重发历史


class SuggestEditRequest(BaseModel):
    page_path: str
    instruction: str
    chat_context: str = ""


class ApplyEditRequest(BaseModel):
    file_path: str
    modified_content: str
