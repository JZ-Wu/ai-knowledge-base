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
    model: str = "claude-opus-4-6"
    thinking: bool = False
    images: list[ImageData] = []


class SuggestEditRequest(BaseModel):
    page_path: str
    instruction: str
    chat_context: str = ""


class ApplyEditRequest(BaseModel):
    file_path: str
    modified_content: str
