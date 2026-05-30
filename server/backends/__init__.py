"""Backend 抽象层。

把 "调用 LLM 生成回复 / 让 LLM 编辑文件" 这件事抽出协议，让两种实现可互换：

- `claude_cli`：通过 Anthropic 官方 CLI 调用 Claude（用户用订阅，免 API key）。
- `openai_api`：直接打 OpenAI 兼容 Chat Completions（要 API key，可对接 DeepSeek 等）。

挑哪个由 `settings_service.active_backend_name()` 决定。
"""

from collections.abc import Generator
from typing import Any, Protocol

from server.services import settings_service


class Backend(Protocol):
    """所有 backend 必须实现的接口。"""

    name: str

    def list_models(self) -> list[dict[str, Any]]:
        """返回前端可展示的模型列表（不含 api_key/base_url 等机密）。"""
        ...

    def status(self) -> dict[str, Any]:
        """简要的健康/可用性状态，供 /api/rate-limits 兼容接口使用。"""
        ...

    def stream_chat(
        self,
        *,
        page_path: str,
        page_content: str,
        selected_text: str,
        messages: list[dict[str, Any]],
        model: str = "",
        thinking: bool = False,
        effort: str = "",
        enable_tools: bool = True,
        images: list[dict[str, Any]] | None = None,
        session_id: str = "",
    ) -> Generator[dict[str, Any], None, None]:
        """对话主入口，yield 前端 SSE 事件 dict。"""
        ...

    def suggest_edit(
        self,
        *,
        page_content: str,
        instruction: str,
        chat_context: str = "",
    ) -> str:
        """根据 instruction 返回完整修改后的 Markdown。"""
        ...


def get_active_backend() -> Backend:
    """工厂：按当前 settings 返回 backend 实例。"""
    name = settings_service.active_backend_name()
    # 延迟 import 避免循环依赖。
    if name == "openai_api":
        from server.backends.openai_api import OpenAIAPIBackend
        return OpenAIAPIBackend()
    from server.backends.claude_cli import ClaudeCLIBackend
    return ClaudeCLIBackend()
