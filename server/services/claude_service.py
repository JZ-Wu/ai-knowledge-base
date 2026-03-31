import json
import logging
import os
import platform
import shutil
import subprocess
import threading
from collections.abc import Generator
from pathlib import Path
from server.config import CLAUDE_CLI, DOCS_ROOT, MAX_PAGE_CHARS

logger = logging.getLogger(__name__)

# 确保子进程能找到 node (Windows npm 全局安装需要)
_env = os.environ.copy()
_nodejs_paths = [
    r"C:\Program Files\nodejs",
    os.path.join(os.environ.get("APPDATA", ""), "npm"),
]
_env["PATH"] = ";".join(_nodejs_paths) + ";" + _env.get("PATH", "")

_ALLOWED_MODELS = {"claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"}

_SYSTEM_PROMPT = (
    "你是一个知识库助手，只能读取和编辑 Markdown (.md) 文件。\n"
    "你不能执行命令、读取配置文件、或修改非 Markdown 文件。\n"
    "只有当用户明确要求时，才编辑文件。\n"
    "请用中文回答。"
)

# ── CLI 沙箱：只暴露知识库内容目录和 .md 文件 ──
# 敏感目录/文件不会出现在沙箱内，Claude CLI 根本找不到它们
_EXCLUDED_NAMES = {
    "server", "docs", ".claude", ".git", ".github",
    "node_modules", "__pycache__", "_cli_sandbox", "_backup",
}
_SANDBOX: Path | None = None


def _create_sandbox() -> Path:
    """创建沙箱工作目录，通过 Junction/Symlink 链接内容目录。

    - 内容目录（大模型/, 强化学习/ 等）→ Junction (Win) / Symlink (Unix)
    - 根目录 .md 文件（README.md, _sidebar.md）→ Hard Link
    - server/, docs/, .claude/, .env 等敏感项 → 完全不链接
    """
    sandbox = DOCS_ROOT / "_cli_sandbox"
    is_windows = platform.system() == "Windows"

    # 清理旧沙箱
    if sandbox.exists():
        for item in sandbox.iterdir():
            try:
                if item.is_dir() and not item.is_symlink():
                    os.rmdir(str(item))
                else:
                    item.unlink()
            except OSError:
                pass
    else:
        sandbox.mkdir()

    # 收集需要链接的目录和文件
    dirs_to_link = []
    files_to_link = []
    for item in DOCS_ROOT.iterdir():
        name = item.name
        if name in _EXCLUDED_NAMES or name.startswith("."):
            continue
        if item.is_dir():
            dirs_to_link.append((item, sandbox / name))
        elif item.suffix.lower() == ".md":
            files_to_link.append((item, sandbox / name))

    # Windows: 一次 cmd 调用批量创建所有 Junction（避免 N 次子进程）
    if is_windows and dirs_to_link:
        batch_cmds = " && ".join(
            f'mklink /J "{dst}" "{src}"' for src, dst in dirs_to_link
        )
        subprocess.run(
            ["cmd", "/c", batch_cmds],
            capture_output=True, check=False,
        )
    else:
        for src, dst in dirs_to_link:
            os.symlink(str(src), str(dst))

    # Hard Link .md 文件
    for src, dst in files_to_link:
        try:
            os.link(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))

    logger.info("CLI sandbox created at %s with %d items",
                sandbox, len(dirs_to_link) + len(files_to_link))
    return sandbox


def _get_sandbox() -> Path:
    """获取或创建沙箱目录（首次调用时创建）。"""
    global _SANDBOX
    if _SANDBOX is None:
        _SANDBOX = _create_sandbox()
    return _SANDBOX


def build_prompt(page_content: str, selected_text: str, messages: list[dict]) -> str:
    """将页面上下文、选中文字和对话历史合并为一个完整 prompt。"""
    if len(page_content) > MAX_PAGE_CHARS:
        page_content = page_content[:MAX_PAGE_CHARS] + "\n\n... (内容已截断)"

    parts = [
        "你是一个 AI 知识库助手，帮助用户浏览和编辑一个个人 AI/ML 知识库。",
        "请用中文回答，使用清晰的 Markdown 格式。",
        "默认只回答问题和解释内容，不要修改任何文件。",
        "只有当用户明确要求修改/添加/删除知识库内容时，才编辑对应的 markdown 文件。",
        "编辑完成后告诉用户改了什么，让他们刷新页面查看。",
        f"\n## 当前页面内容\n\n{page_content}",
    ]
    if selected_text:
        parts.append(f"\n## 用户选中的文字\n\n{selected_text}")

    # 添加对话历史
    if len(messages) > 1:
        parts.append("\n## 对话历史")
        for m in messages[:-1]:
            role_label = "用户" if m["role"] == "user" else "助手"
            parts.append(f"\n**{role_label}**: {m['content']}")

    # 最后一条用户消息
    if messages:
        last = messages[-1]
        parts.append(f"\n## 当前问题\n\n{last['content']}")

    return "\n".join(parts)


def stream_chat(
    page_content: str,
    selected_text: str,
    messages: list[dict],
    model: str = "",
    thinking: bool = False,
    images: list[dict] | None = None,
    session_id: str = "",
) -> Generator[dict, None, None]:
    """调用 claude CLI，yield 事件 dict。

    事件格式:
    - {"type": "text", "content": "..."} — AI 文本回复
    - {"type": "tool", "tool": "Edit", "file": "...", "status": "..."} — 工具调用
    - {"type": "session_id", "session_id": "..."} — 会话 ID（用于后续 resume）
    - {"type": "result", "content": "..."} — 最终结果
    - {"type": "error", "content": "..."} — 错误
    """
    # Validate model
    if model and model not in _ALLOWED_MODELS:
        model = ""

    # Validate session_id format
    if session_id and not all(c.isalnum() or c in "-_" for c in session_id):
        session_id = ""

    # 有 session_id 时复用会话，只发最新一条消息，但附带新页面上下文
    if session_id and messages:
        user_msg = messages[-1]["content"]
        context_parts = []
        if page_content:
            if len(page_content) > MAX_PAGE_CHARS:
                page_content = page_content[:MAX_PAGE_CHARS] + "\n\n... (内容已截断)"
            context_parts.append(f"[当前页面内容]\n{page_content}")
        if selected_text:
            context_parts.append(f"[用户选中的文字]\n{selected_text}")
        if context_parts:
            prompt = "\n\n".join(context_parts) + f"\n\n{user_msg}"
        else:
            prompt = user_msg
    else:
        prompt = build_prompt(page_content, selected_text, messages)

    # 将图片嵌入为 data-URI
    if images:
        img_parts = []
        for img in images:
            media = img.get("media_type", "image/png")
            b64 = img.get("base64", "")
            if not b64:
                continue
            if len(b64) > 1_500_000:  # ~1MB limit
                continue
            img_parts.append(f"![image](data:{media};base64,{b64})")
        if img_parts:
            prompt = "\n".join(img_parts) + "\n\n" + prompt

    logger.info("Prompt size: %d bytes, images: %d", len(prompt.encode("utf-8")), len(images) if images else 0)

    cmd = [CLAUDE_CLI, "-p", "--verbose", "--output-format", "stream-json"]
    cmd.extend(["--allowedTools", "Read,Edit,Write,Glob,Grep,WebSearch,WebFetch"])
    cmd.extend(["--system-prompt", _SYSTEM_PROMPT])
    if session_id:
        cmd.extend(["--resume", session_id])
    cmd.extend(["--model", model if model else "sonnet"])
    cmd.extend(["--thinking", "enabled" if thinking else "disabled"])

    sandbox = _get_sandbox()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(sandbox),
        env=_env,
    )

    # 后台线程收集 stderr，防止缓冲区满导致死锁
    stderr_chunks: list[str] = []

    def _read_stderr():
        try:
            for line in proc.stderr:
                stderr_chunks.append(line.decode("utf-8", errors="replace"))
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
    except BrokenPipeError:
        pass  # 进程可能已提前退出，继续读 stdout

    got_text = False
    for raw_line in proc.stdout:
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not line:
            continue
        try:
            event = json.loads(line)
            event_type = event.get("type", "")

            # 从 init 事件提取 session_id，返回给前端用于后续 resume
            if event_type == "system" and event.get("subtype") == "init":
                sid = event.get("session_id", "")
                if sid:
                    yield {"type": "session_id", "session_id": sid}
                continue

            if event_type == "assistant":
                msg = event.get("message", {})
                # 检测 context management（上下文压缩）
                ctx_mgmt = msg.get("context_management")
                if ctx_mgmt and ctx_mgmt.get("applied_edits"):
                    yield {"type": "context_compact", "edits": ctx_mgmt["applied_edits"]}
                for block in msg.get("content", []):
                    if block.get("type") == "thinking" and block.get("thinking"):
                        yield {"type": "thinking", "content": block["thinking"]}
                    elif block.get("type") == "text" and block.get("text"):
                        got_text = True
                        yield {"type": "text", "content": block["text"]}
                    elif block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        # 只显示文件修改类工具，忽略只读操作
                        if tool_name in ("Edit", "Write", "NotebookEdit"):
                            tool_input = block.get("input", {})
                            file_path = (
                                tool_input.get("file_path", "")
                                or tool_input.get("path", "")
                            )
                            yield {
                                "type": "tool",
                                "tool": tool_name,
                                "file": file_path,
                            }

            elif event_type == "rate_limit_event":
                info = event.get("rate_limit_info", {})
                yield {
                    "type": "rate_limit",
                    "status": info.get("status", ""),
                    "resets_at": info.get("resetsAt", 0),
                    "limit_type": info.get("rateLimitType", ""),
                }

            elif event_type == "result":
                result_text = event.get("result", "")
                if result_text and not got_text:
                    yield {"type": "text", "content": result_text}
                usage = event.get("usage", {})
                if usage:
                    yield {
                        "type": "usage",
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read": usage.get("cache_read_input_tokens", 0),
                        "cache_create": usage.get("cache_creation_input_tokens", 0),
                    }
                duration = event.get("duration_ms", 0)
                if duration:
                    yield {"type": "duration", "ms": duration}

        except json.JSONDecodeError:
            continue

    proc.wait()
    stderr_thread.join(timeout=5)

    if proc.returncode != 0 and not got_text:
        # resume 失败时自动降级为新会话重试
        if session_id:
            yield from stream_chat(
                page_content=page_content,
                selected_text=selected_text,
                messages=messages,
                model=model,
                thinking=thinking,
                images=images,
                session_id="",  # 不再 resume，新建会话
            )
            return
        detail = "".join(stderr_chunks).strip() or f"exit code {proc.returncode}"
        logger.error("Claude CLI error: %s", detail)
        yield {"type": "error", "content": "Claude CLI encountered an error. Please try again."}
