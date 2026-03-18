import base64
import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Generator
from server.config import CLAUDE_CLI, DOCS_ROOT, MAX_PAGE_CHARS

# 确保子进程能找到 node (Windows npm 全局安装需要)
_env = os.environ.copy()
_nodejs_paths = [
    r"C:\Program Files\nodejs",
    os.path.join(os.environ.get("APPDATA", ""), "npm"),
]
_env["PATH"] = ";".join(_nodejs_paths) + ";" + _env.get("PATH", "")


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
    # 有 session_id 时复用会话，只发最新一条消息
    if session_id and messages:
        prompt = messages[-1]["content"]
    else:
        prompt = build_prompt(page_content, selected_text, messages)

    # 将图片写入临时文件
    tmp_files = []
    if images:
        ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
        for img in images:
            ext = ext_map.get(img.get("media_type", ""), ".png")
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(base64.b64decode(img["base64"]))
            tmp.close()
            tmp_files.append(tmp.name)

    cmd = [CLAUDE_CLI, "-p", "--verbose", "--output-format", "stream-json"]
    if session_id:
        cmd.extend(["--resume", session_id])
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--thinking", "enabled" if thinking else "disabled"])
    for f in tmp_files:
        cmd.extend(["--image", f])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(DOCS_ROOT),
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

            elif event_type == "result":
                result_text = event.get("result", "")
                if result_text and not got_text:
                    yield {"type": "text", "content": result_text}

        except json.JSONDecodeError:
            continue

    proc.wait()
    stderr_thread.join(timeout=5)

    # 清理临时图片文件
    for f in tmp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

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
        yield {"type": "error", "content": f"Claude CLI error: {detail}"}
