import base64
import binascii
import json
import logging
import os
import subprocess
import threading
import uuid
from collections.abc import Generator
from pathlib import Path
from server.config import CLAUDE_CLI, DOCS_ROOT, MAX_PAGE_CHARS, MAX_PDF_CHARS

logger = logging.getLogger(__name__)
# Default root level is WARNING; bump so logger.info() actually shows up.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger.setLevel(logging.INFO)

# 最小化子进程环境变量，只保留必要项，防止泄漏敏感信息
_env = {
    "PATH": ";".join([
        r"C:\Program Files\nodejs",
        os.path.join(os.environ.get("APPDATA", ""), "npm"),
        os.environ.get("PATH", ""),
    ]),
    "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
    "TEMP": os.environ.get("TEMP", ""),
    "TMP": os.environ.get("TMP", ""),
    "APPDATA": os.environ.get("APPDATA", ""),
    "USERPROFILE": os.environ.get("USERPROFILE", ""),
    "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
}

_ALLOWED_MODELS = {"claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6", "claude-opus-4-7"}

# 只允许操作 .md 文件的工具集（不给 Bash/WebFetch 等危险工具）
_ALLOWED_TOOLS = "Read,Edit,Write,Glob,Grep"

_SYSTEM_PROMPT = (
    "你是一个知识库助手。严格遵守以下规则：\n\n"
    "## 文件访问规则\n"
    "1. 你只能读取和编辑 Markdown (.md) 文件\n"
    "2. **例外**：当用户消息里给出了 `.tmp_images/` 子目录里的图片绝对路径时，"
    "你必须用 Read 工具读取那些路径，把图片当作用户输入的一部分。这是用户上传图片的标准入口\n"
    "3. 禁止访问以下内容：\n"
    "   - .py, .js, .json, .env, .yml, .yaml, .toml, .cfg, .ini, .html, .css 文件\n"
    "   - server/, docs/, .git/, .claude/, .github/ 目录\n"
    "   - 任何包含 config, secret, key, password, token, credential 的文件名\n"
    "4. 禁止使用绝对路径（如 D:\\, C:\\ 开头的路径）——**仅 .tmp_images/ 下的图片例外**\n"
    "5. 禁止使用 ../ 访问上级目录\n"
    "6. 只在当前工作目录及其子目录内操作\n"
    "7. 禁止创建或修改 CLAUDE.md 文件\n\n"
    "## 行为规则\n"
    "8. 只有当用户明确要求修改时，才编辑文件\n"
    "9. 请用中文回答，使用清晰的 Markdown 格式\n"
)

# 上传图片落地目录（DOCS_ROOT 下的相对子目录，gitignored）。
# 用 .tmp_images/ 是因为系统提示禁止模型用绝对路径，必须给一个相对路径让模型 Read。
_IMG_TMP_DIR = ".tmp_images"
_IMG_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _save_temp_images(images: list[dict] | None) -> tuple[list[Path], list[str]]:
    """把上传图片落到 DOCS_ROOT/.tmp_images/，返回 (绝对路径列表, 给模型用的 posix 字符串列表)。

    数据 URI 文本注入不会让 CLI 真的把内容当图片送给模型——必须落到磁盘，
    再让模型用 Read 工具读绝对路径（Claude Code 的 Read 强制要求绝对路径），
    Read 才会以 multimodal block 的形式把图片真正送给 Claude。
    """
    if not images:
        return [], []
    tmp_dir = DOCS_ROOT / _IMG_TMP_DIR
    tmp_dir.mkdir(exist_ok=True)
    abs_paths: list[Path] = []
    abs_refs: list[str] = []
    for img in images:
        b64 = img.get("base64", "")
        if not b64 or len(b64) > 1_500_000:  # ~1MB binary
            continue
        ext = _IMG_EXT.get(img.get("media_type", ""), "png")
        try:
            data = base64.b64decode(b64, validate=True)
        except (ValueError, binascii.Error):
            continue
        fname = f"{uuid.uuid4().hex}.{ext}"
        abs_path = tmp_dir / fname
        try:
            abs_path.write_bytes(data)
        except OSError:
            continue
        abs_paths.append(abs_path)
        # posix 形式（正斜杠）—— Read 在 Windows 上两种都接受，但 posix 更不容易被
        # 当成转义字符踩到坑。
        abs_refs.append(abs_path.resolve().as_posix())
    return abs_paths, abs_refs


def _build_image_instruction(refs: list[str]) -> str:
    if not refs:
        return ""
    lines = [
        f"## 用户随本次消息上传了 {len(refs)} 张图片",
        "",
        "**你必须先用 Read 工具读取下面每一个绝对路径**，再回答用户的问题。"
        "这些是用户输入的一部分，路径已经是 .tmp_images/ 下的临时文件，"
        "系统提示里已明确允许这种例外：",
        "",
    ]
    for r in refs:
        lines.append(f"- `{r}`")
    lines.append("")
    lines.append("读完后基于图片内容回答。这些是临时文件，处理完无需删除或修改。")
    lines.append("")
    return "\n".join(lines)

# 子进程超时（秒）
_CLI_TIMEOUT = 300


def build_prompt(page_content: str, selected_text: str, messages: list[dict]) -> str:
    """将页面上下文、选中文字和对话历史合并为一个完整 prompt。"""
    # 检测是否是 PDF 全文提取（frontmatter 含 source_pdf:）
    is_pdf_extract = (
        page_content.lstrip().startswith("---")
        and "source_pdf:" in page_content[:800]
    )
    max_chars = MAX_PDF_CHARS if is_pdf_extract else MAX_PAGE_CHARS
    if len(page_content) > max_chars:
        page_content = page_content[:max_chars] + "\n\n... (内容已截断)"

    parts = [
        "你是一个 AI 知识库助手，帮助用户浏览和编辑一个个人 AI/ML 知识库。",
        "请用中文回答，使用清晰的 Markdown 格式。",
        "默认只回答问题和解释内容，不要修改任何文件。",
        "只有当用户明确要求修改/添加/删除知识库内容时，才编辑对应的 markdown 文件。",
        "编辑完成后告诉用户改了什么，让他们刷新页面查看。",
    ]

    if is_pdf_extract:
        parts.append(
            "\n## 阅读模式提示\n\n"
            "当前页面是用户正在阅读的一篇 **学术论文 PDF 的自动抽取全文**（文件由 pymupdf 生成，页码以 `## Page N` 标记）。"
            "回答规则：\n"
            "1. **严格基于该 PDF 内容作答**，不臆测论文未写的细节；引用时标注页码，如 `(Page 3)` 或小节标题。\n"
            "2. 若问题不在 PDF 范围内，明确说「这不在这篇论文里」，可补充你的背景知识但须标注来源。\n"
            "3. 用户选中的文字 = 他在 PDF 中高亮的段落——**围绕该段落回答**，不要答非所问。\n"
            "4. 抽取文本可能有断行、乱码、公式错位，遇到明显 OCR 噪声时结合上下文推断，并提示用户以 PDF 原文为准。\n"
            "5. **不要改写任何文件**——PDF 阅读场景下一律只答问题。"
        )

    parts.append(f"\n## 当前页面内容\n\n{page_content}")

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

    有 session_id 时用 --resume 恢复会话，只发最后一条消息；
    否则用 build_prompt 带完整上下文创建新会话。
    """
    # Validate model
    if model and model not in _ALLOWED_MODELS:
        model = ""

    resuming = bool(session_id)

    if resuming:
        # resume 模式：只发最后一条用户消息，附加选中文字
        last_msg = messages[-1]["content"] if messages else ""
        if selected_text:
            prompt = f"用户选中了以下文字：\n\n{selected_text}\n\n用户的问题：{last_msg}"
        else:
            prompt = last_msg
    else:
        # 新会话：完整 prompt 含页面内容和对话历史
        prompt = build_prompt(page_content, selected_text, messages)

    # 把图片落地到 .tmp_images/，再让模型用 Read 工具查看（CLI 才会真把
    # 图片当 multimodal block 发给 Claude；data-URI 文本注入做不到）。
    img_abs_paths, img_refs = _save_temp_images(images)
    if img_refs:
        prompt = _build_image_instruction(img_refs) + "\n" + prompt
        for r in img_refs:
            logger.info("Image saved for Read: %s", r)

    logger.info("Session: %s | Prompt size: %d bytes, images: %d",
                session_id or "(new)", len(prompt.encode("utf-8")), len(img_refs))

    cmd = [CLAUDE_CLI, "-p", "--verbose", "--output-format", "stream-json"]
    cmd.extend(["--allowedTools", _ALLOWED_TOOLS])
    cmd.extend(["--model", model if model else "sonnet"])
    cmd.extend(["--thinking", "enabled" if thinking else "disabled"])

    if resuming:
        cmd.extend(["--resume", session_id])
    else:
        cmd.extend(["--system-prompt", _SYSTEM_PROMPT])

    # 直接在知识库目录运行，不使用沙箱
    # 安全靠: --allowedTools 限制工具 + system prompt 约束 + 中间件阻止敏感路径
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(DOCS_ROOT),
        env=_env,
    )

    # 超时计时器，防止子进程挂起
    def _kill_on_timeout():
        try:
            proc.kill()
        except OSError:
            pass

    timeout_timer = threading.Timer(_CLI_TIMEOUT, _kill_on_timeout)
    timeout_timer.start()

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
    got_delta = False  # 收到 delta 流后，跳过 assistant 事件中的完整文本
    try:
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

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield {"type": "thinking", "content": delta["thinking"]}
                    elif delta.get("type") == "text_delta" and delta.get("text"):
                        got_text = True
                        got_delta = True
                        yield {"type": "text", "content": delta["text"]}

                elif event_type == "assistant":
                    msg = event.get("message", {})
                    # 检测 context management（上下文压缩）
                    ctx_mgmt = msg.get("context_management")
                    if ctx_mgmt and ctx_mgmt.get("applied_edits"):
                        yield {"type": "context_compact", "edits": ctx_mgmt["applied_edits"]}
                    # 只在没收到 delta 流时才从 assistant 事件取文本（兜底）
                    if not got_delta:
                        for block in msg.get("content", []):
                            if block.get("type") == "thinking" and block.get("thinking"):
                                yield {"type": "thinking", "content": block["thinking"]}
                            elif block.get("type") == "text" and block.get("text"):
                                got_text = True
                                yield {"type": "text", "content": block["text"]}
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
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
                    else:
                        # delta 模式下仍然提取 tool_use 信息
                        for block in msg.get("content", []):
                            if block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
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

        proc.wait(timeout=10)
        stderr_thread.join(timeout=5)

        if proc.returncode != 0 and not got_text:
            detail = "".join(stderr_chunks).strip() or f"exit code {proc.returncode}"
            logger.error("Claude CLI error: %s", detail)

            # resume 失败时自动用完整历史重试新会话
            if resuming:
                logger.info("Resume failed, retrying with full history...")
                yield {"type": "text", "content": "> *Session 已过期，正在新建会话...*\n\n"}
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

            yield {"type": "error", "content": "Claude CLI encountered an error. Please try again."}
    finally:
        timeout_timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        # 清理本次上传的图片临时文件
        for p in img_abs_paths:
            try:
                p.unlink()
            except OSError:
                pass
