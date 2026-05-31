"""Claude CLI backend：通过 Anthropic 官方 CLI 调用 Claude，用户用订阅，免 API key。

从原来的 `server/services/claude_service.py` 重构而来：
- 把 model 列表 / cli_path 改为从 `settings_service` 读
- system prompt 增加多 KB 提示（让模型知道当前 KB slug、相对路径）
- 工具白名单依然只给 Read/Edit/Write/Glob/Grep
- 图片上传通道（.tmp_images/）保留
- session_id resume 保留

工作目录现在切到具体 KB（`knowledge_bases/<slug>/`）而不是 DOCS_ROOT，
这样模型给出的相对路径就直接是 KB 内路径，与多 KB 设计一致。
"""

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
from typing import Any

from server.config import (
    DOCS_ROOT,
    KB_ROOT,
    MAX_PAGE_CHARS,
    MAX_PDF_CHARS,
    MAX_PDF_INDEX_CHARS,
    PDF_RENDER_DPI,
)
from server.services import kb_service, pdf_render, quota_check, settings_service

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger.setLevel(logging.INFO)


# 最小化子进程环境变量，只保留必要项，防止泄漏敏感信息
def _build_env() -> dict[str, str]:
    return {
        "PATH": ";".join(
            [
                r"C:\Program Files\nodejs",
                os.path.join(os.environ.get("APPDATA", ""), "npm"),
                os.environ.get("PATH", ""),
            ]
        ),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "APPDATA": os.environ.get("APPDATA", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
    }


# 只允许操作 .md 文件的工具集（不给 Bash/WebFetch 等危险工具）
_ALLOWED_TOOLS = "Read,Edit,Write,Glob,Grep"

_BASE_SYSTEM_PROMPT = (
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

# 图片落到这里：CLI cwd 切到 KB 根目录，所以 .tmp_images/ 也写到 KB 内。
_IMG_TMP_DIR = ".tmp_images"
_IMG_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}
_IMG_MAX_B64 = 7_000_000
_CLI_TIMEOUT = 300


def _save_temp_images(cwd: Path, images: list[dict] | None) -> tuple[list[Path], list[str]]:
    if not images:
        return [], []
    tmp_dir = cwd / _IMG_TMP_DIR
    tmp_dir.mkdir(exist_ok=True)
    abs_paths: list[Path] = []
    abs_refs: list[str] = []
    for i, img in enumerate(images):
        b64 = img.get("base64", "")
        media = img.get("media_type", "")
        if not b64 or len(b64) > _IMG_MAX_B64:
            logger.warning("[img] #%d skipped (size=%d)", i, len(b64))
            continue
        ext = _IMG_EXT.get(media, "png")
        try:
            data = base64.b64decode(b64, validate=True)
        except (ValueError, binascii.Error) as e:
            logger.warning("[img] #%d decode failed: %s", i, e)
            continue
        fname = f"{uuid.uuid4().hex}.{ext}"
        path = tmp_dir / fname
        try:
            path.write_bytes(data)
        except OSError as e:
            logger.warning("[img] #%d write failed: %s", i, e)
            continue
        abs_paths.append(path)
        abs_refs.append(path.resolve().as_posix())
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


def _is_pdf_extract(page_content: str) -> bool:
    """page_content 是不是 pymupdf 抽取的 PDF 全文（靠 source_pdf: frontmatter 判定）。"""
    return (
        page_content.lstrip().startswith("---")
        and "source_pdf:" in page_content[:800]
    )


def _build_prompt(
    page_path: str,
    page_content: str,
    selected_text: str,
    messages: list[dict],
    kb_slug: str,
    rel_path: str,
    pdf_pages: tuple | None = None,  # (pages_rel, page_count, md_rel)：有则走"按需看图"模式
) -> str:
    is_pdf_extract = _is_pdf_extract(page_content)
    pdf_vision = is_pdf_extract and pdf_pages is not None

    parts = [
        "你是一个 AI 知识库助手，帮助用户浏览和编辑一个 Markdown 知识库。",
        "请用中文回答，使用清晰的 Markdown 格式。",
        "默认只回答问题和解释内容，不要修改任何文件。",
        "只有当用户明确要求修改/添加/删除知识库内容时，才编辑对应的 markdown 文件。",
        "编辑完成后告诉用户改了什么，让他们刷新页面查看。",
    ]

    if kb_slug:
        parts.append(
            f"\n## 当前知识库\n\n"
            f"- 你的工作目录（cwd）已经是知识库根：`knowledge_bases/{kb_slug}/`\n"
            f"- 当前页面相对路径：`{rel_path or 'README.md'}`\n"
            "**所有工具调用（Read/Edit/Write/Glob/Grep）的 file_path 都用相对于当前 cwd 的相对路径**，例如 `大模型/README.md`。"
        )
    else:
        parts.append(f"\n## 当前页面路径\n\n{page_path or 'README.md'}")

    if pdf_vision:
        pages_rel, page_count, md_rel = pdf_pages
        index_excerpt = page_content[:MAX_PDF_INDEX_CHARS]
        parts.append(
            "\n## 阅读模式（PDF · 按需看图）\n\n"
            f"用户正在读一篇学术论文 PDF（共 **{page_count} 页**）。为省上下文又保留图表/公式/版面，"
            "**没把全文塞进来**，而是给你两样东西按需取：\n"
            f"- **文字索引**：`{md_rel}`（pymupdf 抽取全文，按 `## Page N` 分页，可 Grep/Read 定位）。\n"
            f"- **每页图片**：`{pages_rel}/pNNN.png`（三位补零，如 `{pages_rel}/p001.png`；含图表/公式/版面）。\n\n"
            f"**以上路径都相对当前 cwd（`knowledge_bases/{kb_slug}/`），直接 Read/Grep 即可——别去 `wiki/papers/` 等别处找（那里是论文卡片，不是这篇 PDF），也别用 Bash 乱探目录。**\n\n"
            "回答规则：\n"
            f"1. 先用文字索引（Grep/Read `{md_rel}`）定位相关页，**再 `Read` 对应页的 PNG 看图**——问到图/表/公式/架构/具体数字时务必看页图，别只靠抽取文本（常有乱码错位）。\n"
            "2. **严格基于该 PDF 作答**、引用标页码；不在论文里就明说「这不在这篇里」。\n"
            "3. 用户选中的文字 = 他高亮的段落，**围绕它回答**。\n"
            "4. 纯文字的简单问题可只用索引；**别把所有页图都读一遍**，只读需要的。\n"
            "5. **只答问题、不改写任何文件**。\n\n"
            f"### 文字索引开头（仅供快速定位，非全文）\n\n{index_excerpt}\n\n"
            f"... (索引已截断；需要更多文字请 Read `{md_rel}`，需要看版面/图表请 Read 对应 `{pages_rel}/pNNN.png`)"
        )
    elif is_pdf_extract:
        # 渲染不可用（缺 pymupdf / 渲染失败）→ 退回旧行为：塞抽取全文（截断）。
        content = page_content
        if len(content) > MAX_PDF_CHARS:
            content = content[:MAX_PDF_CHARS] + "\n\n... (内容已截断)"
        parts.append(
            "\n## 阅读模式提示\n\n"
            "当前页面是一篇学术论文 PDF 的自动抽取全文（pymupdf，页码 `## Page N`）。"
            "严格基于内容作答、标注页码、不改写文件；抽取文本可能断行/乱码/公式错位，结合上下文推断。"
        )
        parts.append(f"\n## 当前页面内容\n\n{content}")
    else:
        content = page_content
        if len(content) > MAX_PAGE_CHARS:
            content = content[:MAX_PAGE_CHARS] + "\n\n... (内容已截断)"
        parts.append(f"\n## 当前页面内容\n\n{content}")

    if selected_text:
        parts.append(f"\n## 用户选中的文字\n\n{selected_text}")

    if len(messages) > 1:
        parts.append("\n## 对话历史")
        for m in messages[:-1]:
            role_label = "用户" if m["role"] == "user" else "助手"
            parts.append(f"\n**{role_label}**: {m['content']}")

    if messages:
        parts.append(f"\n## 当前问题\n\n{messages[-1]['content']}")

    return "\n".join(parts)


def _resolve_cwd(kb_slug: str) -> Path:
    """决定 CLI 的工作目录：有 slug 切到 KB 根，否则用 DOCS_ROOT（兜底，让模型操作根 README 等）。"""
    if kb_slug:
        try:
            return kb_service.kb_dir(kb_slug)
        except ValueError:
            pass
    return DOCS_ROOT


def _allowed_models() -> set[str]:
    cfg = settings_service.read()["claude_cli"]
    return {p["model"] for p in (cfg.get("models") or []) if p.get("model")}


def _default_model() -> str:
    cfg = settings_service.read()["claude_cli"]
    default_key = cfg.get("default_model_key", "")
    for p in cfg.get("models") or []:
        if p.get("key") == default_key:
            return p["model"]
    if cfg.get("models"):
        return cfg["models"][0]["model"]
    return "sonnet"


class ClaudeCLIBackend:
    name = "claude_cli"

    def list_models(self) -> list[dict[str, Any]]:
        cfg = settings_service.read()["claude_cli"]
        default_key = cfg.get("default_model_key", "")
        return [
            {
                "value": p.get("key") or p.get("model"),
                "label": p.get("name") or p.get("model"),
                "model": p["model"],
                "is_default": p.get("key") == default_key,
                "configured": True,
            }
            for p in cfg.get("models") or []
        ]

    def status(self) -> dict[str, Any]:
        cli_path = settings_service.claude_cli_path()
        result: dict[str, Any] = {
            "backend": self.name,
            "cli_path": cli_path,
            "available": bool(cli_path),
        }
        # 合并实时 5h/7d 用量百分比：带订阅 OAuth token 调一次 Anthropic API、读限流响应头
        # （CLI 的 stream-json 不给百分比）。失败/无凭据静默跳过，只缺百分比不影响其它字段。
        try:
            q = quota_check.check_quota()
            if isinstance(q, dict) and not q.get("error"):
                result.update(q)  # 加上 status / five_hour / seven_day
        except Exception:
            pass
        return result

    def stream_chat(
        self,
        *,
        page_path: str,
        page_content: str,
        selected_text: str,
        messages: list[dict[str, Any]],
        model: str = "",
        thinking: bool = False,
        effort: str = "",  # low/medium/high/max → CLI 的 --effort（思考强度）
        enable_tools: bool = True,  # 是否允许工具（Read/Edit/Write/Glob/Grep）
        images: list[dict[str, Any]] | None = None,
        session_id: str = "",
    ) -> Generator[dict[str, Any], None, None]:
        kb_slug, rel_path = kb_service.split_docsify_path(page_path or "")
        kb_slug = kb_slug or ""
        cwd = _resolve_cwd(kb_slug)
        cwd.mkdir(parents=True, exist_ok=True)

        # 校验/规范化模型名
        allowed = _allowed_models()
        if model and model not in allowed:
            # 允许前端传 key（如 "sonnet-4-6"）—— 用 settings 查到真实 model
            cfg_models = settings_service.read()["claude_cli"]["models"]
            mapped = next((p["model"] for p in cfg_models if p.get("key") == model), "")
            model = mapped if mapped else ""
        if not model:
            model = _default_model()

        # PDF 按需视觉：是 PDF 抽取页就把每页懒渲染成 PNG（缓存 papers/.pages/<stem>/），
        # 让模型按需 Read 看图，而不是把全文塞进 prompt。
        pdf_pages = None
        if page_content and rel_path and _is_pdf_extract(page_content):
            stem = Path(rel_path).stem
            pdf_abs = cwd / "papers" / f"{stem}.pdf"
            count = pdf_render.ensure_pages(pdf_abs, cwd / "papers" / ".pages" / stem, PDF_RENDER_DPI)
            if count:
                pdf_pages = (f"papers/.pages/{stem}", count, f"papers/{stem}.md")

        resuming = bool(session_id)
        if resuming:
            last_msg = messages[-1]["content"] if messages else ""
            if selected_text:
                prompt = f"用户选中了以下文字：\n\n{selected_text}\n\n用户的问题：{last_msg}"
            else:
                prompt = last_msg
            if pdf_pages:
                prompt = (
                    f"（当前在读 PDF：文字索引 `{pdf_pages[2]}`、每页图 `{pdf_pages[0]}/pNNN.png`，"
                    "需要看图就 Read 对应页。）\n\n" + prompt
                )
        else:
            prompt = _build_prompt(
                page_path, page_content, selected_text, messages, kb_slug, rel_path, pdf_pages
            )

        img_abs_paths, img_refs = _save_temp_images(cwd, images)
        if img_refs:
            prompt = _build_image_instruction(img_refs) + "\n" + prompt

        cli = settings_service.claude_cli_path()
        cmd = [cli, "-p", "--verbose", "--output-format", "stream-json"]
        if enable_tools:
            cmd.extend(["--allowedTools", _ALLOWED_TOOLS])
        cmd.extend(["--model", model])
        cmd.extend(["--thinking", "enabled" if thinking else "disabled"])
        if effort in ("low", "medium", "high", "max"):
            cmd.extend(["--effort", effort])
        if resuming:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--system-prompt", _BASE_SYSTEM_PROMPT])

        logger.info(
            "claude_cli call: kb=%s model=%s session=%s prompt=%dB images=%d cwd=%s",
            kb_slug or "(none)", model, session_id or "(new)",
            len(prompt.encode("utf-8")), len(img_refs), cwd,
        )

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=_build_env(),
        )

        timed_out = [False]

        def _kill_on_timeout():
            timed_out[0] = True
            try:
                proc.kill()
            except OSError:
                pass

        timeout_timer = threading.Timer(_CLI_TIMEOUT, _kill_on_timeout)
        timeout_timer.start()

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
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                proc.stdin.close()
            except BrokenPipeError:
                pass

            got_text = False
            got_delta = False
            for raw_line in proc.stdout:
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
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
                    ctx_mgmt = msg.get("context_management")
                    if ctx_mgmt and ctx_mgmt.get("applied_edits"):
                        yield {"type": "context_compact", "edits": ctx_mgmt["applied_edits"]}
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
                                    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
                                    yield {"type": "tool", "tool": tool_name, "file": file_path}
                    else:
                        for block in msg.get("content", []):
                            if block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                if tool_name in ("Edit", "Write", "NotebookEdit"):
                                    tool_input = block.get("input", {})
                                    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
                                    yield {"type": "tool", "tool": tool_name, "file": file_path}

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

            proc.wait(timeout=10)
            stderr_thread.join(timeout=5)

            if timed_out[0]:
                # 超时被 kill：无论是否已流过部分正文，都给一个明确的中断提示，
                # 避免前端表现为"半句话静默结束"误以为正常完成。
                yield {"type": "error", "content": f"对话超时（已运行超过 {int(_CLI_TIMEOUT)} 秒）被中断。"}
            elif proc.returncode != 0 and not got_text:
                detail = "".join(stderr_chunks).strip() or f"exit code {proc.returncode}"
                logger.error("Claude CLI error: %s", detail)
                if resuming:
                    logger.info("Resume failed, retrying with full history...")
                    yield {"type": "text", "content": "> *Session 已过期，正在新建会话...*\n\n"}
                    yield from self.stream_chat(
                        page_path=page_path,
                        page_content=page_content,
                        selected_text=selected_text,
                        messages=messages,
                        model=model,
                        thinking=thinking,
                        effort=effort,
                        enable_tools=enable_tools,
                        images=images,
                        session_id="",
                    )
                    return
                yield {"type": "error", "content": "Claude CLI encountered an error. Please try again."}
        finally:
            timeout_timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            for p in img_abs_paths:
                try:
                    p.unlink()
                except OSError:
                    pass

    def suggest_edit(
        self,
        *,
        page_content: str,
        instruction: str,
        chat_context: str = "",
    ) -> str:
        """简单实现：让 CLI 在一次对话中输出完整修改后的 Markdown 文本。

        chat 模式 / 工具模式都不需要，只要原文 → 改写文本。我们直接用 -p 单轮调用、
        不开工具、不开 thinking。
        """
        cli = settings_service.claude_cli_path()
        prompt = (
            "请根据用户指令修改下面的 Markdown。\n"
            "只输出完整修改后的 Markdown 内容，不要解释，不要使用代码围栏。\n\n"
            f"## 用户指令\n\n{instruction}\n\n"
            f"## 对话上下文\n\n{chat_context or '(无)'}\n\n"
            f"## 原始 Markdown\n\n{page_content}"
        )
        cmd = [cli, "-p", "--model", _default_model(), "--thinking", "disabled"]
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            cwd=str(DOCS_ROOT),
            env=_build_env(),
            timeout=_CLI_TIMEOUT,
        )
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        # 简单去掉首尾代码围栏
        if out.startswith("```"):
            lines = out.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            out = "\n".join(lines).strip()
        return out
