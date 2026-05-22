"""OpenAI-compatible API backend：对接 OpenAI / DeepSeek / 任何兼容 Chat Completions 的服务。

工具调用走服务器侧实现：模型给 tool_call → 我们执行 → 把结果 append 回去 → 继续流。
工具白名单只覆盖 Markdown 文件 + 严格的 KB 范围内路径。

适配自 lupin182/ai-knowledge-base 的 llm_service.py，调整为：
- 配置全部从 `settings_service` 读，不再依赖 AI_API_KEY 等环境变量
- 实现 Backend Protocol：方法名 / 签名与 claude_cli backend 对齐
"""

import json
import logging
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx

from server.config import DOCS_ROOT, MAX_PAGE_CHARS, MAX_PDF_CHARS
from server.services import kb_service, settings_service

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger.setLevel(logging.INFO)

_DANGEROUS_NAME_RE = re.compile(r"(secret|credential|password|token|api[_-]?key)", re.I)

_SYSTEM_PROMPT = (
    "你是一个知识库助手，帮助用户浏览和编辑一个 Markdown 知识库。\n\n"
    "## 行为规则\n"
    "1. 请用中文回答，使用清晰的 Markdown 格式。\n"
    "2. 默认只回答问题和解释内容，不要修改任何文件。\n"
    "3. 只有当用户明确要求修改、添加、删除或整理知识库内容时，才调用工具编辑文件。\n"
    "4. 编辑完成后说明改了哪些文件和内容，让用户刷新页面查看。\n"
    "5. 用户没有明确要求时，不要主动创建新文件。\n\n"
    "## 文件访问规则\n"
    "1. 只能读取、搜索、创建和修改 Markdown (.md) 文件。\n"
    "2. 工具参数中的文件路径必须使用知识库根目录下的相对路径，例如 `大模型/README.md`；"
    "如果工具支持 `kb_slug`，必须优先使用当前页面提示里的知识库 slug。\n"
    "3. 禁止访问隐藏目录、server/、.git/、docs/vendor/ 以及任何疑似密钥文件。\n"
    "4. 禁止使用 ../ 或绝对路径访问知识库外部。\n"
    "5. 如果要编辑当前页面，优先使用提示中给出的 `当前页面路径`。\n\n"
    "## 编辑规则\n"
    "1. 小范围修改优先调用 replace_markdown，整篇重写或新建文件才调用 write_markdown。\n"
    "2. 调用 replace_markdown 时 old_text 必须逐字来自原文；如果不确定，先 read_markdown。\n"
    "3. 保持原有 Markdown 风格、中文表达和相对链接格式。\n"
)

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_markdown",
            "description": "读取知识库内一个 Markdown 文件的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_slug": {"type": "string", "description": "知识库 slug；省略时使用当前页面所属知识库。"},
                    "file_path": {"type": "string", "description": "知识库根目录下的相对 .md 路径。"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_markdown",
            "description": "在一个 Markdown 文件中把 old_text 替换为 new_text。适合局部修改。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_slug": {"type": "string"},
                    "file_path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["file_path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_markdown",
            "description": "创建或完整覆盖一个 Markdown 文件。已有文件会自动创建 .bak 备份。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_slug": {"type": "string"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_markdown",
            "description": "在知识库 Markdown 文件中搜索关键词，返回匹配文件、行号和片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_slug": {"type": "string"},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_markdown_files",
            "description": "列出某个目录下的 Markdown 文件路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_slug": {"type": "string"},
                    "directory": {"type": "string", "default": ""},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 300, "default": 100},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# ── settings 读 helper ──
def _api_cfg() -> dict[str, Any]:
    return settings_service.read()["openai_api"]


def _resolve_profile(model_selection: str) -> dict[str, Any]:
    cfg = _api_cfg()
    profiles: list[dict[str, Any]] = cfg.get("models") or []
    if not profiles:
        raise RuntimeError("No model profile configured. Open Settings to add one.")
    default_key = cfg.get("default_model_key") or profiles[0]["key"]
    requested = (model_selection or "").strip()
    if not cfg.get("allow_client_model", True) or not requested or requested == "api-default":
        return next((p for p in profiles if p["key"] == default_key), profiles[0])
    for p in profiles:
        if requested in {p.get("key"), p.get("name"), p.get("model")}:
            return p
    return next((p for p in profiles if p["key"] == default_key), profiles[0])


def _chat_url(profile: dict[str, Any]) -> str:
    base = (profile.get("api_base_url") or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _headers(profile: dict[str, Any]) -> dict[str, str]:
    if not (profile.get("api_key") or "").strip():
        raise RuntimeError(f"API key is not configured for model '{profile.get('name')}'.")
    return {"Authorization": f"Bearer {profile['api_key']}", "Content-Type": "application/json"}


def _format_api_error(resp: httpx.Response) -> str:
    try:
        body = resp.text
    except Exception:
        body = ""
    try:
        data = json.loads(body) if body else {}
        err = data.get("error", {})
        if isinstance(err, dict):
            return err.get("message") or json.dumps(err, ensure_ascii=False)
        if err:
            return str(err)
    except Exception:
        pass
    return body[:1000] or f"HTTP {resp.status_code}"


def _iter_sse_json(resp: httpx.Response):
    for line in resp.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-json SSE payload: %s", data[:200])


def _stream_completion(messages, profile, tools):
    cfg = _api_cfg()
    payload: dict[str, Any] = {
        "model": profile["model"],
        "messages": messages,
        "stream": True,
        "temperature": cfg.get("temperature", 0.2),
    }
    if tools:
        payload["tools"] = _TOOLS
        payload["tool_choice"] = "auto"
    if cfg.get("include_usage", False):
        payload["stream_options"] = {"include_usage": True}
    timeout = httpx.Timeout(connect=30, read=cfg.get("request_timeout", 300), write=30, pool=30)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", _chat_url(profile), headers=_headers(profile), json=payload) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM API error ({resp.status_code}): {_format_api_error(resp)}")
            yield from _iter_sse_json(resp)


def _completion(messages, profile) -> str:
    cfg = _api_cfg()
    payload = {
        "model": profile["model"],
        "messages": messages,
        "stream": False,
        "temperature": cfg.get("temperature", 0.2),
    }
    timeout = httpx.Timeout(connect=30, read=cfg.get("request_timeout", 300), write=30, pool=30)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(_chat_url(profile), headers=_headers(profile), json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM API error ({resp.status_code}): {_format_api_error(resp)}")
    return resp.json()["choices"][0]["message"].get("content", "")


# ── 工具执行 ──
def _tool_kb_slug(args: dict[str, Any], default_kb_slug: str) -> str:
    requested = str(args.get("kb_slug") or "").strip()
    if requested:
        return kb_service.validate_slug(requested)
    if default_kb_slug:
        return kb_service.validate_slug(default_kb_slug)
    raise ValueError("kb_slug is required outside a knowledge-base page")


def _split_tool_file_path(file_path: str, default_kb_slug: str) -> tuple[str, str]:
    raw = (file_path or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("file_path is required")
    slug, rel = kb_service.split_docsify_path(raw)
    if slug:
        return slug, rel
    if default_kb_slug:
        return kb_service.validate_slug(default_kb_slug), raw
    raise ValueError("file_path must start with kb/<slug>/ when no current KB is active")


def _resolve_markdown_path(file_path: str, kb_slug: str, default_kb_slug: str, must_exist: bool) -> Path:
    slug, rel = _split_tool_file_path(file_path, kb_slug or default_kb_slug)
    resolved = kb_service.resolve_kb_path(slug, rel, must_exist=must_exist)
    if resolved.suffix.lower() != ".md":
        raise ValueError("only Markdown files can be accessed")
    if _DANGEROUS_NAME_RE.search(resolved.name):
        raise ValueError("sensitive-looking filenames are not allowed")
    return resolved


def _rel(path: Path) -> str:
    for item in kb_service.list_knowledge_bases():
        slug = item["slug"]
        root = kb_service.kb_dir(slug).resolve()
        try:
            rel = path.resolve().relative_to(root).as_posix()
            return f"kb/{slug}/{rel}"
        except ValueError:
            continue
    return path.resolve().relative_to(DOCS_ROOT).as_posix()


def _write_text_with_backup(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(content, encoding="utf-8")


def _tool_read_markdown(args, default_kb_slug):
    path = _resolve_markdown_path(str(args.get("file_path", "")), str(args.get("kb_slug") or ""), default_kb_slug, must_exist=True)
    content = path.read_text(encoding="utf-8")
    truncated = False
    if len(content) > MAX_PDF_CHARS:
        content = content[:MAX_PDF_CHARS] + "\n\n... (内容已截断)"
        truncated = True
    return {"ok": True, "file_path": _rel(path), "content": content, "truncated": truncated}


def _tool_replace_markdown(args, default_kb_slug):
    path = _resolve_markdown_path(str(args.get("file_path", "")), str(args.get("kb_slug") or ""), default_kb_slug, must_exist=True)
    old_text = str(args.get("old_text", ""))
    new_text = str(args.get("new_text", ""))
    if not old_text:
        raise ValueError("old_text is required")
    original = path.read_text(encoding="utf-8")
    if old_text not in original:
        raise ValueError("old_text was not found in the target file")
    _write_text_with_backup(path, original.replace(old_text, new_text, 1))
    return {"ok": True, "file_path": _rel(path), "operation": "replace", "changed": True}


def _tool_write_markdown(args, default_kb_slug):
    path = _resolve_markdown_path(str(args.get("file_path", "")), str(args.get("kb_slug") or ""), default_kb_slug, must_exist=False)
    existed = path.exists()
    _write_text_with_backup(path, str(args.get("content", "")))
    return {"ok": True, "file_path": _rel(path), "operation": "update" if existed else "create", "changed": True}


def _tool_search_markdown(args, default_kb_slug):
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    max_results = max(1, min(int(args.get("max_results") or 10), 30))
    needle = query.lower()
    results: list[dict[str, Any]] = []
    slug = _tool_kb_slug(args, default_kb_slug)
    for path in kb_service.iter_markdown_files(slug):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            if needle in line.lower():
                snippet = line.strip()
                if len(snippet) > 220:
                    snippet = snippet[:220] + "..."
                results.append({"file_path": _rel(path), "line": line_no, "text": snippet})
                if len(results) >= max_results:
                    return {"ok": True, "query": query, "results": results}
                break
    return {"ok": True, "query": query, "results": results}


def _tool_list_markdown_files(args, default_kb_slug):
    directory = str(args.get("directory", "") or "").strip().replace("\\", "/")
    max_results = max(1, min(int(args.get("max_results") or 100), 300))
    slug = _tool_kb_slug(args, default_kb_slug)
    root = kb_service.resolve_kb_path(slug, directory, must_exist=True) if directory else kb_service.resolve_kb_path(slug)
    if directory and not root.is_dir():
        raise ValueError("directory not found")
    files = []
    for p in root.rglob("*.md"):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(kb_service.kb_dir(slug)).parts):
            files.append(_rel(p))
    return {"ok": True, "files": sorted(files)[:max_results], "truncated": len(files) > max_results}


def _execute_tool(name, raw_args, default_kb_slug):
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        if name == "read_markdown":
            return _tool_read_markdown(args, default_kb_slug), None
        if name == "replace_markdown":
            r = _tool_replace_markdown(args, default_kb_slug)
            return r, r["file_path"]
        if name == "write_markdown":
            r = _tool_write_markdown(args, default_kb_slug)
            return r, r["file_path"]
        if name == "search_markdown":
            return _tool_search_markdown(args, default_kb_slug), None
        if name == "list_markdown_files":
            return _tool_list_markdown_files(args, default_kb_slug), None
        raise ValueError(f"unknown tool: {name}")
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, None


def _build_prompt(page_path, page_content, selected_text, messages, thinking):
    is_pdf_extract = page_content.lstrip().startswith("---") and "source_pdf:" in page_content[:800]
    max_chars = MAX_PDF_CHARS if is_pdf_extract else MAX_PAGE_CHARS
    if len(page_content) > max_chars:
        page_content = page_content[:max_chars] + "\n\n... (内容已截断)"

    parts = [
        "你是一个 AI 知识库助手，帮助用户浏览和编辑一个 Markdown 知识库。",
        "请用中文回答，使用清晰的 Markdown 格式。",
        "默认只回答问题和解释内容，不要修改任何文件。",
        "只有当用户明确要求修改/添加/删除知识库内容时，才编辑对应的 markdown 文件。",
        f"\n## 当前页面路径\n\n{page_path or 'README.md'}",
    ]
    kb_slug, rel_path = kb_service.split_docsify_path(page_path or "")
    if kb_slug:
        parts.append(
            f"\n## 当前知识库\n\n- slug: `{kb_slug}`\n- 当前页相对路径: `{rel_path or 'README.md'}`\n"
            "工具参数优先使用这个 `kb_slug` 和相对路径。"
        )
    if thinking:
        parts.append("\n## 深度模式\n\n请在内部更仔细地分析问题，但最终只输出必要结论。")
    if is_pdf_extract:
        parts.append(
            "\n## 阅读模式提示\n\n当前页面是 PDF 自动抽取全文。回答规则：\n"
            "1. **严格基于该 PDF 内容作答**，引用时标注页码。\n"
            "2. 不在 PDF 范围内的问题，明确说「这不在这篇论文里」。\n"
            "3. 围绕用户选中的段落回答。\n"
            "4. **不要改写任何文件**。"
        )
    parts.append(f"\n## 当前页面内容\n\n{page_content}")
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


def _make_initial_messages(prompt, images):
    content: str | list[dict[str, Any]] = prompt
    if images:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            b64 = image.get("base64", "")
            media_type = image.get("media_type", "image/png")
            if not b64:
                continue
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}})
        content = blocks
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _accumulate_tool_call(acc, delta):
    idx = int(delta.get("index", 0))
    call = acc.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
    if delta.get("id"):
        call["id"] = delta["id"]
    if delta.get("type"):
        call["type"] = delta["type"]
    fn = delta.get("function") or {}
    if fn.get("name"):
        call["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        call["function"]["arguments"] += fn["arguments"]


def _completed_tool_calls(acc):
    calls = []
    for idx in sorted(acc):
        call = acc[idx]
        if not call.get("id"):
            call["id"] = f"call_{idx}"
        calls.append(call)
    return calls


class OpenAIAPIBackend:
    name = "openai_api"

    def list_models(self) -> list[dict[str, Any]]:
        cfg = _api_cfg()
        profiles = cfg.get("models") or []
        default_key = cfg.get("default_model_key") or (profiles[0]["key"] if profiles else "")
        return [
            {
                "value": p["key"],
                "label": (p.get("name") or "").strip() or (p.get("model") or "").strip() or p["key"],
                "model": p.get("model"),
                "context": p.get("context", 0),
                "configured": bool((p.get("api_key") or "").strip()),
                "is_default": p["key"] == default_key,
            }
            for p in profiles
        ]

    def status(self) -> dict[str, Any]:
        cfg = _api_cfg()
        profiles = cfg.get("models") or []
        configured = [p for p in profiles if (p.get("api_key") or "").strip()]
        return {
            "backend": self.name,
            "configured_count": len(configured),
            "available": bool(configured),
        }

    def stream_chat(
        self,
        *,
        page_path: str,
        page_content: str,
        selected_text: str,
        messages: list[dict[str, Any]],
        model: str = "",
        thinking: bool = False,
        images: list[dict[str, Any]] | None = None,
        session_id: str = "",
    ) -> Generator[dict[str, Any], None, None]:
        del session_id  # API 模式无服务端会话；前端发完整历史。

        cfg = _api_cfg()
        try:
            profile = _resolve_profile(model)
        except RuntimeError as exc:
            yield {"type": "error", "content": str(exc)}
            return

        default_kb_slug, _ = kb_service.split_docsify_path(page_path or "")
        default_kb_slug = default_kb_slug or ""
        prompt = _build_prompt(page_path, page_content, selected_text, messages, thinking)
        api_messages = _make_initial_messages(prompt, images)
        enable_tools = cfg.get("enable_tools", True)
        max_rounds = cfg.get("max_tool_rounds", 5)

        logger.info(
            "openai_api call: profile=%s model=%s prompt=%dB images=%d",
            profile["key"], profile["model"], len(prompt.encode("utf-8")), len(images or []),
        )

        for round_idx in range(max_rounds + 1):
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            content_parts: list[str] = []
            reasoning_parts: list[str] = []

            try:
                for event in _stream_completion(api_messages, profile, tools=enable_tools):
                    usage = event.get("usage")
                    if usage:
                        yield {
                            "type": "usage",
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "cache_read": 0,
                            "cache_create": 0,
                        }
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {})
                    text = delta.get("content")
                    if text:
                        content_parts.append(text)
                        yield {"type": "text", "content": text}
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        yield {"type": "thinking", "content": reasoning}
                    for tool_delta in delta.get("tool_calls") or []:
                        _accumulate_tool_call(tool_calls_acc, tool_delta)
            except Exception as exc:
                logger.error("LLM stream error: %s", exc)
                yield {"type": "error", "content": str(exc)}
                return

            tool_calls = _completed_tool_calls(tool_calls_acc)
            if not tool_calls:
                return

            if round_idx >= max_rounds:
                yield {"type": "error", "content": "Tool call limit exceeded."}
                return

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": tool_calls,
            }
            if reasoning_parts:
                assistant_message["reasoning_content"] = "".join(reasoning_parts)
            api_messages.append(assistant_message)

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                result, edited_file = _execute_tool(name, args, default_kb_slug)
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
                if edited_file:
                    yield {
                        "type": "tool",
                        "tool": "Edit" if name == "replace_markdown" else "Write",
                        "file": edited_file,
                    }

    def suggest_edit(
        self,
        *,
        page_content: str,
        instruction: str,
        chat_context: str = "",
    ) -> str:
        profile = _resolve_profile("")
        prompt = (
            "请根据用户指令修改下面的 Markdown。\n"
            "只输出完整修改后的 Markdown 内容，不要解释，不要使用代码围栏。\n\n"
            f"## 用户指令\n\n{instruction}\n\n"
            f"## 对话上下文\n\n{chat_context or '(无)'}\n\n"
            f"## 原始 Markdown\n\n{page_content}"
        )
        messages = [
            {"role": "system", "content": "你是严谨的 Markdown 编辑助手。"},
            {"role": "user", "content": prompt},
        ]
        text = _completion(messages, profile).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text
