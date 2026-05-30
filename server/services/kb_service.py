"""多知识库管理。

每个 KB 是 `knowledge_bases/<slug>/` 下的独立目录，slug 只能是 [A-Za-z0-9_-]{1,64}。
KB 元信息（名称、创建时间）放在 `knowledge_bases/<slug>/.kb/meta.json`。

适配自 lupin182/ai-knowledge-base，去掉了对旧 URL 路径的 legacy 兼容。
"""

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from server.config import DEFAULT_KB_SLUG, DOCS_ROOT, EXTERNAL_MOUNTS, KB_ROOT

_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_UPLOAD_SUFFIXES = {
    ".md", ".markdown", ".txt", ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".csv", ".tsv", ".json",
}
_WINDOWS_DEVICE_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


@dataclass(frozen=True)
class ResolvedPage:
    kb_slug: str
    rel_path: str
    abs_path: Path
    is_project_page: bool = False


def ensure_kb_root() -> None:
    KB_ROOT.mkdir(exist_ok=True)


def validate_slug(slug: str) -> str:
    slug = (slug or "").strip()
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError("Knowledge base slug must be 1-64 chars: letters, numbers, _ or -")
    return slug


def slugify_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    return slug[:48] or "kb"


def kb_dir(slug: str) -> Path:
    return KB_ROOT / validate_slug(slug)


def meta_path(slug: str) -> Path:
    return kb_dir(slug) / ".kb" / "meta.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_meta(slug: str) -> dict[str, Any]:
    path = meta_path(slug)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"slug": slug, "name": slug, "created_at": "", "updated_at": ""}


def _write_meta(slug: str, data: dict[str, Any]) -> None:
    path = meta_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_knowledge_base(name: str, slug: str | None = None) -> dict[str, Any]:
    ensure_kb_root()
    display_name = (name or "").strip()
    if not display_name:
        raise ValueError("Knowledge base name is required")

    base_slug = validate_slug(slug) if slug else validate_slug(slugify_name(display_name))
    final_slug = base_slug
    i = 2
    while kb_dir(final_slug).exists():
        final_slug = f"{base_slug}-{i}"
        validate_slug(final_slug)
        i += 1

    root = kb_dir(final_slug)
    root.mkdir(parents=True)
    now = _now_iso()
    data = {
        "slug": final_slug,
        "name": display_name,
        "created_at": now,
        "updated_at": now,
    }
    _write_meta(final_slug, data)
    readme = root / "README.md"
    readme.write_text(f"# {display_name}\n\n这是一个新的空知识库。\n", encoding="utf-8")
    return get_knowledge_base(final_slug)


def rename_knowledge_base(slug: str, name: str) -> dict[str, Any]:
    slug = validate_slug(slug)
    display_name = (name or "").strip()
    if not display_name:
        raise ValueError("Knowledge base name is required")
    if not kb_dir(slug).exists():
        raise FileNotFoundError(slug)
    meta = _read_meta(slug)
    meta["name"] = display_name
    meta["updated_at"] = _now_iso()
    _write_meta(slug, meta)
    return get_knowledge_base(slug)


def delete_knowledge_base(slug: str, trash_root: Path) -> dict[str, Any]:
    """把整个 KB 目录 git-style 搬到 trash_root/<timestamp>-<slug>/ 下，不物理删除。"""
    slug = validate_slug(slug)
    root = kb_dir(slug)
    if not root.exists():
        raise FileNotFoundError(slug)
    trash_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = trash_root / f"{stamp}-{slug}"
    shutil.move(str(root), str(target))
    return {"slug": slug, "trashed_at": str(target)}


def get_knowledge_base(slug: str) -> dict[str, Any]:
    slug = validate_slug(slug)
    root = kb_dir(slug)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(slug)
    meta = _read_meta(slug)
    stats = kb_stats(slug)
    return {
        "slug": slug,
        "name": meta.get("name") or slug,
        "created_at": meta.get("created_at", ""),
        "updated_at": meta.get("updated_at", ""),
        "files": stats["files"],
        "chars": stats["chars"],
    }


def list_knowledge_bases() -> list[dict[str, Any]]:
    ensure_kb_root()
    items = []
    for child in sorted(KB_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and _SLUG_RE.fullmatch(child.name):
            try:
                items.append(get_knowledge_base(child.name))
            except Exception:
                continue
    return items


def _clean_rel_path(raw: str) -> str:
    rel = (raw or "").strip().replace("\\", "/")
    while rel.startswith("/"):
        rel = rel[1:]
    parts = []
    for part in rel.split("/"):
        part = part.strip()
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError("Path traversal is not allowed")
        cleaned = part.rstrip(". ")
        if not cleaned or cleaned.startswith("."):
            raise ValueError("Hidden files and directories are not allowed")
        if cleaned.upper() in _WINDOWS_DEVICE_NAMES:
            raise ValueError("Reserved device names are not allowed")
        parts.append(cleaned)
    return "/".join(parts)


def resolve_kb_path(slug: str, rel_path: str = "", must_exist: bool = False) -> Path:
    slug = validate_slug(slug)
    root = kb_dir(slug).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(slug)
    clean = _clean_rel_path(rel_path)
    resolved = (root / clean).resolve() if clean else root
    if not resolved.is_relative_to(root):
        raise ValueError("Path must stay inside the knowledge base")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{slug}/{rel_path}")
    return resolved


def split_docsify_path(url_path: str) -> tuple[str | None, str]:
    """把 Docsify hash 路径切成 (slug, rel)。无 /kb/ 前缀返回 (None, rel)。"""
    path = (url_path or "").strip().strip("/")
    if not path:
        return None, ""
    parts = path.split("/", 2)
    if len(parts) >= 2 and parts[0] == "kb":
        slug = validate_slug(parts[1])
        rel = parts[2] if len(parts) > 2 else ""
        return slug, rel
    return None, path


# ── EXTERNAL_MOUNTS 路径解析 ──

def _match_external_mount(url_path: str) -> tuple[str, Path, str] | None:
    """url_path 命中某个外部挂载前缀时返回 (prefix, mount_root, sub_rel)。"""
    norm = (url_path or "").lstrip("/")
    for prefix, mount_root in EXTERNAL_MOUNTS.items():
        if norm == prefix:
            return prefix, mount_root, ""
        if norm.startswith(f"{prefix}/"):
            return prefix, mount_root, norm[len(prefix) + 1:]
    return None


def resolve_docsify_page(url_path: str) -> ResolvedPage:
    """把 Docsify 页面路径解析为具体 markdown 文件。

    解析顺序：
    1. `kb/<slug>/...` → KB_ROOT / slug / ...
    2. EXTERNAL_MOUNTS 任一前缀命中 → 外部挂载目录
    3. 空路径 → DOCS_ROOT/README.md（项目首页）
    4. 否则 → 视为 DEFAULT_KB_SLUG 下的相对路径
    """
    slug, rel = split_docsify_path(url_path)

    # EXTERNAL_MOUNTS 优先于 fallback 到 DEFAULT_KB_SLUG（用户配置的更"显式"）
    if slug is None:
        if not rel:
            readme = (DOCS_ROOT / "README.md").resolve()
            return ResolvedPage("", "README.md", readme, True)
        ext = _match_external_mount(rel)
        if ext is not None:
            prefix, mount_root, sub = ext
            return _resolve_under_mount(prefix, mount_root, sub)
        slug = DEFAULT_KB_SLUG

    clean = _clean_rel_path(rel)
    if not clean:
        clean = "README.md"
    if clean.endswith(".md"):
        candidate = resolve_kb_path(slug, clean)
        if candidate.exists():
            return ResolvedPage(slug, clean, candidate)
    else:
        candidate = resolve_kb_path(slug, f"{clean}.md")
        if candidate.exists():
            return ResolvedPage(slug, f"{clean}.md", candidate)
        candidate = resolve_kb_path(slug, f"{clean}/README.md")
        if candidate.exists():
            return ResolvedPage(slug, f"{clean}/README.md", candidate)
    raise FileNotFoundError(f"No markdown file found for: {url_path}")


def _resolve_under_mount(prefix: str, mount_root: Path, sub: str) -> ResolvedPage:
    """在某个外部挂载下解析 markdown 文件。"""
    sub = (sub or "").strip().replace("\\", "/").strip("/")
    # 命中 .md 直接给
    if sub.endswith(".md"):
        candidate = (mount_root / sub).resolve()
        if candidate.is_relative_to(mount_root) and candidate.exists():
            return ResolvedPage("", f"{prefix}/{sub}", candidate)
    # 试 sub.md
    candidate = (mount_root / f"{sub}.md").resolve() if sub else (mount_root / "README.md").resolve()
    if candidate.is_relative_to(mount_root) and candidate.exists():
        return ResolvedPage("", f"{prefix}/{sub}.md" if sub else f"{prefix}/README.md", candidate)
    # 试 sub/README.md
    candidate = (mount_root / sub / "README.md").resolve() if sub else (mount_root / "README.md").resolve()
    if candidate.is_relative_to(mount_root) and candidate.exists():
        return ResolvedPage("", f"{prefix}/{sub}/README.md" if sub else f"{prefix}/README.md", candidate)
    raise FileNotFoundError(f"{prefix}/{sub}")


def resolve_static_file(slug: str, rel_path: str) -> Path:
    clean = _clean_rel_path(rel_path)
    if not clean:
        clean = "README.md"
    candidates = [clean]
    if not Path(clean).suffix:
        candidates.extend([f"{clean}.md", f"{clean}/README.md"])
    for candidate in candidates:
        path = resolve_kb_path(slug, candidate)
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"{slug}/{rel_path}")


def iter_markdown_files(slug: str):
    root = resolve_kb_path(slug)
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        # 与 sync-content.mjs 的发布规则一致：跳过任一路径段以 '.' 或 '_'
        # 开头的目录（如 _extracted/），以及文件名以 '_' 开头的 .md（如 _sidebar.md）。
        if any(part.startswith((".", "_")) for part in rel_parts[:-1]):
            continue
        if path.name.startswith((".", "_")):
            continue
        yield path


def kb_stats(slug: str) -> dict[str, int]:
    files = 0
    chars = 0
    try:
        for path in iter_markdown_files(slug):
            files += 1
            try:
                chars += len(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return {"files": files, "chars": chars}


def all_kb_stats() -> dict[str, int]:
    files = 0
    chars = 0
    for item in list_knowledge_bases():
        files += item["files"]
        chars += item["chars"]
    return {"files": files, "chars": chars}


def _title_for_path(path: Path) -> str:
    if path.name.lower() == "readme.md":
        return "总览"
    return path.stem.replace("_", " ")


def _kb_markdown_link(slug: str, rel_path: str = "") -> str:
    slug_part = quote(validate_slug(slug), safe="")
    if not rel_path:
        return f"/kb/{slug_part}/"
    return f"/kb/{slug_part}/{quote(rel_path, safe='/')}"


def _tree_for_dir(slug: str, directory: Path, prefix: str = "") -> list[str]:
    root = kb_dir(slug)
    lines: list[str] = []
    children = sorted(
        # 跳过 . / _ 前缀的目录与文件，与 iter_markdown_files 的发布规则一致
        # （否则 _extracted/、_sidebar.md 等不发布内容会进侧栏，点开即死链 + 计数口径不一致）。
        [p for p in directory.iterdir() if not p.name.startswith((".", "_"))],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )
    for child in children:
        if child.is_dir():
            if any(p.suffix.lower() == ".md" for p in child.rglob("*.md")):
                lines.append(f"{prefix}- {child.name}")
                lines.extend(_tree_for_dir(slug, child, prefix + "  "))
        elif child.suffix.lower() == ".md":
            rel = child.relative_to(root).as_posix()
            title = _title_for_path(child)
            lines.append(f"{prefix}- [{title}]({_kb_markdown_link(slug, rel)})")
    return lines


def build_sidebar_markdown() -> str:
    lines = ["- [首页](/)", "", "- 知识库"]
    items = list_knowledge_bases()
    if not items:
        lines.append("  - 暂无知识库")
        return "\n".join(lines) + "\n"
    for item in items:
        slug = item["slug"]
        name = item["name"]
        lines.append(f"  - [{name}]({_kb_markdown_link(slug)})")
        root = kb_dir(slug)
        if root.exists():
            lines.extend(_tree_for_dir(slug, root, "    "))
    return "\n".join(lines) + "\n"


def _validate_upload_path(raw: str) -> str:
    clean = _clean_rel_path(raw)
    if not clean:
        raise ValueError("Upload path is empty")
    suffix = Path(clean).suffix.lower()
    if suffix and suffix not in _SAFE_UPLOAD_SUFFIXES:
        raise ValueError(f"File type is not allowed: {suffix}")
    return clean


async def save_uploads(slug: str, form: Any) -> dict[str, Any]:
    root = resolve_kb_path(slug)
    files = form.getlist("files")
    rel_paths = form.getlist("relative_paths")
    saved: list[str] = []
    errors: list[dict[str, str]] = []
    for i, upload in enumerate(files):
        rel_raw = rel_paths[i] if i < len(rel_paths) else getattr(upload, "filename", "")
        try:
            rel = _validate_upload_path(rel_raw or getattr(upload, "filename", ""))
            target = (root / rel).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Path must stay inside the knowledge base")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = await upload.read()
            target.write_bytes(data)
            saved.append(rel)
        except Exception as exc:
            errors.append({"path": str(rel_raw), "error": str(exc)})
    meta = _read_meta(slug)
    meta["updated_at"] = _now_iso()
    _write_meta(slug, meta)
    return {"saved": saved, "errors": errors}


# ── 路径校验（提供给老的 file_service 兼容接口）──

def validate_path(file_path: str, *, allow_external: bool = False) -> Path:
    """把任意来源的相对路径解析为可读绝对路径。

    - `knowledge_bases/<slug>/...` 形式 → 解析到该 KB 内
    - `<external-prefix>/...`（命中 EXTERNAL_MOUNTS） → 解析到外部挂载（仅 allow_external 为 True 时）
    - 其它 → 按 docsify 形式解析

    用于 /api/page-source 读取时 allow_external=True；用于 /api/apply-edit 写入时
    保持默认 False —— 外部挂载是其他仓库 / 文件夹，从 KB UI 写回会让两边发散，禁掉。
    """
    raw = (file_path or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("file_path is empty")
    if raw.startswith("knowledge_bases/"):
        rest = raw[len("knowledge_bases/"):]
        parts = rest.split("/", 1)
        if len(parts) < 2:
            raise ValueError("file_path must include a slug and a relative path")
        slug, rel = parts[0], parts[1]
        return resolve_kb_path(slug, rel)
    # 外部挂载（读路径用）
    ext = _match_external_mount(raw)
    if ext is not None:
        if not allow_external:
            raise ValueError("Cannot write to a path under EXTERNAL_MOUNTS")
        _, mount_root, sub = ext
        candidate = (mount_root / sub).resolve() if sub else mount_root.resolve()
        if not candidate.is_relative_to(mount_root):
            raise ValueError(f"Path traversal rejected: {file_path}")
        return candidate
    # 否则按 docsify 形式解析
    resolved = resolve_docsify_page(raw)
    return resolved.abs_path
