from pathlib import Path
from server.config import DOCS_ROOT


def validate_path(file_path: str) -> Path:
    """校验路径在知识库根目录下，防止路径穿越。"""
    resolved = (DOCS_ROOT / file_path).resolve()
    if not resolved.is_relative_to(DOCS_ROOT):
        raise ValueError(f"Path traversal rejected: {file_path}")
    return resolved


def resolve_docsify_path(url_path: str) -> Path:
    """将 Docsify hash 路径解析为 markdown 文件路径。

    Examples:
        "大模型/基础理论/Transformer架构详解" -> DOCS_ROOT/大模型/基础理论/Transformer架构详解.md
        "" or "/" -> DOCS_ROOT/README.md
    """
    path = url_path.strip("/")
    if not path:
        return validate_path("README.md")

    # 如果已经带 .md 后缀
    if path.endswith(".md"):
        return validate_path(path)

    # 尝试直接加 .md
    candidate = DOCS_ROOT / f"{path}.md"
    if candidate.exists():
        return validate_path(f"{path}.md")

    # 尝试作为目录的 README.md
    candidate = DOCS_ROOT / path / "README.md"
    if candidate.exists():
        return validate_path(f"{path}/README.md")

    raise FileNotFoundError(f"No markdown file found for: {url_path}")


def read_source(file_path: str) -> str:
    """读取 markdown 文件内容。"""
    resolved = validate_path(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return resolved.read_text(encoding="utf-8")


def write_source(file_path: str, content: str) -> Path:
    """写入 markdown 文件，创建 .bak 备份。"""
    resolved = validate_path(file_path)
    # Only allow writing markdown files
    if resolved.suffix.lower() != ".md":
        raise ValueError(f"Only .md files can be edited: {file_path}")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 创建备份
    backup = resolved.with_suffix(resolved.suffix + ".bak")
    backup.write_text(resolved.read_text(encoding="utf-8"), encoding="utf-8")

    # 写入新内容
    resolved.write_text(content, encoding="utf-8")
    return resolved
