"""把 PDF 按页渲染成 PNG，供 AI 助手按需"看图"（原生视觉：图/表/公式/版面）。

为什么不用 CLI 内置的 Read 直接读 PDF：claude CLI 2.1.x 的 Read 对这些 arxiv PDF
会误报 "PDF is password-protected"（pymupdf 证实并未加密），读不了。改用 pymupdf
渲染成 PNG，走模型已验证可用的图片视觉通道。

懒渲染 + 磁盘缓存：第一次用到某篇 PDF 时才渲染它的页，缓存到 papers/.pages/<stem>/，
之后复用。缓存目录是 `.` 前缀 + 已 gitignore，不发布、不入 git。
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:  # pragma: no cover - 环境缺 pymupdf 时优雅降级
    _HAVE_FITZ = False


def ensure_pages(pdf_path: Path, cache_dir: Path, dpi: int = 150) -> int:
    """确保 pdf 的每页都渲染成 cache_dir/pNNN.png。

    懒：缓存里 PNG 数与 PDF 页数一致就直接跳过。返回渲染/已存在的页数；
    pymupdf 不可用、PDF 不存在或渲染失败时返回 0（调用方据此回退到纯文本）。
    """
    if not _HAVE_FITZ or not pdf_path.is_file():
        return 0
    try:
        doc = fitz.open(pdf_path)
        n = doc.page_count
        existing = sorted(cache_dir.glob("p*.png")) if cache_dir.exists() else []
        if n > 0 and len(existing) == n:
            doc.close()
            return n  # 已缓存且页数一致
        cache_dir.mkdir(parents=True, exist_ok=True)
        for old in existing:  # 页数变了：清旧重渲
            try:
                old.unlink()
            except OSError:
                pass
        for i in range(n):
            pix = doc.load_page(i).get_pixmap(dpi=dpi)
            pix.save(str(cache_dir / f"p{i + 1:03d}.png"))
        doc.close()
        return n
    except Exception as exc:  # pragma: no cover
        logger.warning("PDF render failed for %s: %s", pdf_path, exc)
        return 0
