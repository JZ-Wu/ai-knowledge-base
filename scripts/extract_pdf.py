"""抽取每个 KB 的 papers/<name>.pdf 全文为同目录 papers/<name>.md。

输出格式：frontmatter (source_pdf / extracted_by) + 每页 ## Page N 标记。后端
（server/backends/{claude_cli,openai_api}.py）用 frontmatter 的 `source_pdf:` 检测，
进入 PDF 阅读模式，把全文喂给 AI 助手——这就是论文卡片里"内嵌阅读 + Ask AI"的上下文来源。

布局（2026-05 起）：内容都在 knowledge_bases/<slug>/papers/*.pdf。本脚本遍历所有 KB。

用法：
  python scripts/extract_pdf.py            # 抽取所有 KB 里缺 .md 的 PDF（跳过已存在的）
  python scripts/extract_pdf.py --force    # 重新抽取（覆盖已存在的 .md）
  python scripts/extract_pdf.py <slug> ... # 只处理指定 KB
"""
from __future__ import annotations
import sys
from pathlib import Path

import pymupdf  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
KB_ROOT = ROOT / "knowledge_bases"


def extract(pdf_path: Path, paper_name: str) -> str:
    doc = pymupdf.open(pdf_path)
    parts = [
        "---",
        f"paper: {paper_name}",
        f"source_pdf: {pdf_path.name}",
        "extracted_by: pymupdf",
        "purpose: full-text context for AI assistant when reading the PDF",
        "---",
        "",
        f"# {paper_name} — Extracted Full Text",
        "",
        f"> Automatic extraction from {pdf_path.name}. Layout may be imperfect; the PDF "
        f"remains the authoritative source. This file exists so the AI assistant can "
        f"answer questions with full paper context.",
        "",
    ]
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        parts.append(f"## Page {i}")
        parts.append("")
        parts.append(text.rstrip())
        parts.append("")
    doc.close()
    return "\n".join(parts)


def iter_paper_pdfs(slugs: list[str]):
    """yield (slug, pdf_path) for every papers/*.pdf under the requested KBs."""
    if not KB_ROOT.is_dir():
        return
    for kb_dir in sorted(KB_ROOT.iterdir()):
        if not kb_dir.is_dir():
            continue
        slug = kb_dir.name
        if slugs and slug not in slugs:
            continue
        papers_dir = kb_dir / "papers"
        if not papers_dir.is_dir():
            continue
        for pdf in sorted(papers_dir.glob("*.pdf")):
            yield slug, pdf


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    slugs = [a for a in args if not a.startswith("--")]

    created = skipped = failed = 0
    for slug, pdf in iter_paper_pdfs(slugs):
        out = pdf.with_suffix(".md")
        if out.exists() and not force:
            print(f"[skip] {slug}/{pdf.name} (extraction exists; --force to overwrite)")
            skipped += 1
            continue
        try:
            text = extract(pdf, pdf.stem)
            out.write_text(text, encoding="utf-8")
            print(f"[ok]   {slug}/{pdf.name} -> {out.name} ({len(text):,} chars)")
            created += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {slug}/{pdf.name}: {exc}")
            failed += 1

    print(f"\nDone. created={created} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
