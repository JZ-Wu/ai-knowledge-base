"""项目级常量。

运行时可变的设置（API key、所选 backend、模型列表、ACCESS_PASSWORD、Claude CLI 路径等）
统一搬到 `server/services/settings_service.py` 持久化到 `server/.settings.json`，
本文件只保留路径、字符上限、外部挂载等真常量 / 启动期一次性配置。
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# 仓库根目录 (server/ 的父目录) —— Docsify 前端静态资源根。
DOCS_ROOT = Path(__file__).parent.parent.resolve()

# 多知识库根目录：每个子目录 = 一个独立 KB。
KB_ROOT = DOCS_ROOT / "knowledge_bases"

# 默认知识库 slug。当 URL 没有 /kb/<slug>/ 前缀时回退到这个 KB。
DEFAULT_KB_SLUG = os.getenv("DEFAULT_KB_SLUG", "ai-ml-interview")

# 单页 markdown 内容传给 LLM 的最大字符数，防止超出上下文窗口。
MAX_PAGE_CHARS = 50000
# PDF 全文抽取出来的页面通常更长，给它单独的更大阈值。
MAX_PDF_CHARS = 200000
# PDF 按需视觉：页面渲染成 PNG 的分辨率（dpi）。150 足够看清图表/公式且体积适中。
PDF_RENDER_DPI = 150
# PDF 阅读模式下，prompt 里给模型作"定位用"的抽取文本上限（不再灌全文，靠它 grep + 看页图）。
MAX_PDF_INDEX_CHARS = 8000

# ── EXTERNAL_MOUNTS：跨盘符 / KB 外目录静态挂载 ──
#
# 设计动机：跨盘 Windows junction 在 DOCS_ROOT 内会触发 Starlette 的
# os.path.commonpath cross-drive ValueError，必须用独立 mount 绕开。也方便
# 把 KB 目录之外的某个文件夹（比如 F:/some-reports）以 /<prefix>/ 暴露出来给
# Docsify / AI sidebar 用。
#
# 通过环境变量 EXTERNAL_MOUNTS 配置（JSON），例如：
#   EXTERNAL_MOUNTS='{"external-reports":"F:/path/to/folder","notes":"D:/other"}'
#
# - key  : URL 前缀（不含开头斜杠，必须避开 "kb"、"api"、"server" 等保留前缀）
# - value: 绝对文件系统路径
# 不存在 / 不是目录的条目会被静默跳过，避免误配置卡住启动。
EXTERNAL_MOUNTS: dict[str, Path] = {}
_external_mounts_raw = os.getenv("EXTERNAL_MOUNTS", "").strip()
if _external_mounts_raw:
    try:
        _RESERVED_PREFIXES = {"kb", "api", "server", "knowledge_bases", "_trash", "_backup", "docs"}
        for prefix, path_str in json.loads(_external_mounts_raw).items():
            clean_prefix = prefix.strip("/")
            if clean_prefix in _RESERVED_PREFIXES or not clean_prefix:
                continue
            mount_path = Path(path_str).resolve()
            if mount_path.exists() and mount_path.is_dir():
                EXTERNAL_MOUNTS[clean_prefix] = mount_path
    except (json.JSONDecodeError, ValueError, OSError):
        pass
