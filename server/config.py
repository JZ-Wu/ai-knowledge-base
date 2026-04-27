import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Claude CLI 路径
_default_claude = shutil.which("claude")
if not _default_claude:
    # Windows npm global bin 的常见路径
    _npm_path = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
    if _npm_path.exists():
        _default_claude = str(_npm_path)
    else:
        _default_claude = "claude"
CLAUDE_CLI = os.getenv("CLAUDE_CLI", _default_claude)

# 知识库根目录 (server/ 的父目录)
DOCS_ROOT = Path(__file__).parent.parent.resolve()

# 最大页面内容长度 (字符数，防止超出 context window)
# 普通 .md 页面用 MAX_PAGE_CHARS；PDF 全文提取文件用 MAX_PDF_CHARS（通常更长）
MAX_PAGE_CHARS = 50000
MAX_PDF_CHARS = 200000

# 局域网访问密码 (为空则不启用认证)
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")
