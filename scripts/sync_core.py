#!/usr/bin/env python3
"""
sync_core.py — 把 kb-core 仓库的框架文件 mirror 到本 KB。

用法：
    python scripts/sync_core.py          # 默认：先 git pull kb-core，再 mirror
    python scripts/sync_core.py --no-pull  # 跳过 git pull，只 mirror 本地状态
    python scripts/sync_core.py --dry-run  # 只打印将要发生的变更
    python scripts/sync_core.py --core PATH  # 自定义 kb-core 路径

默认假设 kb-core 在本 KB 的 sibling：../kb-core

同步语义：mirror（会覆盖旧文件、会删除 kb-core 中已经不存在的文件）。
不要在 KB 里直接改 SYNC_PATHS 下的文件，下次 sync 会被冲掉——去 kb-core 改、
push、再 sync。
"""

import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

# 框架路径——必须和 kb-core README 里列的一致
SYNC_PATHS = [
    "docs/js",
    "docs/css",
    "docs/vendor",
    "docs/tools",
    "server",
    ".claude/skills",
    "run.py",
    "start.sh",
    "start_server.bat",
    "INSTALL.md",
]

# 同步时两端都忽略的文件（既不从 kb-core 复制、也不从 KB 端删除）
# 关键：.env 和 .auth_secret 是每个 KB 自己的本地秘密，绝不能 mirror。
# .env.example 是模板文件（不带 .env 这个精确名字），仍会被同步。
IGNORE_NAMES = {
    "__pycache__", ".DS_Store", "Thumbs.db",
    ".env", ".env.local", ".auth_secret",
}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


def should_ignore(path: Path) -> bool:
    if path.name in IGNORE_NAMES:
        return True
    if path.suffix in IGNORE_SUFFIXES:
        return True
    return False


def list_files(root: Path):
    """Yield all non-ignored files under root, relative to root."""
    if not root.exists():
        return
    if root.is_file():
        if not should_ignore(root):
            yield Path(".")
        return
    for p in root.rglob("*"):
        if should_ignore(p):
            continue
        if any(part in IGNORE_NAMES for part in p.parts):
            continue
        if p.is_file():
            yield p.relative_to(root)


def mirror_path(rel: str, core_root: Path, kb_root: Path, dry_run: bool):
    """Mirror one SYNC_PATHS entry from core into kb. Returns (added, updated, deleted) counts."""
    src = core_root / rel
    dst = kb_root / rel
    added = updated = deleted = 0

    if not src.exists():
        # core 端不存在该路径——KB 端也应该删除
        if dst.exists():
            print(f"  [DELETE-PATH] {rel} (not in core)")
            if not dry_run:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
                deleted += 1
        return added, updated, deleted

    if src.is_file():
        # 单文件
        need_copy = (not dst.exists()) or (not filecmp.cmp(src, dst, shallow=False))
        if need_copy:
            tag = "ADD" if not dst.exists() else "UPDATE"
            print(f"  [{tag}] {rel}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            if tag == "ADD":
                added += 1
            else:
                updated += 1
        return added, updated, deleted

    # 目录：双向遍历
    src_files = set(list_files(src))
    dst_files = set(list_files(dst)) if dst.exists() else set()

    # 删除 KB 端有但 core 端没有的文件
    for rel_file in sorted(dst_files - src_files):
        target = dst / rel_file
        print(f"  [DELETE] {rel}/{rel_file.as_posix()}")
        if not dry_run:
            target.unlink()
            # 清理空目录
            for parent in target.parents:
                if parent == dst:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
        deleted += 1

    # 复制 / 更新
    for rel_file in sorted(src_files):
        s = src / rel_file
        d = dst / rel_file
        if d.exists() and filecmp.cmp(s, d, shallow=False):
            continue
        tag = "ADD" if not d.exists() else "UPDATE"
        print(f"  [{tag}] {rel}/{rel_file.as_posix()}")
        if not dry_run:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        if tag == "ADD":
            added += 1
        else:
            updated += 1

    return added, updated, deleted


def git_pull(core_root: Path) -> str:
    """Run git pull in core_root. Returns the resulting HEAD sha (short)."""
    subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=core_root,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=core_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--core", type=Path, default=None, help="kb-core path (default: ../kb-core)")
    parser.add_argument("--no-pull", action="store_true", help="skip git pull on kb-core")
    parser.add_argument("--dry-run", action="store_true", help="print changes without applying them")
    args = parser.parse_args()

    kb_root = Path(__file__).resolve().parents[1]
    core_root = (args.core or kb_root.parent / "kb-core").resolve()

    if not core_root.exists():
        print(f"[error] kb-core not found at: {core_root}", file=sys.stderr)
        print("        clone it sibling to this KB, or pass --core PATH", file=sys.stderr)
        sys.exit(2)
    if not (core_root / ".git").exists():
        print(f"[error] not a git repo: {core_root}", file=sys.stderr)
        sys.exit(2)

    print(f"KB root:   {kb_root}")
    print(f"kb-core:   {core_root}")

    if not args.no_pull:
        print("\n[1/2] git pull kb-core...")
        try:
            sha = git_pull(core_root)
            print(f"      kb-core HEAD = {sha}")
        except subprocess.CalledProcessError as e:
            print(f"[error] git pull failed: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        print("\n[1/2] skip git pull (--no-pull)")

    print(f"\n[2/2] mirror {len(SYNC_PATHS)} paths...")
    total_added = total_updated = total_deleted = 0
    for rel in SYNC_PATHS:
        a, u, d = mirror_path(rel, core_root, kb_root, args.dry_run)
        total_added += a
        total_updated += u
        total_deleted += d

    print(f"\nDone. added={total_added}  updated={total_updated}  deleted={total_deleted}"
          + ("  (dry-run, no changes applied)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
