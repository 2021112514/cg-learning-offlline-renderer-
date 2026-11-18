# 从教程仓库中同步scene的脚本，由ChatGPT生成
# 使用方法
# python sync_scenes.py --dry-run    # 预览会新增或保留哪些文件
# python sync_scenes.py              # 补齐缺失文件
# python sync_scenes.py --overwrite  # 同时覆盖与教程不同的同名文件
"""Copy tutorial scenes and their assets into this project's scenes directory."""
import argparse
import filecmp
from pathlib import Path
import shutil
import subprocess
import tempfile


REPO = "https://github.com/woyaofacai/FortuneRenderer.git"
DEST = Path(__file__).resolve().parent / "scenes"


def sync(source, overwrite, dry_run):
    if not source.is_dir():
        raise ValueError(f"Scenes directory not found: {source}")
    counts = dict(new=0, replace=0, keep=0, unchanged=0)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = DEST / relative
        if target.exists():
            if target.is_file() and filecmp.cmp(path, target, shallow=False):
                action = "unchanged"
            else:
                action = "replace" if overwrite else "keep"
        else:
            action = "new"
        counts[action] += 1
        if action in ("new", "replace", "keep"):
            print(f"{action}: {relative}")
        if not dry_run and action in ("new", "replace"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    print(("Preview: " if dry_run else "") + ", ".join(f"{k}={v}" for k, v in counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="scenes directory in an existing tutorial checkout")
    parser.add_argument("--overwrite", action="store_true", help="replace differing local files")
    parser.add_argument("--dry-run", action="store_true", help="show changes without writing to the project")
    args = parser.parse_args()
    try:
        if args.source is not None:
            sync(args.source, args.overwrite, args.dry_run)
        else:
            with tempfile.TemporaryDirectory(prefix="fortune-scenes-") as temp:
                checkout = Path(temp) / "repo"
                subprocess.run(
                    ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", REPO, str(checkout)],
                    check=True,
                )
                subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", "scenes"], check=True)
                sync(checkout / "scenes", args.overwrite, args.dry_run)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Sync failed: {exc}\n")


if __name__ == "__main__":
    main()
