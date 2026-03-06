"""唯一路径生成：在目标目录下生成不冲突的文件/文件夹路径"""
from pathlib import Path


def unique_path(
    target_dir: Path,
    name: str,
    is_folder: bool = False,
    suffix_style: str = "paren",
) -> Path:
    """生成不冲突的路径。

    - is_folder=True: 按文件夹规则（无扩展名）
    - suffix_style="paren": 冲突时追加 " (1)", " (2)" 等
    - suffix_style="underscore": 冲突时追加 "_1", "_2" 等（仅文件，在 stem 和 ext 之间）
    """
    dest = target_dir / name
    if not dest.exists():
        return dest

    if is_folder:
        stem = name
        ext = ""
    else:
        stem = dest.stem
        ext = dest.suffix

    counter = 1
    while dest.exists():
        if suffix_style == "underscore" and not is_folder:
            dest = target_dir / f"{stem}_{counter}{ext}"
        else:
            dest = target_dir / f"{stem} ({counter}){ext}" if ext else target_dir / f"{stem} ({counter})"
        counter += 1
    return dest
