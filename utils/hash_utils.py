"""哈希工具"""
import hashlib
from pathlib import Path


def compute_file_md5(photos_dir: Path, relative_path: str) -> str | None:
    """同步计算文件 MD5，文件不存在返回 None"""
    full_path = photos_dir / relative_path
    if not full_path.is_file():
        return None
    try:
        return hashlib.md5(full_path.read_bytes()).hexdigest()
    except OSError:
        return None
