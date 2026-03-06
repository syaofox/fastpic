"""哈希工具"""
import hashlib
from pathlib import Path

_CHUNK_SIZE = 65536


def compute_file_md5_by_path(path: Path) -> str | None:
    """分块计算文件 MD5，避免大文件 OOM。文件不存在或非文件返回 None。"""
    if not path.is_file():
        return None
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def compute_file_md5(photos_dir: Path, relative_path: str) -> str | None:
    """同步计算文件 MD5，文件不存在返回 None"""
    full_path = photos_dir / relative_path
    return compute_file_md5_by_path(full_path)
