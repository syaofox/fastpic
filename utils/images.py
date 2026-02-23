"""图片相关工具"""
import hashlib
from pathlib import Path


def cache_filename(relative_path: str) -> str:
    """根据相对路径生成缩略图缓存文件名，使用分层目录 hash[:2]/hash[2:].webp 避免单目录文件过多"""
    h = hashlib.md5(relative_path.encode()).hexdigest()
    return f"{h[:2]}/{h[2:]}.webp"


def delete_image_files(relative_path: str, photos_dir: Path, cache_dir: Path) -> None:
    """删除图片的原始文件和缓存文件"""
    photo_path = photos_dir / relative_path
    if photo_path.exists():
        photo_path.unlink(missing_ok=True)
    cache_name = cache_filename(relative_path)
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cache_path.unlink(missing_ok=True)
