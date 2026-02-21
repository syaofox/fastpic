"""图片相关工具"""
from pathlib import Path

from scanner import _cache_filename


def delete_image_files(relative_path: str, photos_dir: Path, cache_dir: Path) -> None:
    """删除图片的原始文件和缓存文件"""
    photo_path = photos_dir / relative_path
    if photo_path.exists():
        photo_path.unlink(missing_ok=True)
    cache_name = _cache_filename(relative_path)
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cache_path.unlink(missing_ok=True)
