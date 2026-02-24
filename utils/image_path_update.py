"""图片路径更新与缩略图重建，供 move/rename/merge 等复用"""
import asyncio
import os
from pathlib import Path

from models import Image, natural_sort_key
from scanner import generate_thumbnail_for_media
from utils.images import cache_filename


async def update_image_path_and_regenerate_thumbnail(
    img: Image,
    new_rel: str,
    new_full_path: Path,
    photos_dir: Path,
    cache_dir: Path,
    video_extensions: set[str],
) -> None:
    """更新 Image 路径相关字段并重建缩略图（供 move/rename/merge 复用）。
    仅当 new_full_path 存在且为文件时更新 metadata 并生成缩略图。"""
    old_cache = cache_dir / cache_filename(img.relative_path)
    if old_cache.exists():
        old_cache.unlink(missing_ok=True)
    img.relative_path = new_rel
    img.filename = Path(new_rel).name
    img.filename_natural = natural_sort_key(img.filename)
    img.relative_path_natural = natural_sort_key(new_rel)
    if new_full_path.exists() and new_full_path.is_file():
        img.modified_at = await asyncio.to_thread(os.path.getmtime, new_full_path)
        img.file_size = await asyncio.to_thread(os.path.getsize, new_full_path)
        new_cache = cache_dir / cache_filename(new_rel)
        is_video = new_full_path.suffix.lower() in video_extensions
        await asyncio.to_thread(generate_thumbnail_for_media, new_full_path, new_cache, is_video)
