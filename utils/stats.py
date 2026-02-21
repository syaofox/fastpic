"""统计工具"""
import os
from pathlib import Path


def stats_folder_and_cache(photos_dir: Path, cache_dir: Path) -> tuple[int, int, int]:
    """同步统计文件夹数量和缓存信息，返回 (folder_count, cache_count, cache_size)"""
    folder_count = 0
    for dirpath, dirnames, _ in os.walk(photos_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        folder_count += len(dirnames)
    cache_count = 0
    cache_size = 0
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.suffix == ".webp":
                cache_count += 1
                cache_size += f.stat().st_size
    return folder_count, cache_count, cache_size
