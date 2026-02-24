"""统计工具：优先从数据库统计，避免百万级文件系统遍历"""
import os
import time
from pathlib import Path

_STATS_CACHE_TTL = 300.0  # 5 分钟
_stats_cache: tuple[int, int, float] | None = None  # (count, size, timestamp)


def invalidate_stats_cache() -> None:
    """使 stats_cache_only 的缓存失效（scan/cleanup 后调用）"""
    global _stats_cache
    _stats_cache = None


async def stats_folder_count_from_db(session) -> int:
    """从数据库 SQL 聚合统计文件夹数量（distinct path prefixes），支持百万级"""
    from utils.folder_tree import _get_folder_counts_from_sql

    counts = await _get_folder_counts_from_sql(session)
    return len([k for k in counts if k != ""])


def stats_cache_only(cache_dir: Path) -> tuple[int, int]:
    """仅统计 cache 目录（webp 数量与总大小），用于与 DB 校验。百万级时仍较慢。"""
    global _stats_cache
    now = time.monotonic()
    if _stats_cache is not None and now - _stats_cache[2] < _STATS_CACHE_TTL:
        return _stats_cache[0], _stats_cache[1]
    cache_count = 0
    cache_size = 0
    if cache_dir.exists():
        for f in cache_dir.rglob("*.webp"):
            cache_count += 1
            cache_size += f.stat().st_size
    _stats_cache = (cache_count, cache_size, now)
    return cache_count, cache_size


def stats_folder_and_cache(photos_dir: Path, cache_dir: Path) -> tuple[int, int, int]:
    """同步统计文件夹数量和缓存信息。百万级时 os.walk/rglob 较慢，建议使用 stats_from_db + stats_cache_only。"""
    folder_count = 0
    for dirpath, dirnames, _ in os.walk(photos_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        folder_count += len(dirnames)
    cache_count, cache_size = stats_cache_only(cache_dir)
    return folder_count, cache_count, cache_size
