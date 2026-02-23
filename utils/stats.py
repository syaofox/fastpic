"""统计工具：优先从数据库统计，避免百万级文件系统遍历"""
import asyncio
import os
from pathlib import Path

_STATS_BATCH_SIZE = 20000


async def stats_folder_count_from_db(session, batch_size: int = _STATS_BATCH_SIZE) -> int:
    """从数据库分批统计文件夹数量（distinct path prefixes），支持百万级"""
    from sqlmodel import select
    from models import Image

    folders: set[str] = set()
    last_id = 0
    while True:
        stmt = (
            select(Image.id, Image.relative_path)
            .where(Image.id > last_id)
            .order_by(Image.id)
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            break
        for rid, rp in rows:
            last_id = rid or last_id
            parts = rp.split("/")
            for i in range(1, len(parts)):
                folders.add("/".join(parts[:i]))
        await asyncio.sleep(0)
    return len(folders)


def stats_cache_only(cache_dir: Path) -> tuple[int, int]:
    """仅统计 cache 目录（webp 数量与总大小），用于与 DB 校验。百万级时仍较慢。"""
    cache_count = 0
    cache_size = 0
    if cache_dir.exists():
        for f in cache_dir.rglob("*.webp"):
            cache_count += 1
            cache_size += f.stat().st_size
    return cache_count, cache_size


def stats_folder_and_cache(photos_dir: Path, cache_dir: Path) -> tuple[int, int, int]:
    """同步统计文件夹数量和缓存信息。百万级时 os.walk/rglob 较慢，建议使用 stats_from_db + stats_cache_only。"""
    folder_count = 0
    for dirpath, dirnames, _ in os.walk(photos_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        folder_count += len(dirnames)
    cache_count, cache_size = stats_cache_only(cache_dir)
    return folder_count, cache_count, cache_size
