"""path count 持久化缓存：减轻百万级 COUNT 查询。含内存 + DB 双层缓存供 gallery 使用。"""

import asyncio
import time
from sqlalchemy import text

from app.models import sync_engine

_PATH_COUNT_DB_TTL = 300.0  # 5 分钟
_COUNT_CACHE_TTL = 300.0  # 内存 TTL（与 DB 一致）
_COUNT_CACHE_MAX_SIZE = 1000  # 内存缓存上限
_count_cache: dict[tuple[str, str], tuple[int, float]] = {}


def _prune_count_cache() -> None:
    """移除过期条目，超限时移除最久未访问的"""
    now = time.monotonic()
    expired = [k for k, (_, ts) in _count_cache.items() if now - ts > _COUNT_CACHE_TTL]
    for k in expired:
        del _count_cache[k]
    while len(_count_cache) > _COUNT_CACHE_MAX_SIZE:
        oldest_key = min(_count_cache.keys(), key=lambda k: _count_cache[k][1])
        del _count_cache[oldest_key]


def get_cached_count(path: str, mode: str) -> int | None:
    """查内存缓存，过期返回 None"""
    key = (path or "", mode)
    entry = _count_cache.get(key)
    if entry is None:
        return None
    total, ts = entry
    if time.monotonic() - ts > _COUNT_CACHE_TTL:
        del _count_cache[key]
        return None
    return total


def set_cached_count(path: str, mode: str, total: int) -> None:
    """写入内存缓存"""
    key = (path or "", mode)
    _count_cache[key] = (total, time.monotonic())
    if len(_count_cache) >= _COUNT_CACHE_MAX_SIZE:
        _prune_count_cache()


def _get_path_count_from_db_sync(path: str, mode: str) -> int | None:
    """同步 DB 读取，供 asyncio.to_thread 调用，避免阻塞事件循环"""
    path_key = path or ""
    with sync_engine.connect() as conn:
        r = conn.execute(
            text(
                "SELECT total, updated_at FROM path_count_cache "
                "WHERE path = :p AND mode = :m"
            ),
            {"p": path_key, "m": mode},
        )
        row = r.fetchone()
    if row is None:
        return None
    total, updated_at = row
    if time.time() - updated_at > _PATH_COUNT_DB_TTL:
        return None
    return total


async def get_path_count_from_db(path: str, mode: str) -> int | None:
    """从 DB 读取 path count 缓存，过期返回 None"""
    return await asyncio.to_thread(_get_path_count_from_db_sync, path, mode)


def _set_path_count_to_db_sync(path: str, mode: str, total: int) -> None:
    """同步 DB 写入，供 asyncio.to_thread 调用，避免阻塞事件循环"""
    path_key = path or ""
    now = time.time()
    with sync_engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO path_count_cache (path, mode, total, updated_at) "
                "VALUES (:p, :m, :t, :ts) "
                "ON DUPLICATE KEY UPDATE total = :t, updated_at = :ts"
            ),
            {"p": path_key, "m": mode, "t": total, "ts": now},
        )
        conn.commit()


async def set_path_count_to_db(path: str, mode: str, total: int) -> None:
    """写入 path count 到 DB"""
    await asyncio.to_thread(_set_path_count_to_db_sync, path, mode, total)


def cleanup_expired_path_count_cache() -> int:
    """删除过期的 path_count_cache 记录，返回删除行数"""
    cutoff = time.time() - _PATH_COUNT_DB_TTL
    with sync_engine.connect() as conn:
        r = conn.execute(
            text("DELETE FROM path_count_cache WHERE updated_at < :cutoff"),
            {"cutoff": cutoff},
        )
        conn.commit()
        return r.rowcount


def invalidate_path_count_cache() -> None:
    """清空 path count 持久化缓存（文件夹操作后调用）"""
    with sync_engine.connect() as conn:
        conn.execute(text("DELETE FROM path_count_cache"))
        conn.commit()
