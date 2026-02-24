"""path count 持久化缓存：减轻百万级 COUNT 查询"""
import time
from sqlalchemy import text

from models import sync_engine

_PATH_COUNT_DB_TTL = 300.0  # 5 分钟，与 main.py 一致


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
