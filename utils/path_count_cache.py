"""path count 持久化缓存：减轻百万级 COUNT 查询"""
from sqlalchemy import text

from models import sync_engine


def invalidate_path_count_cache() -> None:
    """清空 path count 持久化缓存（文件夹操作后调用）"""
    with sync_engine.connect() as conn:
        conn.execute(text("DELETE FROM path_count_cache"))
        conn.commit()
