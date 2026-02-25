"""统计工具：优先从数据库统计，避免百万级文件系统遍历"""
async def stats_folder_count_from_db(session) -> int:
    """从数据库 SQL 聚合统计文件夹数量（distinct path prefixes），支持百万级"""
    from utils.folder_tree import _get_folder_counts_from_sql

    counts = await _get_folder_counts_from_sql(session)
    return len([k for k in counts if k != ""])
