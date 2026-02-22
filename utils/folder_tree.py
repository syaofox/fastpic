"""文件夹树：提取、计数、嵌套结构、缓存、子文件夹"""
import asyncio
import time
from pathlib import Path

from sqlalchemy import text
from sqlmodel import select

from models import Image, natural_sort_key

from .path_utils import escape_like


def get_folder_tree(photos_dir: Path, rel_paths: list[str]) -> list[list[str]]:
    """从 relative_path 列表 + 文件系统提取文件夹树，返回 [['2024'], ['2024','01'], ...]"""
    folders: set[tuple[str, ...]] = set()
    folders.add(())
    for rp in rel_paths:
        parts = rp.split("/")
        if len(parts) > 1:
            for i in range(1, len(parts)):
                folders.add(tuple(parts[:i]))

    def _scan_dirs(base: Path, prefix: tuple[str, ...] = ()):
        if not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                path_tuple = prefix + (child.name,)
                folders.add(path_tuple)
                _scan_dirs(child, path_tuple)

    _scan_dirs(photos_dir)
    return [list(f) for f in sorted(folders) if f]


def compute_folder_counts(rel_paths: list[str]) -> dict[str, int]:
    """从 relative_path 列表计算每个文件夹下的图片总数（含子目录）。"""
    counts: dict[str, int] = {"": len(rel_paths)}
    for rp in rel_paths:
        parts = rp.split("/")
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def build_nested_tree(flat_folders: list[list[str]]) -> dict:
    """将扁平文件夹列表转为嵌套树结构。"""
    root: dict = {}
    for parts in flat_folders:
        d = root
        for part in parts:
            if part not in d:
                d[part] = {}
            d = d[part]
    return root


_FOLDER_TREE_CACHE_TTL = 60.0
_folder_tree_cache: dict | None = None
_folder_tree_cache_lock = asyncio.Lock()


def invalidate_folder_tree_cache() -> None:
    """创建/删除文件夹后调用，使缓存失效"""
    global _folder_tree_cache
    _folder_tree_cache = None


async def get_folder_tree_cached(
    photos_dir: Path, rel_paths: list[str]
) -> tuple[list[list[str]], dict, dict[str, int]]:
    """获取 folder_tree、nested_tree、folder_counts，带 60 秒缓存"""
    global _folder_tree_cache
    async with _folder_tree_cache_lock:
        now = time.monotonic()
        if _folder_tree_cache is not None:
            ts = _folder_tree_cache.get("ts", 0)
            if now - ts < _FOLDER_TREE_CACHE_TTL:
                return (
                    _folder_tree_cache["folder_tree"],
                    _folder_tree_cache["nested_tree"],
                    _folder_tree_cache["folder_counts"],
                )
        folder_tree = await asyncio.to_thread(get_folder_tree, photos_dir, rel_paths)
        nested_tree = build_nested_tree(folder_tree)
        folder_counts = compute_folder_counts(rel_paths)
        _folder_tree_cache = {
            "ts": now,
            "folder_tree": folder_tree,
            "nested_tree": nested_tree,
            "folder_counts": folder_counts,
        }
        return folder_tree, nested_tree, folder_counts


def _extract_direct_child_sqlite(path_prefix: str) -> str:
    """生成 SQLite 表达式：从 relative_path 提取直接子文件夹名。
    path_prefix 如 '2024/01/'，relative_path 如 '2024/01/15/photo.jpg' -> '15'
    """
    if not path_prefix:
        # 根路径：取第一段，如 '2024/01/photo.jpg' -> '2024'
        return "SUBSTR(relative_path, 1, INSTR(relative_path || '/', '/') - 1)"
    prefix_len = len(path_prefix)
    # rest = SUBSTR(relative_path, prefix_len + 1) 即 path_prefix 之后的部分
    # sub_name = rest 的第一段（到下一个 '/' 或结尾）
    rest_expr = f"SUBSTR(relative_path, {prefix_len + 1})"
    return f"SUBSTR({rest_expr}, 1, INSTR({rest_expr} || '/', '/') - 1)"


async def get_subfolders(
    session,
    photos_dir: Path,
    path: str,
    path_filter,
    sort_by: str = "filename",
    sort_order: str = "asc",
) -> list[dict]:
    """获取当前路径下的直接子文件夹，每个子文件夹取 4 张代表图。
    使用 SQL 聚合替代全量拉取，避免在大量图片时卡顿。
    """
    path_prefix = path + "/" if path else ""
    sub_name_expr = _extract_direct_child_sqlite(path_prefix)

    if path:
        escaped = escape_like(path_prefix)
        where_clause = (
            f"relative_path LIKE :like_prefix ESCAPE '\\' "
            f"AND relative_path LIKE :like_sub ESCAPE '\\'"
        )
        params = {"like_prefix": f"{escaped}%", "like_sub": f"{escaped}%/%"}
    else:
        where_clause = "relative_path LIKE '%/%'"
        params = {}

    # 1. SQL 聚合：直接子文件夹的 count, max(modified_at), max(file_size)
    agg_sql = f"""
        SELECT
            {sub_name_expr} AS sub_name,
            COUNT(*) AS cnt,
            MAX(modified_at) AS max_mod,
            MAX(file_size) AS max_sz
        FROM images
        WHERE {where_clause}
        GROUP BY sub_name
    """
    agg_result = await session.execute(text(agg_sql), params)
    agg_rows = agg_result.fetchall()

    # 2. 文件系统：补充数据库中无图片的空文件夹
    fs_dir = photos_dir / path if path else photos_dir
    db_names = {r[0] for r in agg_rows}
    if fs_dir.is_dir():
        children = await asyncio.to_thread(
            lambda: [c for c in fs_dir.iterdir() if c.is_dir() and not c.name.startswith(".")]
        )
        for child in children:
            if child.name not in db_names:
                agg_rows.append((child.name, 0, 0.0, 0))

    # 3. 构建 subfolders 列表
    subfolders: list[dict] = []
    for row in agg_rows:
        name, count, max_mod, max_sz = row
        full_path = f"{path}/{name}" if path else name
        subfolders.append({
            "name": name,
            "full_path": full_path,
            "thumbnails": [],
            "image_count": count or 0,
            "_sort_key_filename": natural_sort_key(name),
            "_sort_key_folder_filename": natural_sort_key(full_path),
            "_sort_key_modified_at": float(max_mod or 0.0),
            "_sort_key_file_size": int(max_sz or 0),
        })

    # 4. 单次 SQL 查询获取所有子文件夹的缩略图（ROW_NUMBER 每文件夹取 4 张）
    thumb_by_name: dict[str, list[str]] = {s["name"]: [] for s in subfolders}
    if thumb_by_name:
        thumb_sql = f"""
            WITH base AS (
                SELECT relative_path, {sub_name_expr} AS sub_name,
                    ROW_NUMBER() OVER (PARTITION BY {sub_name_expr} ORDER BY modified_at DESC) AS rn
                FROM images
                WHERE {where_clause}
            )
            SELECT relative_path, sub_name FROM base WHERE rn <= 4
        """
        thumb_result = await session.execute(text(thumb_sql), params)
        for rel_path, sub_name in thumb_result.fetchall():
            if sub_name in thumb_by_name and len(thumb_by_name[sub_name]) < 4:
                thumb_by_name[sub_name].append(rel_path)
        for sub in subfolders:
            sub["thumbnails"] = thumb_by_name.get(sub["name"], [])

    # 5. 排序
    sort_col_map = {
        "filename": "_sort_key_filename",
        "folder_filename": "_sort_key_folder_filename",
        "modified_at": "_sort_key_modified_at",
        "file_size": "_sort_key_file_size",
    }
    key = sort_col_map.get(sort_by, "_sort_key_filename")
    reverse = sort_order == "desc"
    subfolders.sort(key=lambda s: s[key], reverse=reverse)
    return subfolders


def scan_all_dirs_for_search(base: Path, prefix: str, dir_counts: dict[str, int]) -> None:
    """递归扫描目录，将空文件夹加入 dir_counts（用于 search_dirs）"""
    if not base.is_dir():
        return
    for child in sorted(base.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            child_path = f"{prefix}/{child.name}" if prefix else child.name
            if child_path not in dir_counts:
                dir_counts[child_path] = 0
            scan_all_dirs_for_search(child, child_path, dir_counts)
