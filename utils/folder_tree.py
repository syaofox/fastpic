"""文件夹树：提取、计数、嵌套结构、缓存、子文件夹"""
import asyncio
import time
from pathlib import Path

from sqlmodel import select
from sqlalchemy import text

from models import FolderThumbnail, natural_sort_key

from .path_utils import escape_like, LIKE_ESCAPE

_FOLDER_TREE_MAX_DEPTH = 4  # 缓存深度限制，减少百万级时内存；更深层按需加载


def _extract_direct_child(path_prefix: str) -> str:
    """生成 SQL 表达式：从 relative_path 提取直接子文件夹名。
    path_prefix 如 '2024/01/'，relative_path 如 '2024/01/15/photo.jpg' -> '15'
    """
    if not path_prefix:
        return "SUBSTRING_INDEX(relative_path, '/', 1)"
    prefix_len = len(path_prefix)
    rest_expr = f"SUBSTRING(relative_path, {prefix_len + 1})"
    return f"SUBSTRING_INDEX({rest_expr}, '/', 1)"


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


async def _empty_list_coro() -> list:
    return []


async def _get_user_thumbnails(session, folder_paths: list[str], limit: int = 4) -> dict[str, list[str]]:
    """获取用户指定的文件夹缩略图，按 display_order 排序，每文件夹最多 limit 张。"""
    if not folder_paths:
        return {}
    stmt = (
        select(FolderThumbnail.folder_path, FolderThumbnail.image_relative_path, FolderThumbnail.display_order)
        .where(FolderThumbnail.folder_path.in_(folder_paths))
        .order_by(FolderThumbnail.folder_path, FolderThumbnail.display_order)
    )
    result = await session.execute(stmt)
    rows = result.fetchall()
    out: dict[str, list[str]] = {fp: [] for fp in folder_paths}
    for folder_path, rel_path, _ in rows:
        if folder_path in out and len(out[folder_path]) < limit:
            out[folder_path].append(rel_path)
    return out


async def _get_direct_layer_thumbnails(
    session,
    folder_path: str,
    limit: int = 4,
    exclude_paths: set[str] | None = None,
) -> list[str]:
    """查询 folder_path 下直接一层的图片（不含子目录），按 modified_at DESC 取前 limit 张。
    条件：relative_path LIKE 'folder/%' AND NOT LIKE 'folder/%/%'，可利用索引。"""
    exclude_paths = exclude_paths or set()
    escaped = escape_like(folder_path)
    like_prefix = f"{escaped}/%"
    like_prefix_sub = f"{escaped}/%/%"
    sql = text(
        "SELECT relative_path FROM images "
        f"WHERE relative_path LIKE :like_prefix ESCAPE '{LIKE_ESCAPE}' "
        f"AND relative_path NOT LIKE :like_prefix_sub ESCAPE '{LIKE_ESCAPE}' "
        "ORDER BY modified_at DESC LIMIT :lim"
    )
    result = await session.execute(
        sql,
        {"like_prefix": like_prefix, "like_prefix_sub": like_prefix_sub, "lim": limit + 20},
    )
    out: list[str] = []
    for row in result.fetchall():
        rp = row[0]
        if rp in exclude_paths:
            continue
        out.append(rp)
        if len(out) >= limit:
            break
    return out


async def get_root_subfolders_from_counts(
    folder_counts: dict[str, int],
    session,
    limit_thumbnails: int = 4,
) -> list[dict]:
    """从 folder_counts 提取根路径下的直接子文件夹，避免 path='' 时 get_subfolders 的全表扫描。
    使用 _get_direct_layer_thumbnails 获取每层缩略图（索引友好）。
    返回格式与 get_subfolders 兼容：[{name, full_path, thumbnails, image_count}, ...]，按名称自然排序。"""
    result: list[dict] = []
    for path_key, count in folder_counts.items():
        if path_key == "" or "/" in path_key:
            continue
        result.append({
            "name": path_key,
            "full_path": path_key,
            "thumbnails": [],
            "image_count": count,
        })
    result.sort(key=lambda s: natural_sort_key(s["name"]))
    folder_paths = [sub["full_path"] for sub in result]
    user_thumbs = await _get_user_thumbnails(session, folder_paths, limit=limit_thumbnails)
    thumb_tasks = []
    for sub in result:
        fp = sub["full_path"]
        ut = user_thumbs.get(fp, [])
        need = limit_thumbnails - len(ut)
        thumb_tasks.append(
            _get_direct_layer_thumbnails(session, fp, limit=need, exclude_paths=set(ut))
            if need > 0
            else _empty_list_coro()
        )
    auto_results = await asyncio.gather(*thumb_tasks)
    for sub, auto_thumbs in zip(result, auto_results):
        fp = sub["full_path"]
        ut = user_thumbs.get(fp, [])
        sub["thumbnails"] = (ut + list(auto_thumbs))[:limit_thumbnails]
    return result


_FOLDER_TREE_CACHE_TTL = 60.0
_folder_tree_cache: dict | None = None
_folder_tree_cache_lock = asyncio.Lock()


def invalidate_folder_tree_cache() -> None:
    """创建/删除文件夹后调用，使缓存失效"""
    global _folder_tree_cache
    _folder_tree_cache = None
    try:
        from utils.path_count_cache import invalidate_path_count_cache
        invalidate_path_count_cache()
    except ImportError:
        pass


def _build_folder_counts_sql(max_depth: int) -> str:
    """生成 folder_counts 聚合 SQL，max_depth 为最大路径深度（不含文件名）。"""
    parts = ["SELECT '' AS prefix FROM images"]
    for i in range(1, max_depth + 1):
        like_pattern = "%" + "/%" * i
        parts.append(
            f"SELECT SUBSTRING_INDEX(relative_path, '/', {i}) FROM images "
            f"WHERE relative_path LIKE '{like_pattern}'"
        )
    union_sql = " UNION ALL ".join(parts)
    return f"""
        SELECT prefix, COUNT(*) AS cnt FROM (
            {union_sql}
        ) t
        GROUP BY prefix
    """


_SEARCH_DIRS_MAX_DEPTH = 10  # 目录搜索支持的最大深度，超过侧边栏树状图


async def _get_folder_counts_from_sql(
    session, max_depth: int = _FOLDER_TREE_MAX_DEPTH
) -> dict[str, int]:
    """用单条 SQL 聚合查询获取各文件夹路径的图片数量，替代分批加载 + Python 累加。
    max_depth: 最大路径深度，默认 4；search_dirs 等场景可传更大值（如 10）以支持更深目录。"""
    sql = text(_build_folder_counts_sql(max_depth))
    result = await session.execute(sql)
    rows = result.fetchall()
    counts: dict[str, int] = {"": 0}
    for row in rows:
        prefix, cnt = row[0], row[1]
        counts[prefix] = int(cnt)
    return counts


async def get_folder_counts_for_search(session) -> dict[str, int]:
    """获取用于目录搜索的 folder_counts（max_depth=10）"""
    return await _get_folder_counts_from_sql(session, max_depth=_SEARCH_DIRS_MAX_DEPTH)


async def _get_folder_tree_from_db_batched(session, photos_dir: Path):
    """从数据库 SQL 聚合获取 folder_counts，再构建 folder_tree 和 nested_tree。"""
    folder_counts = await _get_folder_counts_from_sql(session)

    folders: set[tuple[str, ...]] = set()
    for path_key in folder_counts.keys():
        if path_key == "":
            continue
        parts = path_key.split("/")
        for i in range(1, len(parts) + 1):
            folders.add(tuple(parts[:i]))

    def _scan_dirs(base: Path, prefix: tuple[str, ...] = ()):
        if not base.is_dir() or len(prefix) >= _FOLDER_TREE_MAX_DEPTH:
            return
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                path_tuple = prefix + (child.name,)
                folders.add(path_tuple)
                _scan_dirs(child, path_tuple)

    await asyncio.to_thread(_scan_dirs, photos_dir)
    folder_tree = [list(f) for f in sorted(folders) if f]
    nested_tree = build_nested_tree(folder_tree)
    return folder_tree, nested_tree, folder_counts


async def get_folder_tree_cached(
    photos_dir: Path,
    rel_paths: list[str] | None = None,
    session=None,
) -> tuple[list[list[str]], dict, dict[str, int]]:
    """获取 folder_tree、nested_tree、folder_counts，带 60 秒缓存。
    若提供 session 则从数据库分批加载（支持百万级）；否则使用传入的 rel_paths。"""
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
        if session is not None:
            folder_tree, nested_tree, folder_counts = await _get_folder_tree_from_db_batched(
                session, photos_dir
            )
        else:
            folder_tree = await asyncio.to_thread(
                get_folder_tree, photos_dir, rel_paths or []
            )
            nested_tree = build_nested_tree(folder_tree)
            folder_counts = compute_folder_counts(rel_paths or [])
        _folder_tree_cache = {
            "ts": now,
            "folder_tree": folder_tree,
            "nested_tree": nested_tree,
            "folder_counts": folder_counts,
        }
        return folder_tree, nested_tree, folder_counts


async def get_subfolders(
    session,
    photos_dir: Path,
    path: str,
    path_filter,
    sort_by: str = "filename",
    sort_order: str = "asc",
) -> list[dict]:
    """获取当前路径下的直接子文件夹，每个子文件夹取 4 张代表图。"""
    path_prefix = path + "/" if path else ""
    sub_name_expr = _extract_direct_child(path_prefix)

    if path:
        escaped = escape_like(path_prefix)
        where_clause = (
            f"relative_path LIKE :like_prefix ESCAPE '{LIKE_ESCAPE}' "
            f"AND relative_path LIKE :like_sub ESCAPE '{LIKE_ESCAPE}'"
        )
        params = {"like_prefix": f"{escaped}%", "like_sub": f"{escaped}%/%"}
    else:
        where_clause = "relative_path LIKE '%/%'"
        params = {}

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

    fs_dir = photos_dir / path if path else photos_dir
    db_names = {r[0] for r in agg_rows}
    if fs_dir.is_dir():
        children = await asyncio.to_thread(
            lambda: [c for c in fs_dir.iterdir() if c.is_dir() and not c.name.startswith(".")]
        )
        for child in children:
            if child.name not in db_names:
                agg_rows.append((child.name, 0, 0.0, 0))

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

    folder_paths = [sub["full_path"] for sub in subfolders]
    user_thumbs = await _get_user_thumbnails(session, folder_paths, limit=4)
    thumb_tasks = []
    for sub in subfolders:
        fp = sub["full_path"]
        ut = user_thumbs.get(fp, [])
        need = 4 - len(ut)
        thumb_tasks.append(
            _get_direct_layer_thumbnails(session, fp, limit=need, exclude_paths=set(ut))
            if need > 0
            else _empty_list_coro()
        )
    auto_results = await asyncio.gather(*thumb_tasks)
    for sub, auto_thumbs in zip(subfolders, auto_results):
        fp = sub["full_path"]
        ut = user_thumbs.get(fp, [])
        sub["thumbnails"] = (ut + list(auto_thumbs))[:4]

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