"""文件夹树：提取、计数、嵌套结构、缓存、子文件夹"""
import asyncio
import time
from pathlib import Path

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


async def get_subfolders(
    session,
    photos_dir: Path,
    path: str,
    path_filter,
    sort_by: str = "filename",
    sort_order: str = "asc",
) -> list[dict]:
    """获取当前路径下的直接子文件夹，每个子文件夹取 4 张代表图。"""
    path_depth = len(path.split("/")) if path else 0
    path_prefix = path + "/" if path else ""
    batch_stmt = select(Image.relative_path, Image.modified_at, Image.file_size)
    if path:
        escaped = escape_like(path_prefix)
        batch_stmt = batch_stmt.where(
            Image.relative_path.like(f"{escaped}%", escape="\\"),
            Image.relative_path.like(f"{escaped}%/%", escape="\\"),
        )
    else:
        batch_stmt = batch_stmt.where(Image.relative_path.like("%/%"))
    result = await session.execute(batch_stmt)
    rows = result.fetchall()
    subfolder_data: dict[str, list[tuple[str, float, int]]] = {}
    for rel, mod, size in rows:
        parts = rel.split("/")
        if len(parts) <= path_depth:
            continue
        sub_name = parts[path_depth]
        if sub_name not in subfolder_data:
            subfolder_data[sub_name] = []
        subfolder_data[sub_name].append((rel, mod or 0.0, size or 0))
    fs_dir = photos_dir / path if path else photos_dir
    if fs_dir.is_dir():
        children = await asyncio.to_thread(
            lambda: [c for c in fs_dir.iterdir() if c.is_dir() and not c.name.startswith(".")]
        )
        for child in children:
            if child.name not in subfolder_data:
                subfolder_data[child.name] = []
    subfolders: list[dict] = []
    for name, items in subfolder_data.items():
        full_path = f"{path}/{name}" if path else name
        if items:
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            count = len(items)
            max_mod = items_sorted[0][1] if items_sorted else 0.0
            max_size = max((x[2] for x in items), default=0)
            thumbnails = [x[0] for x in items_sorted[:4]]
        else:
            count = 0
            max_mod = 0.0
            max_size = 0
            thumbnails = []
        subfolders.append({
            "name": name,
            "full_path": full_path,
            "thumbnails": thumbnails,
            "image_count": count,
            "_sort_key_filename": natural_sort_key(name),
            "_sort_key_folder_filename": natural_sort_key(full_path),
            "_sort_key_modified_at": max_mod,
            "_sort_key_file_size": max_size,
        })
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
