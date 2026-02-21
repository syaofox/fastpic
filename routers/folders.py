"""文件夹 API：移动、删除、合并、创建、搜索"""
import asyncio
import os
import shutil
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR
from models import Image, get_async_session, natural_sort_key
from scanner import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _cache_filename,
    _generate_thumbnail,
    _generate_video_thumbnail,
)
from schemas import (
    MoveImagesRequest,
    MoveFoldersRequest,
    DeleteFoldersRequest,
    MergeFoldersRequest,
    CreateFolderRequest,
)
from utils.path_utils import path_filter_for_prefix
from utils.unique_path import unique_path
from utils.images import delete_image_files
from utils.folder_tree import (
    get_folder_tree_cached,
    invalidate_folder_tree_cache,
    get_subfolders,
    scan_all_dirs_for_search,
)
from utils.search import search_match
from utils.hash_utils import compute_file_md5

router = APIRouter(prefix="/api", tags=["folders"])


@router.post("/move-images")
async def move_images(
    body: MoveImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定图片移动到目标文件夹"""
    if not body.ids:
        return {"moved": 0, "errors": []}
    target_path = (body.target_path or "").strip().strip("/")
    if ".." in target_path or target_path.startswith("/"):
        return {"moved": 0, "errors": ["目标路径不合法"]}
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stmt = select(Image).where(Image.id.in_(body.ids))
    result = await session.execute(stmt)
    images = list(result.scalars().all())
    moved = 0
    errors = []
    for img in images:
        src_path = PHOTOS_DIR / img.relative_path
        if not src_path.exists():
            errors.append(f"{img.filename}: 文件不存在")
            continue
        ext = Path(img.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            errors.append(f"{img.filename}: 不支持的格式")
            continue
        new_rel = f"{target_path}/{img.filename}" if target_path else img.filename
        if new_rel == img.relative_path:
            continue
        dest_path = target_dir / img.filename
        if dest_path.exists() and dest_path.resolve() != src_path.resolve():
            dest_path = unique_path(target_dir, img.filename, suffix_style="underscore")
            new_rel = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")
        try:
            shutil.move(str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{img.filename}: {e}")
            continue
        old_cache = CACHE_DIR / _cache_filename(img.relative_path)
        if old_cache.exists():
            old_cache.unlink(missing_ok=True)
        img.relative_path = new_rel
        img.filename = dest_path.name
        img.filename_natural = natural_sort_key(dest_path.name)
        img.relative_path_natural = natural_sort_key(new_rel)
        img.modified_at = os.path.getmtime(dest_path)
        img.file_size = dest_path.stat().st_size
        new_cache = CACHE_DIR / _cache_filename(new_rel)
        _generate_thumbnail(dest_path, new_cache)
        session.add(img)
        moved += 1
    await session.commit()
    return {"moved": moved, "errors": errors}


@router.post("/move-folders")
async def move_folders(
    body: MoveFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定文件夹（含子文件夹和图片）移动到目标父目录"""
    if not body.paths:
        return {"moved": 0, "errors": []}
    target_path = (body.target_path or "").strip().strip("/")
    if ".." in target_path or target_path.startswith("/"):
        return {"moved": 0, "errors": ["目标路径不合法"]}
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    errors = []
    for folder_path in body.paths:
        folder_path = folder_path.strip().strip("/")
        if not folder_path:
            continue
        if target_path == folder_path or target_path.startswith(folder_path + "/"):
            errors.append(f"{folder_path}: 不能移动到自身或子文件夹内")
            continue
        folder_name = Path(folder_path).name
        would_be_path = f"{target_path}/{folder_name}" if target_path else folder_name
        if would_be_path == folder_path:
            continue
        src_path = PHOTOS_DIR / folder_path
        if not src_path.exists() or not src_path.is_dir():
            errors.append(f"{folder_path}: 文件夹不存在")
            continue
        dest_path = unique_path(target_dir, folder_name, is_folder=True)
        new_prefix = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")
        if src_path.resolve() == dest_path.resolve():
            continue
        try:
            shutil.move(str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{folder_path}: {e}")
            continue
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        stmt = select(Image).where(pf)
        result = await session.execute(stmt)
        images = list(result.scalars().all())
        for img in images:
            suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path):]
            new_rel = new_prefix + suffix
            old_cache = CACHE_DIR / _cache_filename(img.relative_path)
            if old_cache.exists():
                old_cache.unlink(missing_ok=True)
            img.relative_path = new_rel
            img.filename = Path(new_rel).name
            img.filename_natural = natural_sort_key(img.filename)
            img.relative_path_natural = natural_sort_key(new_rel)
            new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
            if new_full.exists() and new_full.is_file():
                img.modified_at = os.path.getmtime(new_full)
                img.file_size = os.path.getsize(new_full)
                new_cache = CACHE_DIR / _cache_filename(new_rel)
                _generate_thumbnail(new_full, new_cache)
            session.add(img)
            moved += 1
        print(f"[api] 移动文件夹: {folder_path} → {new_prefix}", flush=True)
    await session.commit()
    return {"moved": moved, "errors": errors}


@router.post("/delete-folders")
async def delete_folders(
    body: DeleteFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """删除指定文件夹路径下所有图片（数据库 + 文件系统），并删除文件夹目录"""
    if not body.paths:
        return {"deleted_images": 0, "deleted_folders": 0}
    total_images = 0
    total_folders = 0
    for folder_path in body.paths:
        folder_path = folder_path.strip().strip("/")
        if not folder_path:
            continue
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        stmt = select(Image).where(pf)
        result = await session.execute(stmt)
        images = list(result.scalars().all())
        for img in images:
            delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
            await session.delete(img)
            total_images += 1
        folder_fs_path = PHOTOS_DIR / folder_path
        if folder_fs_path.exists() and folder_fs_path.is_dir():
            shutil.rmtree(folder_fs_path, ignore_errors=True)
            total_folders += 1
    await session.commit()
    if total_folders > 0:
        invalidate_folder_tree_cache()
    return {"deleted_images": total_images, "deleted_folders": total_folders}


@router.post("/merge-folders")
async def merge_folders(
    body: MergeFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """合并两个文件夹：通过 MD5 去重"""
    folder_a = (body.folder_a or "").strip().strip("/")
    folder_b = (body.folder_b or "").strip().strip("/")
    if not folder_a or not folder_b:
        return {"ok": False, "error": "请指定两个文件夹路径"}
    if ".." in folder_a or folder_a.startswith("/") or ".." in folder_b or folder_b.startswith("/"):
        return {"ok": False, "error": "路径不合法"}
    if folder_a == folder_b:
        return {"ok": False, "error": "不能选择相同的文件夹"}
    if folder_a.startswith(folder_b + "/") or folder_b.startswith(folder_a + "/"):
        return {"ok": False, "error": "不能合并互为父子关系的文件夹"}
    path_a = PHOTOS_DIR / folder_a
    path_b = PHOTOS_DIR / folder_b
    if not path_a.exists() or not path_a.is_dir():
        return {"ok": False, "error": f"文件夹不存在: {folder_a}"}
    if not path_b.exists() or not path_b.is_dir():
        return {"ok": False, "error": f"文件夹不存在: {folder_b}"}
    photos_dir = PHOTOS_DIR.resolve()
    media_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

    async def _get_images_under(prefix: str) -> list[Image]:
        pf = path_filter_for_prefix(Image.relative_path, prefix)
        stmt = select(Image).where(pf)
        return list((await session.execute(stmt)).scalars().all())

    images_a = await _get_images_under(folder_a)
    images_b = await _get_images_under(folder_b)
    count_a, count_b = len(images_a), len(images_b)
    if body.target == "folder_b":
        target_prefix, source_prefix = folder_b, folder_a
        source_images, target_images = images_a, images_b
        source_path, target_path = path_a, path_b
    else:
        target_prefix, source_prefix = folder_a, folder_b
        source_images, target_images = images_b, images_a
        source_path, target_path = path_b, path_a
    if body.target == "auto" and count_b > count_a:
        target_prefix, source_prefix = folder_b, folder_a
        source_images, target_images = images_a, images_b
        source_path, target_path = path_a, path_b

    def _belongs_to(rel: str, prefix: str) -> bool:
        return rel == prefix or rel.startswith(prefix + "/")

    all_images = [*[(img, "a") for img in images_a], *[(img, "b") for img in images_b]]
    preferred = "a" if count_a >= count_b else "b"
    by_hash: dict[str, list[tuple[Image, str]]] = defaultdict(list)
    for img, src in all_images:
        full = photos_dir / img.relative_path
        if not full.is_file() or full.suffix.lower() not in media_extensions:
            continue
        h = await asyncio.to_thread(compute_file_md5, photos_dir, img.relative_path)
        if h is None:
            continue
        by_hash[h].append((img, src))
    to_keep: dict[str, Image] = {}
    to_delete: set[int] = set()
    for h, items in by_hash.items():
        from_preferred = [(img, src) for img, src in items if src == preferred]
        from_other = [(img, src) for img, src in items if src != preferred]
        if from_preferred:
            keeper = min(from_preferred, key=lambda x: x[0].relative_path)[0]
            to_keep[h] = keeper
            for img, _ in from_preferred:
                if img.id != keeper.id:
                    to_delete.add(img.id)
            for img, _ in from_other:
                to_delete.add(img.id)
        else:
            keeper = min(from_other, key=lambda x: x[0].relative_path)[0]
            to_keep[h] = keeper
            for img, _ in from_other:
                if img.id != keeper.id:
                    to_delete.add(img.id)
    target_hashes: set[str] = set()
    for img, src in all_images:
        if img.id in to_delete:
            continue
        if _belongs_to(img.relative_path, target_prefix):
            h = await asyncio.to_thread(compute_file_md5, photos_dir, img.relative_path)
            if h:
                target_hashes.add(h)
    for img_id in to_delete:
        result = await session.execute(select(Image).where(Image.id == img_id))
        img = result.scalar_one_or_none()
        if img:
            delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
            await session.delete(img)
    moved = 0
    for img, src in all_images:
        if img.id in to_delete:
            continue
        if not _belongs_to(img.relative_path, source_prefix):
            continue
        h = await asyncio.to_thread(compute_file_md5, photos_dir, img.relative_path)
        if not h or h in target_hashes:
            if h in target_hashes:
                delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
                await session.delete(img)
            continue
        suffix = img.relative_path[len(source_prefix):].lstrip("/")
        new_rel = f"{target_prefix}/{suffix}" if suffix else target_prefix
        new_full = target_path / suffix if suffix else target_path
        new_full.parent.mkdir(parents=True, exist_ok=True)
        if new_full.exists():
            new_full = unique_path(new_full.parent, new_full.name, suffix_style="paren")
            new_rel = str(new_full.relative_to(PHOTOS_DIR)).replace("\\", "/")
        try:
            shutil.move(str(photos_dir / img.relative_path), str(new_full))
        except OSError as e:
            await session.rollback()
            return {"ok": False, "error": f"移动文件失败 {img.relative_path}: {e}"}
        old_cache = CACHE_DIR / _cache_filename(img.relative_path)
        if old_cache.exists():
            old_cache.unlink(missing_ok=True)
        img.relative_path = new_rel
        img.filename = Path(new_rel).name
        img.filename_natural = natural_sort_key(img.filename)
        img.relative_path_natural = natural_sort_key(new_rel)
        img.modified_at = os.path.getmtime(new_full)
        img.file_size = os.path.getsize(new_full)
        new_cache = CACHE_DIR / _cache_filename(new_rel)
        if new_full.suffix.lower() in VIDEO_EXTENSIONS:
            _generate_video_thumbnail(new_full, new_cache)
        else:
            _generate_thumbnail(new_full, new_cache)
        session.add(img)
        target_hashes.add(h)
        moved += 1
    if source_path.exists():
        for d in sorted(source_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except OSError:
                    pass
        if not any(source_path.iterdir()):
            try:
                source_path.rmdir()
            except OSError:
                pass
    await session.commit()
    invalidate_folder_tree_cache()
    print(f"[api] 合并文件夹: {folder_a} + {folder_b} -> {target_prefix}, 移动 {moved} 个文件", flush=True)
    return {"ok": True, "moved": moved, "deleted": len(to_delete), "target": target_prefix}


@router.post("/create-folder")
async def create_folder(body: CreateFolderRequest):
    """在指定路径下创建子文件夹"""
    parent = (body.path or "").strip().strip("/")
    name = body.name.strip().strip("/")
    if not name:
        return {"error": "文件夹名不能为空", "ok": False}
    if ".." in name or "/" in name or "\\" in name:
        return {"error": "文件夹名不合法", "ok": False}
    folder_path = PHOTOS_DIR / parent / name if parent else PHOTOS_DIR / name
    if folder_path.exists():
        return {"error": "文件夹已存在", "ok": False}
    folder_path.mkdir(parents=True, exist_ok=True)
    rel = f"{parent}/{name}" if parent else name
    invalidate_folder_tree_cache()
    print(f"[api] 创建文件夹: {rel}", flush=True)
    return {"ok": True, "path": rel}


@router.get("/subfolders")
async def get_subfolders_api(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """获取指定路径下的直接子文件夹"""
    path = (path or "").strip().strip("/")
    if ".." in path or path.startswith("/"):
        return {"subfolders": []}
    pf = path_filter_for_prefix(Image.relative_path, path) if path else None
    subfolders = await get_subfolders(session, PHOTOS_DIR, path, pf)
    return {
        "subfolders": [
            {"name": s["name"], "full_path": s["full_path"], "image_count": s["image_count"]}
            for s in subfolders
        ]
    }


@router.get("/search-dirs")
async def search_dirs(
    q: str = "",
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """全局目录搜索"""
    q = (q or "").strip()
    if not q:
        return {"dirs": []}
    result = await session.execute(select(Image.relative_path))
    all_paths = [r[0] for r in result.fetchall()]
    dir_counts: dict[str, int] = {}
    for rp in all_paths:
        parts = rp.rsplit("/", 1)
        dir_path = parts[0] if len(parts) == 2 else ""
        dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1
    full_dir_counts: dict[str, int] = {}
    for dir_path, count in dir_counts.items():
        if not dir_path:
            continue
        parts = dir_path.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            full_dir_counts[prefix] = full_dir_counts.get(prefix, 0) + count
    await asyncio.to_thread(scan_all_dirs_for_search, PHOTOS_DIR, "", full_dir_counts)
    matched = []
    for dir_path, count in sorted(full_dir_counts.items()):
        if search_match(q, dir_path):
            matched.append({"path": dir_path, "image_count": count})
            if len(matched) >= limit:
                break
    return {"dirs": matched}


@router.get("/list-subdirs")
async def list_subdirs(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """列出指定路径下的直接子文件夹"""
    path = (path or "").strip().strip("/")
    path_parts = path.split("/") if path else []
    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree, _, folder_counts = await get_folder_tree_cached(PHOTOS_DIR, rel_paths)
    depth = len(path_parts) + 1
    subdirs: list[dict] = []
    seen: set[str] = set()
    for parts in folder_tree:
        if len(parts) != depth:
            continue
        if path_parts and parts[: len(path_parts)] != path_parts:
            continue
        sub_path = "/".join(parts)
        if sub_path in seen:
            continue
        seen.add(sub_path)
        count = folder_counts.get(sub_path, 0)
        subdirs.append({"path": sub_path, "name": parts[-1], "image_count": count})
    subdirs.sort(key=lambda x: x["name"])
    return {"dirs": subdirs, "parent": path}
