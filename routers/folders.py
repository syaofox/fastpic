"""文件夹 API：移动、删除、合并、创建、搜索"""
import asyncio
import os
import shutil
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR
from models import Image, FolderThumbnail, get_async_session, natural_sort_key
from scanner import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _generate_thumbnail,
    _generate_video_thumbnail,
)
from utils.images import cache_filename
from schemas import (
    MoveImagesRequest,
    MoveFoldersRequest,
    DeleteFoldersRequest,
    MergeFoldersRequest,
    CreateFolderRequest,
    RenameFolderRequest,
    RenameImageRequest,
    BatchRenameInfoRequest,
    BatchRenameRequest,
    AddFolderThumbnailRequest,
)
from utils.path_utils import normalize_path, path_filter_for_prefix
from utils.unique_path import unique_path
from utils.images import delete_image_files
from utils.folder_tree import (
    get_folder_tree_cached,
    get_folder_counts_for_search,
    invalidate_folder_tree_cache,
    get_subfolders,
    scan_all_dirs_for_search,
)

_FOLDER_OP_BATCH_SIZE = 1000  # 文件夹操作分批大小，支持大文件夹
_IN_CLAUSE_BATCH_SIZE = 1000  # IN 子句分批大小，避免 max_allowed_packet
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
    target_path = normalize_path(body.target_path, allow_empty=True)
    if target_path is None:
        return {"moved": 0, "errors": ["目标路径不合法"]}
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    images: list[Image] = []
    for i in range(0, len(body.ids), _IN_CLAUSE_BATCH_SIZE):
        batch_ids = body.ids[i : i + _IN_CLAUSE_BATCH_SIZE]
        stmt = select(Image).where(Image.id.in_(batch_ids))
        result = await session.execute(stmt)
        images.extend(result.scalars().all())
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
            await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{img.filename}: {e}")
            continue
        old_cache = CACHE_DIR / cache_filename(img.relative_path)
        if old_cache.exists():
            old_cache.unlink(missing_ok=True)
        img.relative_path = new_rel
        img.filename = dest_path.name
        img.filename_natural = natural_sort_key(dest_path.name)
        img.relative_path_natural = natural_sort_key(new_rel)
        img.modified_at = await asyncio.to_thread(os.path.getmtime, dest_path)
        img.file_size = await asyncio.to_thread(os.path.getsize, dest_path)
        new_cache = CACHE_DIR / cache_filename(new_rel)
        await asyncio.to_thread(_generate_thumbnail, dest_path, new_cache)
        session.add(img)
        moved += 1
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        errors.append("路径冲突（可能与已有文件重复），请重试")
    return {"moved": moved, "errors": errors}


@router.post("/move-folders")
async def move_folders(
    body: MoveFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定文件夹（含子文件夹和图片）移动到目标父目录"""
    if not body.paths:
        return {"moved": 0, "errors": []}
    target_path = normalize_path(body.target_path, allow_empty=True)
    if target_path is None:
        return {"moved": 0, "errors": ["目标路径不合法"]}
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    errors = []
    for folder_path in body.paths:
        folder_path = normalize_path(folder_path, allow_empty=False)
        if folder_path is None:
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
            await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{folder_path}: {e}")
            continue
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        last_id = 0
        while True:
            stmt = (
                select(Image)
                .where(pf)
                .where(Image.id > last_id)
                .order_by(Image.id)
                .limit(_FOLDER_OP_BATCH_SIZE)
            )
            result = await session.execute(stmt)
            images = list(result.scalars().all())
            if not images:
                break
            for img in images:
                suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path):]
                new_rel = new_prefix + suffix
                old_cache = CACHE_DIR / cache_filename(img.relative_path)
                if old_cache.exists():
                    old_cache.unlink(missing_ok=True)
                img.relative_path = new_rel
                img.filename = Path(new_rel).name
                img.filename_natural = natural_sort_key(img.filename)
                img.relative_path_natural = natural_sort_key(new_rel)
                new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
                if new_full.exists() and new_full.is_file():
                    img.modified_at = await asyncio.to_thread(os.path.getmtime, new_full)
                    img.file_size = await asyncio.to_thread(os.path.getsize, new_full)
                    new_cache = CACHE_DIR / cache_filename(new_rel)
                    await asyncio.to_thread(_generate_thumbnail, new_full, new_cache)
                session.add(img)
                moved += 1
                last_id = img.id or last_id
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                errors.append(f"{folder_path}: 路径冲突，请重试")
                break
            await asyncio.sleep(0)
        print(f"[api] 移动文件夹: {folder_path} → {new_prefix}", flush=True)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        if not any("路径冲突" in e for e in errors):
            errors.append("路径冲突，请重试")
    return {"moved": moved, "errors": errors}


@router.post("/rename-folder")
async def rename_folder(
    body: RenameFolderRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """重命名文件夹（移动到同一父目录下的新名称）"""
    folder_path = normalize_path(body.path, allow_empty=False)
    if folder_path is None:
        return {"ok": False, "error": "路径不合法"}
    new_name = (body.new_name or "").strip().strip("/")
    if not new_name:
        return {"ok": False, "error": "新名称不能为空"}
    if ".." in new_name or "/" in new_name or "\\" in new_name:
        return {"ok": False, "error": "新名称不合法"}

    parts = folder_path.rsplit("/", 1)
    parent = parts[0] if len(parts) == 2 else ""
    target_dir = PHOTOS_DIR / parent if parent else PHOTOS_DIR
    new_prefix = f"{parent}/{new_name}" if parent else new_name

    if new_prefix == folder_path and Path(folder_path).name == new_name:
        return {"ok": True, "path": new_prefix}

    src_path = PHOTOS_DIR / folder_path
    if not src_path.exists() or not src_path.is_dir():
        return {"ok": False, "error": "文件夹不存在"}

    dest_path = target_dir / new_name
    if dest_path.exists() and dest_path.resolve() != src_path.resolve():
        return {"ok": False, "error": "目标文件夹已存在"}

    try:
        await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
    except OSError as e:
        return {"ok": False, "error": f"重命名失败: {e}"}

    try:
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        last_id = 0
        while True:
            stmt = (
                select(Image)
                .where(pf)
                .where(Image.id > last_id)
                .order_by(Image.id)
                .limit(_FOLDER_OP_BATCH_SIZE)
            )
            result = await session.execute(stmt)
            images = list(result.scalars().all())
            if not images:
                break
            for img in images:
                suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path):]
                new_rel = new_prefix + suffix
                old_cache = CACHE_DIR / cache_filename(img.relative_path)
                if old_cache.exists():
                    old_cache.unlink(missing_ok=True)
                img.relative_path = new_rel
                img.filename = Path(new_rel).name
                img.filename_natural = natural_sort_key(img.filename)
                img.relative_path_natural = natural_sort_key(new_rel)
                new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
                if new_full.exists() and new_full.is_file():
                    img.modified_at = await asyncio.to_thread(os.path.getmtime, new_full)
                    img.file_size = await asyncio.to_thread(os.path.getsize, new_full)
                    new_cache = CACHE_DIR / cache_filename(new_rel)
                    if new_full.suffix.lower() in VIDEO_EXTENSIONS:
                        await asyncio.to_thread(_generate_video_thumbnail, new_full, new_cache)
                    else:
                        await asyncio.to_thread(_generate_thumbnail, new_full, new_cache)
                session.add(img)
                last_id = img.id or last_id
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return {"ok": False, "error": "路径冲突，请重试"}
            await asyncio.sleep(0)

        invalidate_folder_tree_cache()
        print(f"[api] 重命名文件夹: {folder_path} → {new_prefix}", flush=True)
        return {"ok": True, "path": new_prefix}
    except Exception as e:
        await session.rollback()
        return {"ok": False, "error": f"更新数据库失败: {e}"}


def _invalid_filename(name: str) -> bool:
    """检查文件名是否包含非法字符"""
    if not name or ".." in name:
        return True
    for c in "/\\:*?\"<>|":
        if c in name:
            return True
    return False


@router.post("/rename-image")
async def rename_image(
    body: RenameImageRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """重命名单张图片/视频（仅修改文件名，保持在同一目录）"""
    new_filename = (body.new_filename or "").strip()
    if not new_filename:
        return {"ok": False, "error": "新文件名不能为空"}
    if _invalid_filename(new_filename):
        return {"ok": False, "error": "文件名包含非法字符"}

    media_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    if Path(new_filename).suffix.lower() not in media_extensions:
        return {"ok": False, "error": "不支持的文件格式"}

    result = await session.execute(select(Image).where(Image.id == body.id))
    img = result.scalar_one_or_none()
    if not img:
        return {"ok": False, "error": "图片不存在"}

    src_path = PHOTOS_DIR / img.relative_path
    if not src_path.exists() or not src_path.is_file():
        return {"ok": False, "error": "文件不存在"}

    parent_dir = src_path.parent
    dest_path = parent_dir / new_filename
    if dest_path.exists() and dest_path.resolve() != src_path.resolve():
        dest_path = unique_path(parent_dir, new_filename, suffix_style="underscore")
        new_filename = dest_path.name

    new_rel = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")
    if new_rel == img.relative_path:
        return {"ok": True, "path": new_rel}

    try:
        await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
    except OSError as e:
        return {"ok": False, "error": f"重命名失败: {e}"}

    old_cache = CACHE_DIR / cache_filename(img.relative_path)
    if old_cache.exists():
        old_cache.unlink(missing_ok=True)
    img.relative_path = new_rel
    img.filename = dest_path.name
    img.filename_natural = natural_sort_key(dest_path.name)
    img.relative_path_natural = natural_sort_key(new_rel)
    img.modified_at = await asyncio.to_thread(os.path.getmtime, dest_path)
    img.file_size = await asyncio.to_thread(os.path.getsize, dest_path)
    new_cache = CACHE_DIR / cache_filename(new_rel)
    if dest_path.suffix.lower() in VIDEO_EXTENSIONS:
        await asyncio.to_thread(_generate_video_thumbnail, dest_path, new_cache)
    else:
        await asyncio.to_thread(_generate_thumbnail, dest_path, new_cache)
    session.add(img)
    try:
        await session.commit()
        print(f"[api] 重命名图片: {img.relative_path} → {new_rel}", flush=True)
        return {"ok": True, "path": new_rel}
    except IntegrityError:
        await session.rollback()
        return {"ok": False, "error": "路径冲突（可能与已有文件重复），请重试"}


@router.post("/batch-rename-info")
async def batch_rename_info(
    body: BatchRenameInfoRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """获取批量重命名所需的项目信息（当前名称等）"""
    images: list[dict] = []
    folders: list[dict] = []

    if body.image_ids:
        for i in range(0, len(body.image_ids), _IN_CLAUSE_BATCH_SIZE):
            batch_ids = body.image_ids[i : i + _IN_CLAUSE_BATCH_SIZE]
            stmt = select(Image).where(Image.id.in_(batch_ids))
            result = await session.execute(stmt)
            for img in result.scalars().all():
                images.append({
                    "id": img.id,
                    "filename": img.filename,
                    "relative_path": img.relative_path,
                    "modified_at": img.modified_at,
                })

    for folder_path in body.folder_paths or []:
        folder_path = normalize_path(folder_path, allow_empty=False)
        if folder_path is None:
            continue
        src_path = PHOTOS_DIR / folder_path
        if not src_path.exists() or not src_path.is_dir():
            continue
        name = Path(folder_path).name
        try:
            mtime = await asyncio.to_thread(os.path.getmtime, src_path)
        except OSError:
            mtime = 0.0
        folders.append({"path": folder_path, "name": name, "modified_at": mtime})

    return {"images": images, "folders": folders}


@router.post("/batch-rename")
async def batch_rename(
    body: BatchRenameRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """执行批量重命名（文件夹 + 图片）"""
    folder_count = 0
    image_count = 0
    errors: list[str] = []

    # 先处理文件夹重命名
    for item in body.folder_renames or []:
        folder_path = normalize_path(item.path, allow_empty=False)
        if folder_path is None:
            errors.append(f"{item.path}: 路径不合法")
            continue
        new_name = (item.new_name or "").strip().strip("/")
        if not new_name:
            errors.append(f"{item.path}: 新名称不能为空")
            continue
        if ".." in new_name or "/" in new_name or "\\" in new_name:
            errors.append(f"{item.path}: 新名称不合法")
            continue

        parts = folder_path.rsplit("/", 1)
        parent = parts[0] if len(parts) == 2 else ""
        target_dir = PHOTOS_DIR / parent if parent else PHOTOS_DIR
        new_prefix = f"{parent}/{new_name}" if parent else new_name

        src_path = PHOTOS_DIR / folder_path
        if not src_path.exists() or not src_path.is_dir():
            errors.append(f"{folder_path}: 文件夹不存在")
            continue
        if Path(folder_path).name == new_name and (target_dir / new_name).resolve() == src_path.resolve():
            folder_count += 1
            continue
        dest_path = target_dir / new_name
        if dest_path.exists() and dest_path.resolve() != src_path.resolve():
            errors.append(f"{folder_path}: 目标已存在")
            continue

        try:
            await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{folder_path}: {e}")
            continue

        try:
            pf = path_filter_for_prefix(Image.relative_path, folder_path)
            stmt = select(Image).where(pf)
            result = await session.execute(stmt)
            for img in result.scalars().all():
                suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path):]
                new_rel = new_prefix + suffix
                old_cache = CACHE_DIR / cache_filename(img.relative_path)
                if old_cache.exists():
                    old_cache.unlink(missing_ok=True)
                img.relative_path = new_rel
                img.filename = Path(new_rel).name
                img.filename_natural = natural_sort_key(img.filename)
                img.relative_path_natural = natural_sort_key(new_rel)
                new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
                if new_full.exists() and new_full.is_file():
                    img.modified_at = await asyncio.to_thread(os.path.getmtime, new_full)
                    img.file_size = await asyncio.to_thread(os.path.getsize, new_full)
                    new_cache = CACHE_DIR / cache_filename(new_rel)
                    if new_full.suffix.lower() in VIDEO_EXTENSIONS:
                        await asyncio.to_thread(_generate_video_thumbnail, new_full, new_cache)
                    else:
                        await asyncio.to_thread(_generate_thumbnail, new_full, new_cache)
                session.add(img)
            invalidate_folder_tree_cache()
            folder_count += 1
            print(f"[api] 批量重命名文件夹: {folder_path} → {new_prefix}", flush=True)
        except Exception as e:
            errors.append(f"{folder_path}: 更新数据库失败 {e}")

    # 再处理图片重命名
    media_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    for item in body.image_renames or []:
        result = await session.execute(select(Image).where(Image.id == item.id))
        img = result.scalar_one_or_none()
        if not img:
            errors.append(f"id={item.id}: 图片不存在")
            continue
        new_filename = (item.new_filename or "").strip()
        if not new_filename:
            errors.append(f"{img.filename}: 新文件名不能为空")
            continue
        if _invalid_filename(new_filename):
            errors.append(f"{img.filename}: 文件名包含非法字符")
            continue
        if Path(new_filename).suffix.lower() not in media_extensions:
            errors.append(f"{img.filename}: 不支持的格式")
            continue

        src_path = PHOTOS_DIR / img.relative_path
        if not src_path.exists() or not src_path.is_file():
            errors.append(f"{img.filename}: 文件不存在")
            continue

        parent_dir = src_path.parent
        dest_path = parent_dir / new_filename
        if dest_path.exists() and dest_path.resolve() != src_path.resolve():
            dest_path = unique_path(parent_dir, new_filename, suffix_style="underscore")
            new_filename = dest_path.name

        new_rel = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")
        if new_rel == img.relative_path:
            image_count += 1
            continue

        try:
            await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{img.filename}: {e}")
            continue

        old_cache = CACHE_DIR / cache_filename(img.relative_path)
        if old_cache.exists():
            old_cache.unlink(missing_ok=True)
        img.relative_path = new_rel
        img.filename = dest_path.name
        img.filename_natural = natural_sort_key(dest_path.name)
        img.relative_path_natural = natural_sort_key(new_rel)
        img.modified_at = await asyncio.to_thread(os.path.getmtime, dest_path)
        img.file_size = await asyncio.to_thread(os.path.getsize, dest_path)
        new_cache = CACHE_DIR / cache_filename(new_rel)
        if dest_path.suffix.lower() in VIDEO_EXTENSIONS:
            await asyncio.to_thread(_generate_video_thumbnail, dest_path, new_cache)
        else:
            await asyncio.to_thread(_generate_thumbnail, dest_path, new_cache)
        session.add(img)
        image_count += 1
        print(f"[api] 批量重命名图片: {img.relative_path} → {new_rel}", flush=True)

    try:
        await session.commit()
    except Exception as e:
        await session.rollback()
        errors.append(f"提交失败: {e}")

    return {
        "ok": len(errors) == 0,
        "folder_count": folder_count,
        "image_count": image_count,
        "errors": errors,
    }


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
        folder_path = normalize_path(folder_path, allow_empty=False)
        if folder_path is None:
            continue
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        last_id = 0
        while True:
            stmt = (
                select(Image)
                .where(pf)
                .where(Image.id > last_id)
                .order_by(Image.id)
                .limit(_FOLDER_OP_BATCH_SIZE)
            )
            result = await session.execute(stmt)
            images = list(result.scalars().all())
            if not images:
                break
            for img in images:
                delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
                await session.delete(img)
                total_images += 1
                last_id = img.id or last_id
            await session.commit()
            await asyncio.sleep(0)
        folder_fs_path = PHOTOS_DIR / folder_path
        if folder_fs_path.exists() and folder_fs_path.is_dir():
            await asyncio.to_thread(shutil.rmtree, folder_fs_path, ignore_errors=True)
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
    folder_a = normalize_path(body.folder_a, allow_empty=False)
    folder_b = normalize_path(body.folder_b, allow_empty=False)
    if folder_a is None or folder_b is None:
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

    def _belongs_to(rel: str, prefix: str) -> bool:
        return rel == prefix or rel.startswith(prefix + "/")

    async def _collect_items(prefix: str, src: str) -> list[tuple[int, str, str]]:
        """分批加载，返回 [(id, relative_path, src), ...]"""
        pf = path_filter_for_prefix(Image.relative_path, prefix)
        items: list[tuple[int, str, str]] = []
        last_id = 0
        while True:
            stmt = (
                select(Image.id, Image.relative_path)
                .where(pf)
                .where(Image.id > last_id)
                .order_by(Image.id)
                .limit(_FOLDER_OP_BATCH_SIZE)
            )
            result = await session.execute(stmt)
            rows = result.fetchall()
            if not rows:
                break
            for rid, rp in rows:
                items.append((rid, rp, src))
                last_id = rid or last_id
            await asyncio.sleep(0)
        return items

    items_a = await _collect_items(folder_a, "a")
    items_b = await _collect_items(folder_b, "b")
    count_a, count_b = len(items_a), len(items_b)
    if body.target == "folder_b":
        target_prefix, source_prefix = folder_b, folder_a
        source_letter, target_letter = "a", "b"
        source_items, target_items = items_a, items_b
        source_path, target_path = path_a, path_b
    else:
        target_prefix, source_prefix = folder_a, folder_b
        source_letter, target_letter = "b", "a"
        source_items, target_items = items_b, items_a
        source_path, target_path = path_b, path_a
    if body.target == "auto" and count_b > count_a:
        target_prefix, source_prefix = folder_b, folder_a
        source_letter, target_letter = "a", "b"
        source_items, target_items = items_a, items_b
        source_path, target_path = path_a, path_b

    preferred = "a" if count_a >= count_b else "b"
    by_hash: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for item in items_a + items_b:
        img_id, rel_path, src = item
        full = photos_dir / rel_path
        if not full.is_file() or full.suffix.lower() not in media_extensions:
            continue
        h = await asyncio.to_thread(compute_file_md5, photos_dir, rel_path)
        if h is None:
            continue
        by_hash[h].append(item)
    to_keep: dict[str, tuple[int, str, str]] = {}
    to_delete: set[int] = set()
    for h, items in by_hash.items():
        from_preferred = [x for x in items if x[2] == preferred]
        from_other = [x for x in items if x[2] != preferred]
        if from_preferred:
            keeper = min(from_preferred, key=lambda x: x[1])
            to_keep[h] = keeper
            for x in from_preferred:
                if x[0] != keeper[0]:
                    to_delete.add(x[0])
            for x in from_other:
                to_delete.add(x[0])
        else:
            keeper = min(from_other, key=lambda x: x[1])
            to_keep[h] = keeper
            for x in from_other:
                if x[0] != keeper[0]:
                    to_delete.add(x[0])
    target_hashes = {
        h for h, k in to_keep.items()
        if _belongs_to(k[1], target_prefix)
    }
    to_move: list[tuple[int, str]] = []
    for h, items in by_hash.items():
        for img_id, rel_path, src in items:
            if img_id in to_delete or src != source_letter:
                continue
            if h not in target_hashes:
                to_move.append((img_id, rel_path))
            elif h in target_hashes:
                delete_image_files(rel_path, PHOTOS_DIR, CACHE_DIR)
                to_delete.add(img_id)
    for img_id in to_delete:
        result = await session.execute(select(Image).where(Image.id == img_id))
        img = result.scalar_one_or_none()
        if img:
            delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
            await session.delete(img)
    moved = 0
    for img_id, rel_path in to_move:
        result = await session.execute(select(Image).where(Image.id == img_id))
        img = result.scalar_one_or_none()
        if not img:
            continue
        suffix = rel_path[len(source_prefix):].lstrip("/")
        new_rel = f"{target_prefix}/{suffix}" if suffix else target_prefix
        new_full = target_path / suffix if suffix else target_path
        new_full.parent.mkdir(parents=True, exist_ok=True)
        if new_full.exists():
            new_full = unique_path(new_full.parent, new_full.name, suffix_style="paren")
            new_rel = str(new_full.relative_to(PHOTOS_DIR)).replace("\\", "/")
        try:
            await asyncio.to_thread(shutil.move, str(photos_dir / rel_path), str(new_full))
        except OSError as e:
            await session.rollback()
            return {"ok": False, "error": f"移动文件失败 {rel_path}: {e}"}
        old_cache = CACHE_DIR / cache_filename(rel_path)
        if old_cache.exists():
            old_cache.unlink(missing_ok=True)
        img.relative_path = new_rel
        img.filename = Path(new_rel).name
        img.filename_natural = natural_sort_key(img.filename)
        img.relative_path_natural = natural_sort_key(new_rel)
        img.modified_at = await asyncio.to_thread(os.path.getmtime, new_full)
        img.file_size = await asyncio.to_thread(os.path.getsize, new_full)
        new_cache = CACHE_DIR / cache_filename(new_rel)
        if new_full.suffix.lower() in VIDEO_EXTENSIONS:
            await asyncio.to_thread(_generate_video_thumbnail, new_full, new_cache)
        else:
            await asyncio.to_thread(_generate_thumbnail, new_full, new_cache)
        session.add(img)
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
    try:
        await session.commit()
        invalidate_folder_tree_cache()
        print(f"[api] 合并文件夹: {folder_a} + {folder_b} -> {target_prefix}, 移动 {moved} 个文件", flush=True)
        return {"ok": True, "moved": moved, "deleted": len(to_delete), "target": target_prefix}
    except IntegrityError:
        await session.rollback()
        return {"ok": False, "error": "路径冲突，请重试"}


@router.post("/create-folder")
async def create_folder(body: CreateFolderRequest):
    """在指定路径下创建子文件夹"""
    parent = normalize_path(body.path, allow_empty=True) or ""
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


@router.post("/folders/{folder_path:path}/thumbnails")
async def add_folder_thumbnail(
    folder_path: str,
    body: AddFolderThumbnailRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定图片设为文件夹缩略图（图片需在该文件夹下含子目录）。最多 4 张，FIFO：后设置的插在最前面，超出 4 张时移除末尾最旧的。"""
    folder_path = normalize_path(folder_path, allow_empty=False)
    if folder_path is None:
        raise HTTPException(status_code=400, detail="文件夹路径不合法")
    rel_path = normalize_path(body.relative_path, allow_empty=False)
    if rel_path is None:
        raise HTTPException(status_code=400, detail="图片路径不合法")
    if rel_path != folder_path and not rel_path.startswith(folder_path + "/"):
        raise HTTPException(status_code=400, detail="图片路径不在该文件夹下")
    img = await session.execute(select(Image).where(Image.relative_path == rel_path))
    img = img.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="图片不存在或已删除")
    existing = await session.execute(
        select(FolderThumbnail).where(FolderThumbnail.folder_path == folder_path)
    )
    existing_list = list(existing.scalars().all())
    # 若该图片已是缩略图，先删除旧记录（相当于刷新到最前）
    for ft in existing_list[:]:
        if ft.image_relative_path == rel_path:
            await session.delete(ft)
            existing_list.remove(ft)
            break
    # 新封面插在最前：现有记录 display_order += 1，新记录 display_order=0
    for ft in existing_list:
        ft.display_order += 1
        if ft.display_order >= 4:
            await session.delete(ft)  # 超出 4 张，移除末尾最旧的
        else:
            session.add(ft)
    session.add(FolderThumbnail(folder_path=folder_path, image_relative_path=rel_path, display_order=0))
    await session.commit()
    return {"ok": True, "folder_path": folder_path, "relative_path": rel_path}


@router.delete("/folders/{folder_path:path}/thumbnails/{image_path:path}")
async def remove_folder_thumbnail(
    folder_path: str,
    image_path: str,
    session: AsyncSession = Depends(get_async_session),
):
    """移除文件夹的指定缩略图。"""
    folder_path = normalize_path(folder_path, allow_empty=False)
    if folder_path is None:
        raise HTTPException(status_code=400, detail="文件夹路径不合法")
    rel_path = normalize_path(image_path, allow_empty=False)
    if rel_path is None:
        raise HTTPException(status_code=400, detail="图片路径不合法")
    result = await session.execute(
        select(FolderThumbnail).where(
            FolderThumbnail.folder_path == folder_path,
            FolderThumbnail.image_relative_path == rel_path,
        )
    )
    ft = result.scalar_one_or_none()
    if not ft:
        raise HTTPException(status_code=404, detail="该缩略图不存在")
    await session.delete(ft)
    await session.commit()
    return {"ok": True}


@router.get("/subfolders")
async def get_subfolders_api(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """获取指定路径下的直接子文件夹"""
    path = normalize_path(path, allow_empty=True)
    if path is None:
        return {"subfolders": []}
    path = path or ""
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
    """全局目录搜索（SQL 聚合 folder_counts，max_depth=10 支持更深目录）"""
    q = (q or "").strip()
    if not q:
        return {"dirs": []}
    full_dir_counts = dict(await get_folder_counts_for_search(session))
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
    path = normalize_path(path, allow_empty=True) or ""
    path_parts = path.split("/") if path else []
    folder_tree, _, folder_counts = await get_folder_tree_cached(
        PHOTOS_DIR, session=session
    )
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
