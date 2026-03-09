"""文件夹 API：移动、删除、合并、创建、搜索"""

import asyncio
import os
import shutil
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import (
    BATCH_COMMIT_SIZE,
    CACHE_DIR,
    FOLDER_OP_BATCH_SIZE,
    IN_CLAUSE_BATCH_SIZE,
    PHOTOS_DIR,
)
from app.models import FolderThumbnail, Image, ImageTag, get_async_session
from app.schemas import (
    AddFolderThumbnailRequest,
    BatchRenameInfoRequest,
    BatchRenameRequest,
    CreateFolderRequest,
    DeleteFoldersRequest,
    MergeFoldersRequest,
    MoveFoldersRequest,
    MoveImagesRequest,
    RenameFolderRequest,
    RenameImageRequest,
)
from app.services import task_state
from app.services.scanner import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.services.scheduler import scheduler
from app.services.task_queue import QueueTask, TaskQueue
from app.utils.folder_tree import (
    get_subfolders,
    invalidate_folder_tree_cache,
)
from app.utils.image_batch import (
    collect_image_items_by_prefix,
    iter_images_by_path_prefix,
)
from app.utils.image_path_update import update_image_path_and_regenerate_thumbnail
from app.utils.images import delete_image_files
from app.utils.path_utils import (
    escape_like,
    invalid_filename,
    normalize_path,
    path_filter_for_prefix,
)
from app.utils.search import search_match
from app.utils.unique_path import unique_path

router = APIRouter(prefix="/api", tags=["folders"])


@router.post("/move-images")
async def move_images(
    body: MoveImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定图片移动到目标文件夹"""
    if not task_state.start_task("move-images"):
        return {"moved": 0, "errors": ["有任务正在进行中，请等待完成后再操作"]}
    try:
        if not body.ids:
            return {"moved": 0, "errors": []}
        target_path = normalize_path(body.target_path, allow_empty=True)
        if target_path is None:
            return {"moved": 0, "errors": ["目标路径不合法"]}
        target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        images: list[Image] = []
        for i in range(0, len(body.ids), IN_CLAUSE_BATCH_SIZE):
            batch_ids = body.ids[i : i + IN_CLAUSE_BATCH_SIZE]
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
            if ext not in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
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
            await update_image_path_and_regenerate_thumbnail(
                img, new_rel, dest_path, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
            )
            session.add(img)
            moved += 1
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            errors.append("路径冲突（可能与已有文件重复），请重试")
        if moved > 0:
            invalidate_folder_tree_cache(target_path)
        task_state.end_task({"moved": moved})
        return {"moved": moved, "errors": errors}
    except Exception as e:
        task_state.fail_task(str(e))
        raise


@router.post("/move-folders")
async def move_folders(
    body: MoveFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定文件夹（含子文件夹和图片）移动到目标父目录"""
    if not task_state.start_task("move-folders"):
        return {"moved": 0, "errors": ["有任务正在进行中，请等待完成后再操作"]}
    try:
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
            async for images in iter_images_by_path_prefix(session, folder_path, FOLDER_OP_BATCH_SIZE):
                for img in images:
                    suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path) :]
                    new_rel = new_prefix + suffix
                    new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
                    await update_image_path_and_regenerate_thumbnail(
                        img, new_rel, new_full, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
                    )
                    session.add(img)
                    moved += 1
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
        task_state.end_task({"moved": moved})
        return {"moved": moved, "errors": errors}
    except Exception as e:
        task_state.fail_task(str(e))
        raise


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
        async for images in iter_images_by_path_prefix(session, folder_path, FOLDER_OP_BATCH_SIZE):
            for img in images:
                suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path) :]
                new_rel = new_prefix + suffix
                new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
                await update_image_path_and_regenerate_thumbnail(
                    img, new_rel, new_full, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
                )
                session.add(img)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return {"ok": False, "error": "路径冲突，请重试"}
            await asyncio.sleep(0)

        invalidate_folder_tree_cache(new_prefix)
        print(f"[api] 重命名文件夹: {folder_path} → {new_prefix}", flush=True)
        return {"ok": True, "path": new_prefix}
    except Exception as e:
        await session.rollback()
        return {"ok": False, "error": f"更新数据库失败: {e}"}


@router.post("/rename-image")
async def rename_image(
    body: RenameImageRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """重命名单张图片/视频（仅修改文件名，保持在同一目录）"""
    new_filename = (body.new_filename or "").strip()
    if not new_filename:
        return {"ok": False, "error": "新文件名不能为空"}
    if invalid_filename(new_filename):
        return {"ok": False, "error": "文件名包含非法字符"}

    media_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    if Path(new_filename).suffix.lower() not in media_extensions:
        return {"ok": False, "error": "不支持的文件格式"}

    result = await session.execute(select(Image).where(Image.id == body.id))
    img = result.scalar_one_or_none()
    if not img:
        return {"ok": False, "error": "媒体文件不存在"}

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

    await update_image_path_and_regenerate_thumbnail(img, new_rel, dest_path, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS)
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
        for i in range(0, len(body.image_ids), IN_CLAUSE_BATCH_SIZE):
            batch_ids = body.image_ids[i : i + IN_CLAUSE_BATCH_SIZE]
            stmt = select(Image).where(Image.id.in_(batch_ids))
            result = await session.execute(stmt)
            for img in result.scalars().all():
                images.append(
                    {
                        "id": img.id,
                        "filename": img.filename,
                        "relative_path": img.relative_path,
                        "modified_at": img.modified_at,
                    }
                )

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

    async def _flush_and_commit():
        try:
            await session.commit()
        except Exception as e:
            await session.rollback()
            errors.append(f"提交失败: {e}")

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
            async for images in iter_images_by_path_prefix(session, folder_path, FOLDER_OP_BATCH_SIZE):
                for img in images:
                    suffix = "" if img.relative_path == folder_path else img.relative_path[len(folder_path) :]
                    new_rel = new_prefix + suffix
                    new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
                    await update_image_path_and_regenerate_thumbnail(
                        img, new_rel, new_full, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
                    )
                    session.add(img)
                await _flush_and_commit()
            invalidate_folder_tree_cache(new_prefix)
            folder_count += 1
            print(f"[api] 批量重命名文件夹: {folder_path} → {new_prefix}", flush=True)
        except Exception as e:
            errors.append(f"{folder_path}: 更新数据库失败 {e}")

    # 再处理图片重命名：批量查询 + 定期提交 + 并发控制
    media_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    image_renames = body.image_renames or []
    if image_renames:
        id_list = [item.id for item in image_renames]
        result = await session.execute(select(Image).where(Image.id.in_(id_list)))
        image_map = {img.id: img for img in result.scalars().all()}
        pending_commits = 0

        async def _process_item(item):
            img = image_map.get(item.id)
            if not img:
                return ("error", f"id={item.id}: 图片不存在")
            new_filename = (item.new_filename or "").strip()
            if not new_filename:
                return ("error", f"{img.filename}: 新文件名不能为空")
            if invalid_filename(new_filename):
                return ("error", f"{img.filename}: 文件名包含非法字符")
            if Path(new_filename).suffix.lower() not in media_extensions:
                return ("error", f"{img.filename}: 不支持的格式")

            src_path = PHOTOS_DIR / img.relative_path
            if not src_path.exists() or not src_path.is_file():
                return ("error", f"{img.filename}: 文件不存在")

            parent_dir = src_path.parent
            dest_path = parent_dir / new_filename
            if dest_path.exists() and dest_path.resolve() != src_path.resolve():
                dest_path = unique_path(parent_dir, new_filename, suffix_style="underscore")
                new_filename = dest_path.name

            new_rel = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")
            if new_rel == img.relative_path:
                return ("skip", None)

            try:
                await asyncio.to_thread(shutil.move, str(src_path), str(dest_path))
            except OSError as e:
                return ("error", f"{img.filename}: {e}")

            await update_image_path_and_regenerate_thumbnail(
                img, new_rel, dest_path, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
            )
            session.add(img)
            print(f"[api] 批量重命名图片: {img.relative_path} → {new_rel}", flush=True)
            return ("ok", img)

        tasks = [_process_item(item) for item in image_renames]
        results = await asyncio.gather(
            *[
                scheduler.submit(task, priority=0, task_name=f"rename_{item.id}")
                for item, task in zip(image_renames, tasks)
            ],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"id={image_renames[i].id}: {result}")
            elif result[0] == "error":
                errors.append(result[1])
            elif result[0] == "ok":
                image_count += 1
                pending_commits += 1
                if pending_commits >= BATCH_COMMIT_SIZE:
                    await _flush_and_commit()
                    pending_commits = 0

        if pending_commits > 0:
            await _flush_and_commit()

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
    if not task_state.start_task("delete-folders"):
        return {"deleted_images": 0, "deleted_folders": 0, "error": "有任务正在进行中，请等待完成后再操作"}
    try:
        if not body.paths:
            return {"deleted_images": 0, "deleted_folders": 0}
        total_images = 0
        total_folders = 0
        for folder_path in body.paths:
            folder_path = normalize_path(folder_path, allow_empty=False)
            if folder_path is None:
                continue
            async for images in iter_images_by_path_prefix(session, folder_path, FOLDER_OP_BATCH_SIZE):
                for img in images:
                    delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
                    await session.delete(img)
                    total_images += 1
                await session.commit()
                await asyncio.sleep(0)
            folder_fs_path = PHOTOS_DIR / folder_path
            if folder_fs_path.exists() and folder_fs_path.is_dir():
                await asyncio.to_thread(shutil.rmtree, folder_fs_path, ignore_errors=True)
                total_folders += 1
        await session.commit()
        if total_folders > 0:
            invalidate_folder_tree_cache(body.paths[0] if len(body.paths) == 1 else None)
        task_state.end_task({"deleted_images": total_images, "deleted_folders": total_folders})
        return {"deleted_images": total_images, "deleted_folders": total_folders}
    except Exception as e:
        task_state.fail_task(str(e))
        raise


task_queue = TaskQueue()


async def _run_merge_folders_task(task: QueueTask) -> dict:
    """合并文件夹任务处理器（修正版）"""
    from sqlalchemy import delete, select
    from app.models import Image, async_session_factory

    body = task.params or {}
    folder_a = normalize_path(body.get("folder_a", ""), allow_empty=False)
    folder_b = normalize_path(body.get("folder_b", ""), allow_empty=False)
    target = body.get("target", "auto")
    duplicate_mode = body.get("duplicate_mode", "rename")  # skip / overwrite / rename

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

    # 收集两个文件夹内的所有图片（仅需 id 和 relative_path）
    async with async_session_factory() as session:
        # 辅助函数：获取某个前缀下的所有图片 (id, relative_path, source_letter)
        async def collect_items(prefix: str, letter: str):
            items = []
            stmt = select(Image.id, Image.relative_path).where(
                Image.relative_path.startswith(prefix + "/") if prefix else Image.relative_path != ""
            )
            if prefix:
                stmt = stmt.where((Image.relative_path == prefix) | (Image.relative_path.startswith(prefix + "/")))
            result = await session.execute(stmt)
            for row in result:
                img_id, rel_path = row
                items.append((img_id, rel_path, letter))
            return items

        items_a = await collect_items(folder_a, "a")
        items_b = await collect_items(folder_b, "b")
        all_items = items_a + items_b
        count_a, count_b = len(items_a), len(items_b)

        # 确定目标文件夹（保留图片较多的作为目标，或用户指定）
        if target == "folder_b" or (target == "auto" and count_b > count_a):
            target_prefix, source_prefix = folder_b, folder_a
            source_letter = "a"
            source_path, target_path = path_a, path_b
        else:
            target_prefix, source_prefix = folder_a, folder_b
            source_letter = "b"
            source_path, target_path = path_b, path_a

        # 构建目标路径到图片项的映射（目标路径 = target_prefix + 相对后缀）
        target_map = {}  # 目标路径 -> (img_id, source_letter)
        conflict_map = {}  # 目标路径 -> 冲突的图片项列表
        for img_id, rel_path, src in all_items:
            # 计算该文件在目标文件夹中的预期路径
            if rel_path.startswith(source_prefix + "/"):
                suffix = rel_path[len(source_prefix) + 1 :]  # 去掉源前缀和斜杠
                target_rel = f"{target_prefix}/{suffix}" if suffix else target_prefix
            elif rel_path == source_prefix:
                target_rel = target_prefix
            elif rel_path.startswith(target_prefix + "/") or rel_path == target_prefix:
                # 来自目标文件夹本身，无需移动，但需要记录目标路径以供冲突检测
                target_rel = rel_path
                src = "target"  # 标记为已经存在目标中的文件
            else:
                continue  # 不应该发生

            # 检查是否已存在相同目标路径
            if target_rel in target_map:
                # 发现冲突，记录到冲突列表
                if target_rel not in conflict_map:
                    conflict_map[target_rel] = [target_map[target_rel]]
                conflict_map[target_rel].append((img_id, rel_path, src))
                # 同时从 target_map 中移除（因为我们稍后将统一处理冲突）
                del target_map[target_rel]
            elif target_rel in conflict_map:
                # 已经是冲突路径，直接追加
                conflict_map[target_rel].append((img_id, rel_path, src))
            else:
                target_map[target_rel] = (img_id, rel_path, src)

        # 处理冲突：根据 duplicate_mode 决定每个冲突路径保留哪个文件
        to_move = []  # 需要移动的图片 (img_id, source_rel, target_rel)
        to_delete = []  # 需要删除的图片 (img_id)
        skipped_ids = set()  # 跳过移动的图片ID（用于 skip 模式）

        for target_rel, items in conflict_map.items():
            # 分离来源：来自源文件夹的（source_letter）和来自目标文件夹的（"target"）
            from_source = [it for it in items if it[2] == source_letter]
            from_target = [it for it in items if it[2] == "target"]
            # 来自源文件夹的冲突项
            for img_id, rel_path, src in from_source:
                if duplicate_mode == "skip":
                    # skip：不移动也不删除，保留在原处
                    skipped_ids.add(img_id)
                elif duplicate_mode == "overwrite":
                    # overwrite：用源文件覆盖目标文件
                    # 首先删除目标文件（如果存在）
                    for t_id, t_rel, t_src in from_target:
                        # 删除目标文件（文件系统 + 数据库）
                        delete_image_files(t_rel, PHOTOS_DIR, CACHE_DIR)
                        # 删除数据库记录
                        result = await session.execute(select(Image).where(Image.id == t_id))
                        img = result.scalar_one_or_none()
                        if img:
                            await session.execute(delete(ImageTag).where(ImageTag.image_id == t_id))
                            await session.delete(img)
                            to_delete.append(t_id)
                    # 将源文件加入移动列表
                    to_move.append((img_id, rel_path, target_rel))
                else:  # rename
                    # rename：为目标文件生成新文件名，保留原目标文件
                    # 目标路径已存在，需要生成不冲突的新路径
                    new_target_rel = target_rel  # 先尝试原目标路径
                    parent = Path(new_target_rel).parent
                    name = Path(new_target_rel).name
                    # 使用 unique_path 生成新文件名，但需要基于目标目录和文件名
                    # 注意：unique_path 期望一个完整的目标文件路径，我们构造临时路径
                    temp_path = target_path / name  # 假设新文件在目标文件夹根？实际上要考虑子目录
                    # 但 target_rel 可能包含子目录，需要准确定位
                    full_target = photos_dir / target_rel
                    new_full = unique_path(full_target.parent, full_target.name, suffix_style="paren")
                    new_target_rel = str(new_full.relative_to(photos_dir)).replace("\\", "/")
                    # 将源文件加入移动列表，使用新路径
                    to_move.append((img_id, rel_path, new_target_rel))
            # 来自目标文件夹的冲突项，如果没有被覆盖，则保留原样
            for img_id, rel_path, src in from_target:
                if duplicate_mode == "overwrite":
                    # 已经在上面被删除，无需额外操作
                    pass
                else:
                    # skip 或 rename 时，目标文件保留，无需移动或删除
                    pass

        # 处理无冲突的文件（target_map 中的项）
        for target_rel, (img_id, rel_path, src) in target_map.items():
            if src == source_letter:
                # 来自源文件夹，需要移动
                to_move.append((img_id, rel_path, target_rel))
            # 来自目标文件夹的文件，无需处理

        # 执行移动
        moved = 0
        for img_id, source_rel, target_rel in to_move:
            if img_id in skipped_ids:
                continue
            # 查询图片对象
            result = await session.execute(select(Image).where(Image.id == img_id))
            img = result.scalar_one_or_none()
            if not img:
                continue

            source_full = photos_dir / source_rel
            target_full = photos_dir / target_rel
            target_full.parent.mkdir(parents=True, exist_ok=True)

            # 检查目标是否存在（可能在 rename 时已确保不存在，但仍需防御）
            if target_full.exists():
                # 理论上在冲突处理中已解决，但以防万一
                if duplicate_mode == "skip":
                    continue
                elif duplicate_mode == "overwrite":
                    # 删除目标文件（记录也要删除）
                    delete_image_files(target_rel, PHOTOS_DIR, CACHE_DIR)
                    result = await session.execute(select(Image).where(Image.relative_path == target_rel))
                    existing_img = result.scalar_one_or_none()
                    if existing_img:
                        await session.execute(delete(ImageTag).where(ImageTag.image_id == existing_img.id))
                        await session.delete(existing_img)
                else:  # rename
                    # 重新生成唯一路径
                    target_full = unique_path(target_full.parent, target_full.name, suffix_style="paren")
                    target_rel = str(target_full.relative_to(photos_dir)).replace("\\", "/")

            # 移动文件
            try:
                await asyncio.to_thread(shutil.move, str(source_full), str(target_full))
            except OSError as e:
                await session.rollback()
                return {"ok": False, "error": f"移动文件失败 {source_rel}: {e}"}

            # 更新数据库记录
            await update_image_path_and_regenerate_thumbnail(
                img, target_rel, target_full, PHOTOS_DIR, CACHE_DIR, VIDEO_EXTENSIONS
            )
            session.add(img)
            moved += 1

            # 定期提交（可选）
            if moved % BATCH_COMMIT_SIZE == 0:
                await session.commit()

        # 删除所有标记为删除的图片（目前只有 overwrite 模式下删除目标文件，已直接在冲突处理中删除）
        # 但 to_delete 列表未使用，因为我们直接在冲突处理中删除了目标文件记录。
        # 若需要额外删除，可在此处处理 to_delete（当前为空）

        # 删除空源文件夹
        if source_path.exists():
            # 递归删除空目录
            for root, dirs, files in os.walk(str(source_path), topdown=False):
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except OSError:
                        pass
            try:
                os.rmdir(str(source_path))
            except OSError:
                pass  # 非空则保留

        await session.commit()
        invalidate_folder_tree_cache(target_prefix)
        print(
            f"[api] 合并文件夹: {folder_a} + {folder_b} -> {target_prefix}, 移动 {moved} 个文件",
            flush=True,
        )
        return {
            "ok": True,
            "moved": moved,
            "deleted": len(to_delete),
            "target": target_prefix,
        }


task_queue.register_handler("merge-folders", _run_merge_folders_task)


@router.post("/merge-folders")
async def merge_folders(
    body: MergeFoldersRequest,
):
    """合并两个文件夹，任务进入队列后台执行"""
    params = {
        "folder_a": body.folder_a,
        "folder_b": body.folder_b,
        "target": body.target,
        "duplicate_mode": body.duplicate_mode,
    }
    queue_id = await task_queue.add_task("merge-folders", params, priority=10)
    status = task_queue.get_status()
    running = status.get("merge-folders", {}).get("running")
    pending = status.get("merge-folders", {}).get("pending", [])
    position = 0
    for i, p in enumerate(pending):
        if p.get("queue_id") == queue_id:
            position = i
            break
    if running:
        return {"queue_id": queue_id, "status": "running", "position": 0}
    return {"queue_id": queue_id, "status": "queued", "position": position}


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
    invalidate_folder_tree_cache(parent)
    print(f"[api] 创建文件夹: {rel}", flush=True)
    return {"ok": True, "path": rel}


@router.post("/folders/{folder_path:path}/thumbnails")
async def add_folder_thumbnail(
    folder_path: str,
    body: AddFolderThumbnailRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定图片设为文件夹缩略图（图片需在该文件夹下含子目录）。
    最多 4 张，FIFO：后设置的插在最前面，超出 4 张时移除末尾最旧的。"""
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
    existing = await session.execute(select(FolderThumbnail).where(FolderThumbnail.folder_path == folder_path))
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
            {
                "name": s["name"],
                "full_path": s["full_path"],
                "image_count": s["image_count"],
            }
            for s in subfolders
        ]
    }


@router.get("/search-dirs")
async def search_dirs(
    q: str = "",
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """全局目录搜索（仅用 DB 的 folder_counts，空文件夹不参与搜索）"""
    q = (q or "").strip()
    if not q:
        return {"dirs": []}

    escaped = escape_like(q)
    like_filter = f"%{escaped}%"

    sql = text("""
        SELECT prefix, COUNT(*) AS cnt FROM (
            SELECT SUBSTRING_INDEX(relative_path, '/', 1) AS prefix
            FROM images WHERE relative_path LIKE :like_filter AND relative_path LIKE '%/%'
            UNION ALL
            SELECT SUBSTRING_INDEX(relative_path, '/', 2) AS prefix
            FROM images WHERE relative_path LIKE :like_filter AND relative_path LIKE '%/%/%'
            UNION ALL
            SELECT SUBSTRING_INDEX(relative_path, '/', 3) AS prefix
            FROM images WHERE relative_path LIKE :like_filter AND relative_path LIKE '%/%/%/%'
            UNION ALL
            SELECT SUBSTRING_INDEX(relative_path, '/', 4) AS prefix
            FROM images WHERE relative_path LIKE :like_filter AND relative_path LIKE '%/%/%/%/%'
        ) t
        WHERE prefix IS NOT NULL AND prefix != ''
        GROUP BY prefix
        LIMIT :limit_val
    """)
    result = await session.execute(sql, {"like_filter": like_filter, "limit_val": limit * 5})
    rows = result.fetchall()

    matched = []
    for row in rows:
        dir_path = row[0]
        count = row[1]
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
    """列出指定路径下的直接子文件夹（使用 get_subfolders，避免 _scan_dirs）"""
    path = normalize_path(path, allow_empty=True) or ""
    pf = path_filter_for_prefix(Image.relative_path, path) if path else None
    subfolders = await get_subfolders(session, PHOTOS_DIR, path, pf, sort_by="filename", sort_order="asc")
    subdirs = [{"path": s["full_path"], "name": s["name"], "image_count": s["image_count"]} for s in subfolders]
    return {"dirs": subdirs, "parent": path}
