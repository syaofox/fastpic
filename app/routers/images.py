"""图片 API：删除、下载、上传、信息"""

import asyncio
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import delete, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import (
    CACHE_DIR,
    IN_CLAUSE_BATCH_SIZE,
    MAX_UPLOAD_FILE_SIZE,
    MAX_UPLOAD_TOTAL_SIZE,
    PHOTOS_DIR,
    UPLOAD_PARALLEL,
)
from app.models import (
    FolderThumbnail,
    Image,
    ImageTag,
    Tag,
    async_session_factory,
    get_async_session,
    natural_sort_key,
)
from app.schemas import ApiResponse, DeleteImagesRequest, DownloadZipRequest
from app.services import task_state
from app.services.scanner import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.services.task_manager import task_manager
from app.utils.folder_tree import invalidate_folder_tree_cache
from app.utils.format import format_file_size
from app.utils.hash_utils import compute_file_md5_by_path
from app.utils.image_records import create_image_record
from app.utils.images import cache_filename
from app.utils.path_utils import (
    normalize_path,
    path_filter_for_prefix,
    resolve_and_validate_relative_path,
)
from app.utils.unique_path import unique_path


class UploadError(Exception):
    pass


class FileSizeExceededError(UploadError):
    pass


class DuplicateFileError(UploadError):
    pass


router = APIRouter(prefix="/api", tags=["images"])


async def _get_existing_hashes_from_db(target_dir: Path, subdirs: set[str] | None = None) -> dict[str, str]:
    """从数据库获取已有图片的 MD5 哈希，返回 hash -> relative_path。
    比读取所有文件计算哈希快得多。"""
    existing_hashes: dict[str, str] = {}
    prefix = str(target_dir.relative_to(PHOTOS_DIR)).replace("\\", "/").strip("/")
    if prefix:
        prefix += "/"

    async with async_session_factory() as sess:
        if subdirs:
            for subdir in subdirs:
                subdir = (subdir or "").strip().replace("\\", "/").strip("/")
                if subdir:
                    search_prefix = f"{prefix}{subdir}/"
                else:
                    search_prefix = prefix
                if search_prefix:
                    result = await sess.execute(
                        select(Image.md5_hash, Image.relative_path).where(Image.relative_path.like(f"{search_prefix}%"))
                    )
                else:
                    result = await sess.execute(select(Image.md5_hash, Image.relative_path))
                for md5_hash, rel_path in result.all():
                    if md5_hash:
                        existing_hashes[md5_hash] = rel_path
        else:
            result = await sess.execute(
                select(Image.md5_hash, Image.relative_path).where(Image.relative_path.like(f"{prefix}%"))
            )
            for md5_hash, rel_path in result.all():
                if md5_hash:
                    existing_hashes[md5_hash] = rel_path
    return existing_hashes


def _compute_existing_hashes(target_dir: Path, image_extensions: set[str]) -> dict[str, str]:
    """同步计算目标目录中已有图片的 MD5 哈希，返回 hash -> 相对路径（仅根目录直接子文件）"""
    existing_hashes: dict[str, str] = {}
    if not target_dir.is_dir():
        return existing_hashes
    for existing in target_dir.iterdir():
        if existing.is_file() and existing.suffix.lower() in image_extensions:
            h = compute_file_md5_by_path(existing)
            if h is not None:
                existing_hashes[h] = existing.name
    return existing_hashes


def _compute_existing_hashes_recursive(target_dir: Path, media_extensions: set[str]) -> dict[str, str]:
    """递归计算目标目录及子目录中已有媒体文件的 MD5 哈希，返回 hash -> 相对路径（相对 target_dir）"""
    existing_hashes: dict[str, str] = {}
    if not target_dir.is_dir():
        return existing_hashes

    def _walk(base: Path, prefix: str) -> None:
        try:
            for p in base.iterdir():
                rel = f"{prefix}/{p.name}" if prefix else p.name
                if p.is_file() and p.suffix.lower() in media_extensions:
                    h = compute_file_md5_by_path(p)
                    if h is not None:
                        existing_hashes[h] = rel.replace("\\", "/")
                elif p.is_dir():
                    _walk(p, rel)
        except OSError:
            pass

    _walk(target_dir, "")
    return existing_hashes


def _compute_existing_hashes_for_subdirs(
    target_dir: Path, subdirs: set[str], media_extensions: set[str]
) -> dict[str, str]:
    """仅对指定子目录计算已有媒体文件的 MD5 哈希，返回 hash -> 相对路径（相对 target_dir）。
    用于文件夹上传时按需哈希，避免扫描整个图库。"""
    existing_hashes: dict[str, str] = {}
    for subdir in subdirs:
        subdir = (subdir or "").strip().replace("\\", "/").strip("/")
        dir_path = target_dir / subdir if subdir else target_dir
        if not dir_path.is_dir():
            continue
        if subdir:
            # 递归哈希子目录，路径加前缀
            partial = _compute_existing_hashes_recursive(dir_path, media_extensions)
            prefix = subdir + "/"
            for h, rel in partial.items():
                existing_hashes[h] = (prefix + rel).replace("//", "/")
        else:
            # 根目录：仅直接子文件
            partial = _compute_existing_hashes(dir_path, media_extensions)
            existing_hashes.update(partial)
    return existing_hashes


def _sanitize_upload_filename(filename: str) -> str | None:
    """校验并规范化上传文件名中的路径，非法返回 None。允许纯文件名或 subpath/filename。"""
    if not filename or not filename.strip():
        return None
    p = filename.strip().replace("\\", "/").strip("/")
    if ".." in p or p.startswith("/"):
        return None
    return p


@router.post("/delete-images")
async def delete_images(
    body: DeleteImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """删除指定 ID 的图片（数据库记录 + 原图 + 缓存），分批处理支持大批量"""
    if not task_state.start_task("delete-images"):
        return ApiResponse.error("有任务正在进行中，请等待完成后再操作")
    _task = await task_manager.create_task(session, "delete-images", "正在删除文件", len(body.ids or []))
    try:
        if not body.ids:
            await task_manager.complete_task(_task.id, session, "无文件需删除")
            return ApiResponse.success({"deleted": 0})

        all_images = []
        for i in range(0, len(body.ids), IN_CLAUSE_BATCH_SIZE):
            batch_ids = body.ids[i : i + IN_CLAUSE_BATCH_SIZE]
            stmt = select(Image).where(Image.id.in_(batch_ids))
            result = await session.execute(stmt)
            all_images.extend(result.scalars().all())

        photo_paths = []
        cache_paths = []
        for img in all_images:
            photo_paths.append(PHOTOS_DIR / img.relative_path)
            cache_name = cache_filename(img.relative_path)
            cache_paths.append(CACHE_DIR / cache_name)

        def _delete_files(paths: list[Path]):
            for p in paths:
                if p.exists():
                    p.unlink(missing_ok=True)

        await asyncio.gather(
            asyncio.to_thread(_delete_files, photo_paths),
            asyncio.to_thread(_delete_files, cache_paths),
        )

        deleted_paths = [img.relative_path for img in all_images]
        if deleted_paths:
            await session.execute(delete(FolderThumbnail).where(FolderThumbnail.image_relative_path.in_(deleted_paths)))

        for img in all_images:
            await session.delete(img)
        await session.commit()

        if len(all_images) > 0:
            parent_paths = set()
            for img in all_images:
                parent = str(Path(img.relative_path).parent)
                if parent != ".":
                    parent_paths.add(parent)
            if len(parent_paths) == 1:
                invalidate_folder_tree_cache(next(iter(parent_paths)))
            elif len(parent_paths) > 1:
                invalidate_folder_tree_cache()
            else:
                invalidate_folder_tree_cache("")
        await task_manager.complete_task(_task.id, session, f"已删除 {len(all_images)} 项")
        return ApiResponse.success({"deleted": len(all_images)}, f"已删除 {len(all_images)} 项")
    except Exception as e:
        await task_manager.fail_task(_task.id, session, str(e))
        return ApiResponse.error(str(e))


@router.get("/download/image")
async def download_image(
    id: int | None = None,
    relative_path: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """单图下载"""
    if id is not None:
        result = await session.execute(select(Image).where(Image.id == id))
        img = result.scalar_one_or_none()
        if not img:
            raise HTTPException(status_code=404, detail="媒体文件不存在")
        rel = img.relative_path
        filename = img.filename
    elif relative_path:
        full = resolve_and_validate_relative_path(relative_path, PHOTOS_DIR)
        if not full:
            raise HTTPException(status_code=400, detail="路径不合法或文件不存在")
        rel = relative_path.strip().strip("/")
        filename = full.name
    else:
        raise HTTPException(status_code=400, detail="请提供 id 或 relative_path")
    file_path = PHOTOS_DIR / rel
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
    )


@router.post("/download/zip")
async def download_zip(
    body: DownloadZipRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """批量下载：打包为 ZIP"""
    rel_paths: set[str] = set()
    if body.image_ids:
        for i in range(0, len(body.image_ids), IN_CLAUSE_BATCH_SIZE):
            batch_ids = body.image_ids[i : i + IN_CLAUSE_BATCH_SIZE]
            result = await session.execute(select(Image.relative_path).where(Image.id.in_(batch_ids)))
            for row in result.fetchall():
                rel_paths.add(row[0])
    for raw_path in body.folder_paths or []:
        path = normalize_path(raw_path, allow_empty=False)
        if path is None:
            continue
        pf = path_filter_for_prefix(Image.relative_path, path)
        result = await session.execute(select(Image.relative_path).where(pf))
        for row in result.fetchall():
            rel_paths.add(row[0])
    existing = [rp for rp in rel_paths if (PHOTOS_DIR / rp).is_file()]
    if not existing:
        raise HTTPException(status_code=400, detail="没有可下载的文件")
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rp in existing:
                full = PHOTOS_DIR / rp
                if full.is_file():
                    zf.write(full, rp)
        background_tasks.add_task(os.unlink, tmp_path)
        return FileResponse(
            path=tmp_path,
            filename="download.zip",
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="download.zip"'},
        )
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail="打包下载失败")


@router.get("/image-info/{image_id:int}")
async def get_image_info(
    image_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """获取单张图片的详细信息"""
    from datetime import datetime

    stmt = (
        select(Image, func.group_concat(Tag.name).label("tags"))
        .outerjoin(ImageTag, ImageTag.image_id == Image.id)
        .outerjoin(Tag, Tag.id == ImageTag.tag_id)
        .where(Image.id == image_id)
        .group_by(Image.id)
    )
    result = await session.execute(stmt)
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="媒体文件不存在或已被删除")
    img = row[0]
    tags = sorted((row[1] or "").split(",")) if row[1] else []
    full_path = str((PHOTOS_DIR / img.relative_path).resolve())
    modified_dt = datetime.fromtimestamp(img.modified_at)
    modified_str = modified_dt.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "full_path": full_path,
        "filename": img.filename,
        "relative_path": img.relative_path,
        "resolution": f"{img.width} × {img.height}" if (img.width and img.height) else "—",
        "file_size": format_file_size(img.file_size or 0),
        "modified_at": modified_str,
        "tags": tags,
    }


@router.post("/upload")
async def upload_images(request: Request):
    """上传图片或视频到指定路径，支持子目录结构（拖拽/选择文件夹）"""
    from starlette.formparsers import MultiPartParser

    from app.services.scanner import get_media_metadata_and_thumbnail
    from app.utils.tags import DAMAGED_TAG_NAME, add_tag_to_image, ensure_tag_exists

    # 提高 SpooledTemporaryFile 溢出阈值到 100MB，避免大量小图片上传时创建过多磁盘临时文件耗尽 FD
    MultiPartParser.spool_max_size = max(100 * 1024 * 1024, MAX_UPLOAD_FILE_SIZE)

    form_data = await request.form(
        max_part_size=MAX_UPLOAD_FILE_SIZE + 1024,
        max_files=2000,
    )

    raw_path = form_data.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) else ""
    raw_dup = form_data.get("on_duplicate")
    on_duplicate = (raw_dup or "skip").strip() if isinstance(raw_dup, str) else "skip"
    files = [f for f in (form_data.getlist("files") or []) if hasattr(f, "read") and hasattr(f, "filename")]

    raw_paths = form_data.get("file_paths")
    try:
        file_paths = json.loads(raw_paths) if isinstance(raw_paths, str) else []
    except (json.JSONDecodeError, TypeError):
        file_paths = []
    if not isinstance(file_paths, list):
        file_paths = []
    while len(file_paths) < len(files):
        file_paths.append("")
    file_paths = file_paths[: len(files)]

    if not files:
        return {
            "uploaded": 0,
            "skipped": 0,
            "errors": ["未收到任何文件，请检查是否选择了图片或视频"],
        }

    target_path = normalize_path(path, allow_empty=True) or ""
    print(f"[upload] 开始: {len(files)} 个文件 -> {target_path or '根目录'}", flush=True)
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as _task_sess:
        _task = await task_manager.create_task(_task_sess, "upload", "正在上传文件", len(files))
    _task_id = _task.id

    has_subpath = any("/" in (fp or "") or "\\" in (fp or "") for fp in file_paths)
    if has_subpath:
        # 仅哈希即将写入的子目录，避免扫描整个图库
        subdirs: set[str] = set()
        for i, f in enumerate(files):
            display_name = (file_paths[i] if i < len(file_paths) else "") or (getattr(f, "filename", "") or "")
            sanitized = _sanitize_upload_filename(display_name.strip())
            if sanitized is None:
                continue
            parts = Path(sanitized).parts
            if len(parts) <= 1:
                subdirs.add("")
            else:
                subdirs.add("/".join(parts[:-1]))
        existing_hashes = await _get_existing_hashes_from_db(target_dir, subdirs)
    else:
        existing_hashes = await _get_existing_hashes_from_db(target_dir)

    uploaded = 0
    skipped = 0
    errors: list[str] = []
    total_uploaded_bytes = 0
    sem = asyncio.Semaphore(UPLOAD_PARALLEL)

    async def _process_one(
        i: int,
        f,
        display_name: str,
        sanitized: str,
        content: bytes,
        content_hash: str,
        dest: Path,
        is_video: bool,
        on_duplicate: str,
    ) -> tuple[bool, bool, str | None]:
        """处理单个文件：写入、缩略图、入库。返回 (uploaded, skipped, error_msg)"""
        if len(content) > MAX_UPLOAD_FILE_SIZE:
            raise FileSizeExceededError(
                f"{display_name}: 单文件超过大小限制 ({MAX_UPLOAD_FILE_SIZE // (1024 * 1024)}MB)"
            )

        rel_path = str(dest.relative_to(PHOTOS_DIR)).replace("\\", "/")
        async with sem:
            try:
                dest.write_bytes(content)
                dest_rel = str(dest.relative_to(target_dir)).replace("\\", "/")
                existing_hashes[content_hash] = dest_rel
                cache_name = cache_filename(rel_path)
                cache_path = CACHE_DIR / cache_name
                data = await asyncio.to_thread(get_media_metadata_and_thumbnail, dest, cache_path, is_video)
                if data is None:
                    print(f"[upload] 处理失败: {display_name}", flush=True)
                    return False, False, f"{display_name}: 处理失败"
                width, height, modified_at, file_size, is_corrupted = data
                async with async_session_factory() as sess:
                    existing_record = (
                        await sess.execute(select(Image).where(Image.relative_path == rel_path))
                    ).scalar_one_or_none()
                    if existing_record:
                        existing_record.filename = dest.name
                        existing_record.filename_natural = natural_sort_key(dest.name)
                        existing_record.relative_path_natural = natural_sort_key(rel_path)
                        existing_record.modified_at = modified_at
                        existing_record.file_size = file_size
                        existing_record.width = width
                        existing_record.height = height
                        existing_record.media_type = "video" if is_video else "image"
                        existing_record.md5_hash = content_hash
                        sess.add(existing_record)
                        record = existing_record
                    else:
                        record = create_image_record(
                            filename=dest.name,
                            relative_path=rel_path,
                            modified_at=modified_at,
                            file_size=file_size,
                            width=width,
                            height=height,
                            media_type="video" if is_video else "image",
                            md5_hash=content_hash,
                        )
                        sess.add(record)
                    if is_corrupted:
                        damaged_tag = await ensure_tag_exists(sess, DAMAGED_TAG_NAME)
                        if damaged_tag:
                            await sess.flush()
                            await add_tag_to_image(sess, record.id, damaged_tag)
                    try:
                        await sess.commit()
                        print(f"[upload] 成功: {rel_path}", flush=True)
                        return True, False, None
                    except IntegrityError:
                        await sess.rollback()
                        print(f"[upload] 成功(竞态): {rel_path}", flush=True)
                        return True, False, None
            except Exception as e:
                print(f"[upload] 失败: {display_name} - {e}", flush=True)
                return False, False, f"{display_name}: {str(e)}"

    tasks: list[tuple[int, object, str, str, bytes, str, Path, bool, str]] = []
    for i, f in enumerate(files):
        display_name = file_paths[i] if i < len(file_paths) else (f.filename or "")
        if not display_name.strip():
            display_name = f.filename or ""
        sanitized = _sanitize_upload_filename(display_name)
        if sanitized is None:
            errors.append(f"{display_name or '未知'}: 路径不合法")
            continue
        ext = Path(sanitized).suffix.lower()
        if ext not in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
            errors.append(f"{display_name or '未知'}: 不支持的格式 {ext}")
            continue
        if total_uploaded_bytes >= MAX_UPLOAD_TOTAL_SIZE:
            errors.append(
                f"{display_name or '未知'}: 本次上传总大小已达限制 ({MAX_UPLOAD_TOTAL_SIZE // (1024 * 1024)}MB)"
            )
            continue
        is_video = ext in VIDEO_EXTENSIONS
        try:
            content = await f.read(MAX_UPLOAD_FILE_SIZE + 1)
        except Exception as e:
            errors.append(f"{display_name}: 读取失败 {e}")
            continue
        finally:
            await f.close()
        if total_uploaded_bytes + len(content) > MAX_UPLOAD_TOTAL_SIZE:
            errors.append(f"{display_name}: 本次上传总大小将超限")
            continue
        total_uploaded_bytes += len(content)
        content_hash = hashlib.md5(content).hexdigest()
        base_name = Path(sanitized).name
        if content_hash in existing_hashes:
            if on_duplicate == "skip":
                skipped += 1
                continue
            elif on_duplicate == "overwrite":
                dest = target_dir / existing_hashes[content_hash]
            else:
                dest_parent = (target_dir / Path(sanitized).parent) if "/" in sanitized else target_dir
                dest_parent.mkdir(parents=True, exist_ok=True)
                dest = unique_path(dest_parent, base_name, suffix_style="underscore")
        else:
            dest_parent = (target_dir / Path(sanitized).parent) if "/" in sanitized else target_dir
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest = dest_parent / base_name
            if dest.exists():
                if on_duplicate == "skip":
                    skipped += 1
                    continue
                elif on_duplicate == "overwrite":
                    pass
                else:
                    dest = unique_path(dest_parent, base_name, suffix_style="underscore")
        tasks.append((i, f, display_name, sanitized, content, content_hash, dest, is_video, on_duplicate))

    results = await asyncio.gather(
        *[_process_one(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]) for t in tasks],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, FileSizeExceededError):
            errors.append(f"文件大小超限: {str(r)}")
        elif isinstance(r, DuplicateFileError):
            skipped += 1
        elif isinstance(r, Exception):
            errors.append(str(r))
        elif isinstance(r, tuple):
            u, s, err = r
            if u:
                uploaded += 1
            elif s:
                skipped += 1
            elif err:
                errors.append(err)

    async with async_session_factory() as _task_sess:
        await task_manager.update_progress(
            _task_id,
            _task_sess,
            processed_items=uploaded,
            total_items=len(files),
            current_operation=f"已完成 ({uploaded}/{len(files)})",
        )
    if uploaded > 0:
        invalidate_folder_tree_cache(target_path)

    result = {"uploaded": uploaded, "skipped": skipped, "errors": errors}
    async with async_session_factory() as _task_sess:
        if uploaded > 0 or not errors:
            summary = f"已上传 {uploaded} 个文件"
            if skipped:
                summary += f"，{skipped} 个跳过"
            await task_manager.complete_task(_task_id, _task_sess, summary)
        else:
            await task_manager.fail_task(_task_id, _task_sess, "; ".join(errors[:3]))

    print(
        f"[upload] 完成: {uploaded} 成功, {skipped} 跳过, {len(errors)} 失败",
        flush=True,
    )
    return result
