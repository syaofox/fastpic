"""图片 API：删除、下载、上传、信息"""
import asyncio
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy.exc import IntegrityError
from fastapi.responses import FileResponse
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR
from models import Image, Tag, ImageTag, get_async_session, natural_sort_key
from scanner import _cache_filename, IMAGE_EXTENSIONS
from schemas import DeleteImagesRequest, DownloadZipRequest
from utils.path_utils import escape_like, path_filter_for_prefix, resolve_and_validate_relative_path
from utils.unique_path import unique_path
from utils.format import format_file_size
from utils.images import delete_image_files

router = APIRouter(prefix="/api", tags=["images"])


def _compute_existing_hashes(target_dir: Path, image_extensions: set[str]) -> dict[str, str]:
    """同步计算目标目录中已有图片的 MD5 哈希"""
    existing_hashes: dict[str, str] = {}
    if not target_dir.is_dir():
        return existing_hashes
    for existing in target_dir.iterdir():
        if existing.is_file() and existing.suffix.lower() in image_extensions:
            try:
                h = hashlib.md5(existing.read_bytes()).hexdigest()
                existing_hashes[h] = existing.name
            except OSError:
                pass
    return existing_hashes


@router.post("/delete-images")
async def delete_images(
    body: DeleteImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """删除指定 ID 的图片（数据库记录 + 原图 + 缓存）"""
    if not body.ids:
        return {"deleted": 0}
    stmt = select(Image).where(Image.id.in_(body.ids))
    result = await session.execute(stmt)
    images = list(result.scalars().all())
    deleted = 0
    for img in images:
        delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
        await session.delete(img)
        deleted += 1
    await session.commit()
    return {"deleted": deleted}


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
            raise HTTPException(status_code=404, detail="图片不存在")
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
        result = await session.execute(select(Image.relative_path).where(Image.id.in_(body.image_ids)))
        for row in result.fetchall():
            rel_paths.add(row[0])
    for raw_path in body.folder_paths or []:
        path = (raw_path or "").strip().strip("/")
        if not path or ".." in path or path.startswith("/"):
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

    result = await session.execute(select(Image).where(Image.id == image_id))
    img = result.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="图片不存在或已被删除")
    tag_result = await session.execute(
        select(Tag.name)
        .join(ImageTag, ImageTag.tag_id == Tag.id)
        .where(ImageTag.image_id == image_id)
        .order_by(Tag.name)
    )
    tags = [r[0] for r in tag_result.fetchall()]
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
async def upload_images(
    path: str = Form(""),
    on_duplicate: str = Form("skip"),
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_async_session),
):
    """上传图片到指定路径"""
    from scanner import get_media_metadata_and_thumbnail

    target_path = (path or "").strip().strip("/")
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_hashes = await asyncio.to_thread(
        _compute_existing_hashes, target_dir, IMAGE_EXTENSIONS
    )
    uploaded = 0
    skipped = 0
    errors = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            errors.append(f"{f.filename}: 不支持的格式 {ext}")
            continue
        try:
            content = await f.read()
        except Exception as e:
            errors.append(f"{f.filename}: 读取失败 {e}")
            continue
        content_hash = hashlib.md5(content).hexdigest()
        is_overwrite = False
        if content_hash in existing_hashes:
            if on_duplicate == "skip":
                skipped += 1
                continue
            elif on_duplicate == "overwrite":
                dest = target_dir / existing_hashes[content_hash]
                is_overwrite = True
            else:
                dest = unique_path(target_dir, Path(f.filename).name, suffix_style="underscore")
        else:
            safe_name = Path(f.filename).name
            dest = target_dir / safe_name
            if dest.exists():
                if on_duplicate == "skip":
                    skipped += 1
                    continue
                elif on_duplicate == "overwrite":
                    is_overwrite = True
                else:
                    dest = unique_path(target_dir, Path(f.filename).name, suffix_style="underscore")
        try:
            dest.write_bytes(content)
            existing_hashes[content_hash] = dest.name
            rel_path = str(dest.relative_to(PHOTOS_DIR)).replace("\\", "/")
            existing_record = (
                await session.execute(select(Image).where(Image.relative_path == rel_path))
            ).scalar_one_or_none()
            cache_name = _cache_filename(rel_path)
            cache_path = CACHE_DIR / cache_name
            data = await asyncio.to_thread(
                get_media_metadata_and_thumbnail, dest, cache_path, False
            )
            if data is None:
                errors.append(f"{f.filename}: 处理失败")
                continue
            width, height, modified_at, file_size = data
            if existing_record:
                existing_record.filename = dest.name
                existing_record.filename_natural = natural_sort_key(dest.name)
                existing_record.relative_path_natural = natural_sort_key(rel_path)
                existing_record.modified_at = modified_at
                existing_record.file_size = file_size
                existing_record.width = width
                existing_record.height = height
                session.add(existing_record)
            else:
                record = Image(
                    filename=dest.name,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
                    filename_natural=natural_sort_key(dest.name),
                    relative_path_natural=natural_sort_key(rel_path),
                )
                session.add(record)
            try:
                await session.commit()
                uploaded += 1
            except IntegrityError:
                await session.rollback()
                # 竞态：watcher 已先入库，视为成功
                uploaded += 1
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")
    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}
