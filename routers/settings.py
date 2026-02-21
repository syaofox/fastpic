"""设置/维护 API"""
import asyncio
from collections import defaultdict

from fastapi import APIRouter, Depends, Request
from sqlmodel import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR
from models import Image, get_async_session
from scanner import scan_photos, scan_videos, cleanup_database, _cache_filename
from scan_state import begin_scan, end_scan
from schemas import ScanDuplicatesRequest
from app_common import templates
from utils.path_utils import path_filter_for_prefix
from utils.hash_utils import compute_file_md5
from utils.stats import stats_folder_and_cache

router = APIRouter(tags=["settings"])


@router.get("/settings")
async def settings_page(request: Request):
    """设置页面"""
    return templates.TemplateResponse("settings.html", {"request": request})


@router.get("/api/scan-status")
async def get_scan_status():
    """返回当前是否有扫描任务在进行"""
    from scan_state import is_scanning
    return {"scanning": is_scanning()}


@router.post("/scan")
async def trigger_scan():
    """手动触发扫描"""
    begin_scan()
    try:
        n_img = await scan_photos(PHOTOS_DIR, CACHE_DIR)
        n_vid = await scan_videos(PHOTOS_DIR, CACHE_DIR)
        return {"scanned": n_img + n_vid, "images": n_img, "videos": n_vid}
    finally:
        end_scan()


@router.post("/api/cleanup")
async def trigger_cleanup():
    """手动触发数据库清理同步"""
    result = await cleanup_database(PHOTOS_DIR, CACHE_DIR)
    return result


@router.post("/api/scan-duplicates")
async def scan_duplicates(
    body: ScanDuplicatesRequest | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """扫描重复文件"""
    folder_path = (body.folder_path if body else None) or ""
    folder_path = folder_path.strip().strip("/")
    if folder_path:
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        result = await session.execute(select(Image).where(pf))
    else:
        result = await session.execute(select(Image))
    all_images = list(result.scalars().all())
    photos_dir = PHOTOS_DIR.resolve()
    by_size: dict[int, list[Image]] = defaultdict(list)
    for img in all_images:
        by_size[img.file_size].append(img)
    candidate_groups = [g for g in by_size.values() if len(g) > 1]
    if not candidate_groups:
        return {"groups": []}
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for group in candidate_groups:
        for img in group:
            h = await asyncio.to_thread(compute_file_md5, photos_dir, img.relative_path)
            if h is None:
                continue
            by_hash[h].append({
                "id": img.id,
                "relative_path": img.relative_path,
                "filename": img.filename,
                "file_size": img.file_size,
                "modified_at": img.modified_at,
                "cache_key": _cache_filename(img.relative_path),
            })
    groups = []
    for content_hash, items in by_hash.items():
        if len(items) > 1:
            groups.append({
                "content_hash": content_hash,
                "file_size": items[0]["file_size"],
                "items": items,
            })
    return {"groups": groups}


@router.get("/api/stats")
async def get_stats(session: AsyncSession = Depends(get_async_session)):
    """获取数据库和文件系统统计信息"""
    total_images = (await session.execute(select(func.count(Image.id)))).scalar() or 0
    total_size = (await session.execute(select(func.sum(Image.file_size)))).scalar() or 0
    folder_count, cache_count, cache_size = await asyncio.to_thread(
        stats_folder_and_cache, PHOTOS_DIR, CACHE_DIR
    )
    return {
        "total_images": total_images,
        "total_size": total_size,
        "folder_count": folder_count,
        "cache_count": cache_count,
        "cache_size": cache_size,
        "photos_dir": str(PHOTOS_DIR.resolve()),
        "cache_dir": str(CACHE_DIR.resolve()),
    }
