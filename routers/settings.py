"""设置/维护 API"""
import asyncio
from collections import defaultdict

from fastapi import APIRouter, Depends, Request
from sqlmodel import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR
from models import Image, get_async_session
from scanner import scan_photos, scan_videos, cleanup_database
from utils.images import cache_filename
from scan_state import begin_scan, end_scan
from schemas import ScanDuplicatesRequest
from app_common import templates
from utils.path_utils import normalize_path, path_filter_for_prefix
from utils.hash_utils import compute_file_md5
from utils.stats import stats_folder_count_from_db, stats_cache_only

router = APIRouter(tags=["settings"])


@router.get("/settings")
async def settings_page(request: Request):
    """设置页面"""
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "db_display": "MariaDB"},
    )


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
    begin_scan()
    try:
        result = await cleanup_database(PHOTOS_DIR, CACHE_DIR)
        return result
    finally:
        end_scan()


_SCAN_DUPLICATES_BATCH_SIZE = 5000


@router.post("/api/scan-duplicates")
async def scan_duplicates(
    body: ScanDuplicatesRequest | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """扫描重复文件（分批加载，支持百万级；仅对同 size 候选组计算 MD5）"""
    folder_path = normalize_path((body.folder_path if body else None) or "", allow_empty=True)
    base_stmt = select(
        Image.id, Image.relative_path, Image.filename, Image.file_size, Image.modified_at
    )
    if folder_path:
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        base_stmt = base_stmt.where(pf)
    photos_dir = PHOTOS_DIR.resolve()
    by_size: dict[int, list[dict]] = defaultdict(list)
    last_id = 0
    while True:
        stmt = (
            base_stmt.where(Image.id > last_id)
            .order_by(Image.id)
            .limit(_SCAN_DUPLICATES_BATCH_SIZE)
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            break
        for row in rows:
            img_id, rel_path, filename, file_size, modified_at = row
            last_id = img_id or last_id
            by_size[file_size or 0].append({
                "id": img_id,
                "relative_path": rel_path,
                "filename": filename,
                "file_size": file_size,
                "modified_at": modified_at,
                "cache_key": cache_filename(rel_path),
            })
        await asyncio.sleep(0)
    candidate_groups = [g for g in by_size.values() if len(g) > 1]
    if not candidate_groups:
        return {"groups": []}
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for group in candidate_groups:
        for item in group:
            h = await asyncio.to_thread(
                compute_file_md5, photos_dir, item["relative_path"]
            )
            if h is None:
                continue
            by_hash[h].append(item)
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
    """获取数据库和文件系统统计信息（优先从 DB 统计，支持百万级）"""
    image_count = (
        await session.execute(select(func.count(Image.id)).where(Image.media_type == "image"))
    ).scalar() or 0
    video_count = (
        await session.execute(select(func.count(Image.id)).where(Image.media_type == "video"))
    ).scalar() or 0
    total_size = (await session.execute(select(func.sum(Image.file_size)))).scalar() or 0
    folder_count = await stats_folder_count_from_db(session)
    cache_count = image_count + video_count  # 与 DB 一致，避免百万级 rglob
    cache_size = 0
    cache_count_fs, cache_size = await asyncio.to_thread(
        stats_cache_only, CACHE_DIR
    )
    if cache_count_fs != cache_count:
        cache_count = cache_count_fs  # 有孤儿/缺失时以文件系统为准
    return {
        "image_count": image_count,
        "video_count": video_count,
        "total_files": image_count + video_count,
        "total_size": total_size,
        "folder_count": folder_count,
        "cache_count": cache_count,
        "cache_size": cache_size,
        "photos_dir": str(PHOTOS_DIR.resolve()),
        "cache_dir": str(CACHE_DIR.resolve()),
    }
