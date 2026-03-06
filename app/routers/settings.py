"""设置/维护 API"""

import asyncio
from collections import defaultdict

from fastapi import APIRouter, Depends, Request
from sqlmodel import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import PHOTOS_DIR, CACHE_DIR, SCAN_DUPLICATES_BATCH_SIZE
from app.models import Image, get_async_session
from app.services.scanner import run_full_scan
from app.utils.images import cache_filename
from app.services.scan_state import begin_scan, end_scan
from app.schemas import ScanDuplicatesRequest
from app.app_common import templates
from app.utils.path_utils import normalize_path, path_filter_for_prefix
from app.utils.hash_utils import compute_file_md5
from app.utils.stats import stats_folder_count_from_db
from app.utils.folder_tree import invalidate_folder_tree_cache

router = APIRouter(tags=["settings"])


def _estimate_thumb_bytes(width: int, height: int) -> int:
    """根据原图尺寸估算 300px WebP 缩略图字节数"""
    if width <= 0 or height <= 0:
        return 18_000  # 默认 300×200 约 18KB
    tw = min(300, width)
    th = int(height * tw / width) if width else 169
    pixels = tw * th
    return int(pixels * 0.3)  # 实测约 0.27 B/px，取 0.3


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
    from app.services.scan_state import is_scanning

    return {"scanning": is_scanning()}


@router.post("/scan")
async def trigger_scan():
    """手动触发扫描。复用 run_full_scan，一次 os.walk 完成 cleanup + scan。"""
    begin_scan()
    try:
        result = await run_full_scan(PHOTOS_DIR, CACHE_DIR)
        n_img = result.get("images_added", 0)
        n_vid = result.get("videos_added", 0)
        return {"scanned": n_img + n_vid, "images": n_img, "videos": n_vid}
    finally:
        end_scan()
        invalidate_folder_tree_cache()


@router.post("/api/cleanup")
async def trigger_cleanup():
    """手动触发数据库清理同步。复用 run_full_scan，一次 os.walk 完成 cleanup + scan。"""
    begin_scan()
    try:
        result = await run_full_scan(PHOTOS_DIR, CACHE_DIR)
        return {
            "stale_removed": result.get("stale_removed", 0),
            "orphan_cache_removed": result.get("orphan_cache_removed", 0),
            "cache_regenerated": result.get("cache_regenerated", 0),
        }
    finally:
        end_scan()
        invalidate_folder_tree_cache()


@router.post("/api/full-sync")
async def trigger_full_sync():
    """完整同步：一次 os.walk 完成 cleanup + scan，供「完整重建」等场景使用。"""
    begin_scan()
    try:
        return await run_full_scan(PHOTOS_DIR, CACHE_DIR)
    finally:
        end_scan()
        invalidate_folder_tree_cache()


@router.post("/api/scan-duplicates")
async def scan_duplicates(
    body: ScanDuplicatesRequest | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """扫描重复文件（分批加载，支持百万级；仅对同 size 候选组计算 MD5）"""
    folder_path = normalize_path(
        (body.folder_path if body else None) or "", allow_empty=True
    )
    base_stmt = select(
        Image.id,
        Image.relative_path,
        Image.filename,
        Image.file_size,
        Image.modified_at,
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
            .limit(SCAN_DUPLICATES_BATCH_SIZE)
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            break
        for row in rows:
            img_id, rel_path, filename, file_size, modified_at = row
            last_id = img_id or last_id
            by_size[file_size or 0].append(
                {
                    "id": img_id,
                    "relative_path": rel_path,
                    "filename": filename,
                    "file_size": file_size,
                    "modified_at": modified_at,
                    "cache_key": cache_filename(rel_path),
                }
            )
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
            groups.append(
                {
                    "content_hash": content_hash,
                    "file_size": items[0]["file_size"],
                    "items": items,
                }
            )
    return {"groups": groups}


def _stats_cache_realtime_sync() -> tuple[int, int]:
    """同步执行 rglob 统计 cache 目录（按需调用，避免阻塞）"""
    count = 0
    total_size = 0
    for p in CACHE_DIR.resolve().rglob("*.webp"):
        try:
            total_size += p.stat().st_size
        except OSError:
            pass
        count += 1
    return count, total_size


@router.post("/api/stats-cache-realtime")
async def get_cache_stats_realtime():
    """按需执行 rglob 获取 cache 精确统计，用于手动刷新（孤儿/缺失缓存时与 DB 估算可能不一致）"""
    count, total_size = await asyncio.to_thread(_stats_cache_realtime_sync)
    return {"cache_count": count, "cache_size": total_size}


@router.get("/api/stats")
async def get_stats(session: AsyncSession = Depends(get_async_session)):
    """获取数据库和文件系统统计信息（优先从 DB 统计，支持百万级）"""
    image_count = (
        await session.execute(
            select(func.count(Image.id)).where(Image.media_type == "image")
        )
    ).scalar() or 0
    video_count = (
        await session.execute(
            select(func.count(Image.id)).where(Image.media_type == "video")
        )
    ).scalar() or 0
    total_size_raw = (
        await session.execute(select(func.sum(Image.file_size)))
    ).scalar() or 0
    total_size = int(total_size_raw) if total_size_raw else 0
    folder_count = await stats_folder_count_from_db(session)
    cache_count = image_count + video_count
    result = await session.execute(select(Image.width, Image.height))
    rows = result.fetchall()
    cache_size = sum(_estimate_thumb_bytes(w or 0, h or 0) for w, h in rows)
    return {
        "image_count": image_count,
        "video_count": video_count,
        "total_files": image_count + video_count,
        "total_size": total_size,
        "folder_count": folder_count,
        "cache_count": cache_count,
        "cache_size": cache_size,
        "cache_size_estimated": True,
        "photos_dir": str(PHOTOS_DIR.resolve()),
        "cache_dir": str(CACHE_DIR.resolve()),
    }
