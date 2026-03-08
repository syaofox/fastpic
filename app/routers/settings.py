"""设置/维护 API"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.app_common import templates
from app.config import CACHE_DIR, PHOTOS_DIR, SCAN_DUPLICATES_BATCH_SIZE
from app.models import Image, get_async_session
from app.schemas import ScanDuplicatesRequest
from app.services import task_state
from app.services.scan_state import begin_scan, end_scan
from app.services.scanner import run_full_scan
from app.services.task_queue import QueueTask, TaskQueue
from app.utils.folder_tree import invalidate_folder_tree_cache
from app.utils.hash_utils import compute_file_md5
from app.utils.images import cache_filename
from app.utils.path_utils import normalize_path, path_filter_for_prefix
from app.utils.stats import stats_folder_count_from_db

router = APIRouter(tags=["settings"])

task_queue = TaskQueue()


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
        request,
        "settings.html",
        {"db_display": "MariaDB"},
    )


@router.get("/api/scan-status")
async def get_scan_status():
    """返回当前是否有扫描任务在进行"""
    from app.services.scan_state import is_scanning

    return {"scanning": is_scanning()}


@router.get("/api/task-status")
async def get_task_status():
    """返回当前进行中的任务状态，用于页面刷新后恢复 UI"""
    status = task_state.get_status()
    if status:
        return status
    last_result = task_state.get_last_result()
    if last_result:
        return last_result
    return {"task_type": None, "is_running": False}


@router.post("/api/task-status/clear")
async def clear_task_status():
    """清除任务状态"""
    task_state.clear()
    return {"ok": True}


@router.get("/api/task-events")
async def task_events(request: Request):
    """Server-Sent Events 实时推送任务进度"""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        queue = task_state.get_queue_for_sse()
        last_sent_status = None
        disconnected = False

        while True:
            if await request.is_disconnected():
                if not disconnected:
                    disconnected = True
                    yield "data: \n\n"
                break

            try:
                await asyncio.wait_for(queue.get(), timeout=25)

                status = task_state.get_status()
                if status:
                    last_sent_status = status
                    yield f"data: {status}\n\n"
                elif last_sent_status and last_sent_status.get("finished_at"):
                    yield f"data: {last_sent_status}\n\n"
                    last_sent_status = None
            except TimeoutError:
                yield "data: \n\n"
            except Exception:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/queue-status")
async def get_queue_status():
    """获取队列状态"""
    return task_queue.get_status()


class CancelTaskRequest(BaseModel):
    queue_id: str


@router.post("/api/queue-cancel")
async def cancel_queue_task(request: CancelTaskRequest):
    """取消队列中的任务"""
    success = task_queue.cancel_task(request.queue_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或已完成")
    return {"success": True}


async def _run_scan_task(task: QueueTask) -> dict:
    """扫描任务处理器"""
    begin_scan()
    try:
        result = await run_full_scan(PHOTOS_DIR, CACHE_DIR)
        n_img = result.get("images_added", 0)
        n_vid = result.get("videos_added", 0)
        return {"scanned": n_img + n_vid, "images": n_img, "videos": n_vid}
    finally:
        end_scan()
        invalidate_folder_tree_cache()


async def _run_cleanup_task(task: QueueTask) -> dict:
    """清理任务处理器"""
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


async def _run_full_sync_task(task: QueueTask) -> dict:
    """完整同步任务处理器"""
    begin_scan()
    try:
        return await run_full_scan(PHOTOS_DIR, CACHE_DIR)
    finally:
        end_scan()
        invalidate_folder_tree_cache()


task_queue.register_handler("scan", _run_scan_task)
task_queue.register_handler("cleanup", _run_cleanup_task)
task_queue.register_handler("full-sync", _run_full_sync_task)


@router.post("/scan")
async def trigger_scan():
    """手动触发扫描，任务进入队列后台执行"""
    queue_id = await task_queue.add_task("scan", priority=10)
    status = task_queue.get_status()
    running = status.get("scan", {}).get("running")
    pending = status.get("scan", {}).get("pending", [])
    position = 0
    for i, p in enumerate(pending):
        if p.get("queue_id") == queue_id:
            position = i
            break
    if running:
        return {"queue_id": queue_id, "status": "running", "position": 0}
    return {"queue_id": queue_id, "status": "queued", "position": position}


@router.post("/api/cleanup")
async def trigger_cleanup():
    """手动触发数据库清理同步，任务进入队列后台执行"""
    queue_id = await task_queue.add_task("cleanup", priority=10)
    status = task_queue.get_status()
    running = status.get("cleanup", {}).get("running")
    pending = status.get("cleanup", {}).get("pending", [])
    position = 0
    for i, p in enumerate(pending):
        if p.get("queue_id") == queue_id:
            position = i
            break
    if running:
        return {"queue_id": queue_id, "status": "running", "position": 0}
    return {"queue_id": queue_id, "status": "queued", "position": position}


@router.post("/api/full-sync")
async def trigger_full_sync():
    """完整同步：任务进入队列后台执行"""
    queue_id = await task_queue.add_task("full-sync", priority=10)
    status = task_queue.get_status()
    running = status.get("full-sync", {}).get("running")
    pending = status.get("full-sync", {}).get("pending", [])
    position = 0
    for i, p in enumerate(pending):
        if p.get("queue_id") == queue_id:
            position = i
            break
    if running:
        return {"queue_id": queue_id, "status": "running", "position": 0}
    return {"queue_id": queue_id, "status": "queued", "position": position}


async def _run_scan_duplicates_task(task: QueueTask) -> dict:
    """扫描重复文件任务处理器"""
    from collections import defaultdict

    from app.models import Image, async_session_factory

    body = task.params or {}
    folder_path = normalize_path(body.get("folder_path", ""), allow_empty=True)

    base_stmt = select(
        Image.id,
        Image.relative_path,
        Image.filename,
        Image.file_size,
        Image.modified_at,
        Image.md5_hash,
    )
    if folder_path:
        pf = path_filter_for_prefix(Image.relative_path, folder_path)
        base_stmt = base_stmt.where(pf)

    photos_dir = PHOTOS_DIR.resolve()
    by_size: dict[int, list[dict]] = defaultdict(list)

    async with async_session_factory() as session:
        last_id = 0
        while True:
            stmt = base_stmt.where(Image.id > last_id).order_by(Image.id).limit(SCAN_DUPLICATES_BATCH_SIZE)
            result = await session.execute(stmt)
            rows = result.fetchall()
            if not rows:
                break
            for row in rows:
                img_id, rel_path, filename, file_size, modified_at, md5_hash = row
                last_id = img_id or last_id
                by_size[file_size or 0].append(
                    {
                        "id": img_id,
                        "relative_path": rel_path,
                        "filename": filename,
                        "file_size": file_size,
                        "modified_at": modified_at,
                        "md5_hash": md5_hash,
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
            h = item.get("md5_hash")
            if h is None:
                h = await asyncio.to_thread(compute_file_md5, photos_dir, item["relative_path"])
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


task_queue.register_handler("scan-duplicates", _run_scan_duplicates_task)


@router.post("/api/scan-duplicates")
async def scan_duplicates(
    body: ScanDuplicatesRequest | None = None,
):
    """扫描重复文件，任务进入队列后台执行"""
    params = {"folder_path": body.folder_path} if body and body.folder_path else {}
    queue_id = await task_queue.add_task("scan-duplicates", params, priority=10)
    status = task_queue.get_status()
    running = status.get("scan-duplicates", {}).get("running")
    pending = status.get("scan-duplicates", {}).get("pending", [])
    position = 0
    for i, p in enumerate(pending):
        if p.get("queue_id") == queue_id:
            position = i
            break
    if running:
        return {"queue_id": queue_id, "status": "running", "position": 0}
    return {"queue_id": queue_id, "status": "queued", "position": position}


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
    image_count = (await session.execute(select(func.count(Image.id)).where(Image.media_type == "image"))).scalar() or 0
    video_count = (await session.execute(select(func.count(Image.id)).where(Image.media_type == "video"))).scalar() or 0
    total_size_raw = (await session.execute(select(func.sum(Image.file_size)))).scalar() or 0
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


@router.post("/api/fix-md5-hashes")
async def fix_md5_hashes(session: AsyncSession = Depends(get_async_session)):
    """补全数据库中缺失的 MD5 哈希（扫描已有文件计算）"""
    result = await session.execute(
        select(Image.id, Image.relative_path).where(Image.md5_hash == None)  # noqa: E711
    )
    rows = result.fetchall()
    if not rows:
        return {"updated": 0, "message": "无需补全"}

    updated = 0
    for image_id, rel_path in rows:
        md5 = await asyncio.to_thread(compute_file_md5, PHOTOS_DIR, rel_path)
        if md5:
            await session.execute(Image.__table__.update().where(Image.id == image_id).values(md5_hash=md5))
            updated += 1
        if updated % 100 == 0:
            await session.commit()

    await session.commit()
    return {"updated": updated, "message": f"已补全 {updated} 条记录"}
