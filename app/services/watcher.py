"""
文件系统监听器：监控 photos 目录的变化，自动同步数据库。

设计要点：
- watchdog Observer 在独立线程中运行，收集文件事件到队列
- asyncio 定时任务定期消费队列，批量处理变化（去抖动）
- 新增图片 → 生成缩略图 + 入库
- 删除图片 → 清除数据库记录 + 缓存
- 移动/重命名 → 更新路径
"""

import asyncio
import logging
import time
from pathlib import Path
from queue import Empty, Queue

from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from watchdog.events import (
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from app.models import Image, async_session_factory, natural_sort_key
from app.services.scan_state import begin_scan, end_scan, is_scanning
from app.services.scanner import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    get_media_metadata_and_thumbnail,
)
from app.utils.image_records import create_image_record
from app.utils.images import cache_filename
from app.utils.path_utils import relative_path
from app.utils.tags import DAMAGED_TAG_NAME, add_tag_to_image, ensure_tag_exists

logger = logging.getLogger(__name__)

# 去抖动间隔：收集事件后等待这么久再处理（秒）
DEBOUNCE_SECONDS = 3.0
# 轮询间隔
POLL_INTERVAL = 2.0


def _is_media(path: str) -> bool:
    """判断文件路径是否为支持的图片或视频格式"""
    return Path(path).suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)


def _is_video(path: str) -> bool:
    """判断文件路径是否为视频格式"""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


class _PhotoEventHandler(FileSystemEventHandler):
    """收集 photos 目录中的图片和视频文件事件"""

    def __init__(self, queue: Queue):
        super().__init__()
        self._queue = queue

    def on_created(self, event):
        if not event.is_directory and _is_media(event.src_path):
            self._queue.put(("created", event.src_path, None, time.monotonic()))

    def on_deleted(self, event):
        if not event.is_directory and _is_media(event.src_path):
            self._queue.put(("deleted", event.src_path, None, time.monotonic()))

    def on_moved(self, event):
        if not event.is_directory:
            src_media = _is_media(event.src_path)
            dst_media = _is_media(event.dest_path)
            if src_media or dst_media:
                self._queue.put(("moved", event.src_path, event.dest_path, time.monotonic()))


async def _process_created(photos_dir: Path, cache_dir: Path, full_path: Path):
    """处理新增图片或视频：生成缩略图 + 写入数据库"""
    if not full_path.exists() or not full_path.is_file():
        return

    rel_path = relative_path(photos_dir, full_path)

    async with async_session_factory() as session:
        result = await session.execute(select(Image).where(Image.relative_path == rel_path))
        if result.scalar_one_or_none():
            return

        try:
            is_vid = _is_video(str(full_path))
            cache_name = cache_filename(rel_path)
            cache_path = cache_dir / cache_name

            # 在线程中执行 PIL/ffmpeg 等阻塞操作
            data = await asyncio.to_thread(
                get_media_metadata_and_thumbnail,
                full_path,
                cache_path,
                is_vid,
            )
            if data is None:
                return

            width, height, modified_at, file_size, is_corrupted = data
            media_type = "video" if is_vid else "image"

            damaged_tag = await ensure_tag_exists(session, DAMAGED_TAG_NAME) if is_corrupted else None

            record = create_image_record(
                filename=full_path.name,
                relative_path=rel_path,
                modified_at=modified_at,
                file_size=file_size,
                width=width,
                height=height,
                media_type=media_type,
            )
            session.add(record)
            await session.flush()
            if is_corrupted and damaged_tag:
                await add_tag_to_image(session, record.id, damaged_tag)
            await session.commit()
            print(f"[watcher] 新增: {rel_path}", flush=True)
        except IntegrityError as e:
            # 竞态：scanner 已先入库，静默忽略
            logger.debug(
                "IntegrityError on add %s (scanner may have inserted first): %s",
                rel_path,
                e,
            )
        except Exception as e:
            print(f"[watcher] 处理新增失败 {rel_path}: {e}", flush=True)


async def _process_deleted(photos_dir: Path, cache_dir: Path, full_path: Path):
    """处理删除图片：移除数据库记录 + 缓存"""
    rel_path = relative_path(photos_dir, full_path)

    async with async_session_factory() as session:
        result = await session.execute(select(Image).where(Image.relative_path == rel_path))
        img = result.scalar_one_or_none()
        if not img:
            return

        # 删除缓存
        cache_name = cache_filename(rel_path)
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)

        await session.delete(img)
        await session.commit()
        print(f"[watcher] 删除: {rel_path}", flush=True)


async def _process_moved(photos_dir: Path, cache_dir: Path, src_path: Path, dst_path: Path):
    """处理移动/重命名：更新数据库记录路径 + 缓存"""
    src_rel = relative_path(photos_dir, src_path)
    dst_rel = relative_path(photos_dir, dst_path)

    async with async_session_factory() as session:
        result = await session.execute(select(Image).where(Image.relative_path == src_rel))
        img = result.scalar_one_or_none()

        if img:
            # 删除旧缓存
            old_cache = cache_dir / cache_filename(src_rel)
            if old_cache.exists():
                old_cache.unlink(missing_ok=True)

            # 若 dst_rel 已有记录（文件被覆盖），先删除旧记录
            dst_result = await session.execute(select(Image).where(Image.relative_path == dst_rel))
            dst_img = dst_result.scalar_one_or_none()
            if dst_img is not None and dst_img.id != img.id:
                await session.delete(dst_img)

            # 更新记录
            img.relative_path = dst_rel
            img.filename = dst_path.name
            img.filename_natural = natural_sort_key(dst_path.name)
            img.relative_path_natural = natural_sort_key(dst_rel)
            if dst_path.exists():
                new_cache = cache_dir / cache_filename(dst_rel)
                is_video = getattr(img, "media_type", "image") == "video"
                data = await asyncio.to_thread(
                    get_media_metadata_and_thumbnail,
                    dst_path,
                    new_cache,
                    is_video,
                )
                if data:
                    _, _, img.modified_at, img.file_size, is_corrupted = data
                    if is_corrupted:
                        damaged_tag = await ensure_tag_exists(session, DAMAGED_TAG_NAME)
                        if damaged_tag:
                            await add_tag_to_image(session, img.id, damaged_tag)

            try:
                session.add(img)
                await session.commit()
                print(f"[watcher] 移动: {src_rel} → {dst_rel}", flush=True)
            except IntegrityError as e:
                # 竞态：dst_rel 刚被其他操作占用，静默忽略
                logger.debug("IntegrityError on move %s -> %s: %s", src_rel, dst_rel, e)
                await session.rollback()
        else:
            # 源记录不存在，当作新增处理
            if dst_path.exists() and _is_media(str(dst_path)):
                await _process_created(photos_dir, cache_dir, dst_path)


async def _drain_queue(queue: Queue, photos_dir: Path, cache_dir: Path):
    """消费事件队列，去抖动后批量处理"""
    # 收集所有积压事件
    events = []
    try:
        while True:
            events.append(queue.get_nowait())
    except Empty:
        pass

    if not events:
        return

    # 按路径去重：对同一路径只保留最后一个事件
    # 先过滤掉太新的事件（还在去抖动窗口内）
    now = time.monotonic()
    ready = []
    deferred = []
    for ev in events:
        event_type, src, dst, ts = ev
        if now - ts >= DEBOUNCE_SECONDS:
            ready.append(ev)
        else:
            deferred.append(ev)

    # 把未到期的放回队列
    for ev in deferred:
        queue.put(ev)

    if not ready:
        return

    # 按路径去重，保留最后事件
    # moved 事件使用 (src, dst) 作为键，避免被同路径的 created/deleted 覆盖
    path_events: dict[str | tuple, tuple] = {}
    for ev in ready:
        event_type, src, dst, ts = ev
        if event_type == "moved":
            key = ("moved", src, dst)
        else:
            key = (event_type, src)
        path_events[key] = ev

    # 若有扫描/清理在进行，暂不处理，将事件放回队列
    if is_scanning():
        for ev in path_events.values():
            queue.put(ev)
        return

    begin_scan()
    try:
        processed = 0
        for key, ev in path_events.items():
            event_type, src, dst, ts = ev
            try:
                if event_type == "created":
                    await _process_created(photos_dir, cache_dir, Path(src))
                elif event_type == "deleted":
                    await _process_deleted(photos_dir, cache_dir, Path(src))
                elif event_type == "moved":
                    await _process_moved(photos_dir, cache_dir, Path(src), Path(dst))
                processed += 1
            except Exception as e:
                print(f"[watcher] 处理事件失败 ({event_type} {src}): {e}", flush=True)

        if processed:
            print(f"[watcher] 批量处理 {processed} 个文件变化", flush=True)
    finally:
        end_scan()
        if processed > 0:
            from app.utils.folder_tree import invalidate_folder_tree_cache

            invalidate_folder_tree_cache()


def start_watcher(photos_dir: Path, cache_dir: Path, loop: asyncio.AbstractEventLoop) -> Observer:
    """
    启动文件系统监听器。

    参数：
        photos_dir: 照片目录
        cache_dir: 缓存目录
        loop: 主事件循环（用于调度异步任务）

    返回：
        Observer 实例（可在关闭时调用 .stop()）
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()

    event_queue: Queue = Queue()
    handler = _PhotoEventHandler(event_queue)

    observer = Observer()
    observer.schedule(handler, str(photos_dir), recursive=True)
    observer.daemon = True
    observer.start()

    print(f"[watcher] 开始监听: {photos_dir}", flush=True)

    async def _poll_loop():
        """在主事件循环中定期处理文件变化"""
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                await _drain_queue(event_queue, photos_dir, cache_dir)
            except Exception as e:
                print(f"[watcher] 轮询异常: {e}", flush=True)

    # 在主事件循环中启动轮询协程
    asyncio.run_coroutine_threadsafe(_poll_loop(), loop)

    return observer
