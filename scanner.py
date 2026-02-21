import asyncio
import hashlib
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image as PILImage
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image, async_session_factory, natural_sort_key

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}
THUMBNAIL_WIDTH = 300
# 多进程缩略图：并行度与批大小
_MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)
_PROCESS_BATCH_SIZE = min(16, _MAX_WORKERS * 2)


def _cache_filename(relative_path: str) -> str:
    """根据相对路径生成缩略图缓存文件名"""
    return hashlib.md5(relative_path.encode()).hexdigest() + ".webp"


def _relative_path(photos_dir: Path, full_path: Path) -> str:
    """计算相对路径，统一使用 / 分隔"""
    rel = full_path.relative_to(photos_dir)
    return str(rel).replace("\\", "/")


def _generate_thumbnail(full_path: Path, cache_path: Path) -> bool:
    """为指定图片生成缩略图，返回是否成功"""
    try:
        with PILImage.open(full_path) as img:
            img.load()
            if img.width > THUMBNAIL_WIDTH:
                ratio = THUMBNAIL_WIDTH / img.width
                new_size = (THUMBNAIL_WIDTH, int(img.height * ratio))
                thumb = img.resize(new_size, PILImage.Resampling.LANCZOS)
            else:
                thumb = img.copy()
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            thumb.save(cache_path, "WEBP", quality=85)
        return True
    except Exception as e:
        print(f"[cache] 生成缩略图失败 {full_path}: {e}", flush=True)
        return False


def get_media_metadata_and_thumbnail(
    full_path: Path, cache_path: Path, is_video: bool
) -> tuple[int, int, float, int] | None:
    """同步获取媒体元数据并生成缩略图，返回 (width, height, modified_at, file_size)，失败返回 None。
    供 watcher、上传等场景复用。"""
    try:
        modified_at = os.path.getmtime(full_path)
        file_size = os.path.getsize(full_path)
        if is_video:
            width, height = _get_video_dimensions(full_path)
            _generate_video_thumbnail(full_path, cache_path)
        else:
            with PILImage.open(full_path) as img:
                width, height = img.size
            _generate_thumbnail(full_path, cache_path)
        return (width, height, modified_at, file_size)
    except Exception as e:
        print(f"[scanner] 处理失败 {full_path}: {e}", flush=True)
        return None


def _get_video_dimensions(full_path: Path) -> tuple[int, int]:
    """使用 ffprobe 获取视频宽高，失败时返回 (1920, 1080)"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(full_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"[cache] ffprobe 失败 {full_path}: {e}", flush=True)
    return 1920, 1080


def _generate_video_thumbnail(full_path: Path, cache_path: Path) -> bool:
    """使用 ffmpeg 从视频第一帧提取缩略图并转为 WebP"""
    try:
        tmp_jpg = cache_path.with_suffix(".tmp.jpg")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(full_path),
                "-vframes", "1",
                "-q:v", "2",
                str(tmp_jpg),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or not tmp_jpg.exists():
            raise RuntimeError("ffmpeg 提取帧失败")
        try:
            with PILImage.open(tmp_jpg) as img:
                img.load()
                if img.width > THUMBNAIL_WIDTH:
                    ratio = THUMBNAIL_WIDTH / img.width
                    new_size = (THUMBNAIL_WIDTH, int(img.height * ratio))
                    thumb = img.resize(new_size, PILImage.Resampling.LANCZOS)
                else:
                    thumb = img.copy()
                if thumb.mode in ("RGBA", "P"):
                    thumb = thumb.convert("RGB")
                thumb.save(cache_path, "WEBP", quality=85)
        finally:
            tmp_jpg.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"[cache] 视频缩略图失败 {full_path}: {e}", flush=True)
        # 无 ffmpeg 时生成灰色占位图
        try:
            placeholder = PILImage.new("RGB", (THUMBNAIL_WIDTH, 169), (80, 80, 80))
            placeholder.save(cache_path, "WEBP", quality=85)
            return True
        except Exception:
            return False


def _collect_image_files(photos_dir: Path) -> list[Path]:
    """在线程中收集所有图片文件路径"""
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(photos_dir.rglob(f"*{ext}"))
    return files


def _collect_video_files(photos_dir: Path) -> list[Path]:
    """在线程中收集所有视频文件路径"""
    files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(photos_dir.rglob(f"*{ext}"))
    return files


def _process_single_image_sync(
    full_path: Path, photos_dir: Path, cache_dir: Path
) -> tuple[str, str, float, int, int, int] | None:
    """同步处理单张图片：读取尺寸、生成缩略图，返回入库所需数据，失败返回 None"""
    try:
        rel_path = _relative_path(photos_dir, full_path)
        modified_at = os.path.getmtime(full_path)
        file_size = os.path.getsize(full_path)
        with PILImage.open(full_path) as img:
            width, height = img.size
            img.load()
            if img.width > THUMBNAIL_WIDTH:
                ratio = THUMBNAIL_WIDTH / img.width
                new_size = (THUMBNAIL_WIDTH, int(img.height * ratio))
                thumb = img.resize(new_size, PILImage.Resampling.LANCZOS)
            else:
                thumb = img.copy()
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            cache_name = _cache_filename(rel_path)
            cache_path = cache_dir / cache_name
            thumb.save(cache_path, "WEBP", quality=85)
        return (full_path.name, rel_path, modified_at, file_size, width, height)
    except Exception as e:
        print(f"[scan] 处理失败 {full_path}: {e}", flush=True)
        return None


def _process_single_video_sync(
    full_path: Path, photos_dir: Path, cache_dir: Path
) -> tuple[str, str, float, int, int, int] | None:
    """同步处理单个视频：获取尺寸、生成缩略图，返回入库所需数据，失败返回 None"""
    try:
        rel_path = _relative_path(photos_dir, full_path)
        modified_at = os.path.getmtime(full_path)
        file_size = os.path.getsize(full_path)
        width, height = _get_video_dimensions(full_path)
        cache_name = _cache_filename(rel_path)
        cache_path = cache_dir / cache_name
        if not _generate_video_thumbnail(full_path, cache_path):
            return None
        return (full_path.name, rel_path, modified_at, file_size, width, height)
    except Exception as e:
        print(f"[scan] 视频处理失败 {full_path}: {e}", flush=True)
        return None


async def scan_photos(photos_dir: Path, cache_dir: Path) -> int:
    """
    异步扫描 photos 目录，生成缩略图并写入数据库。
    返回新扫描的图片数量。
    使用 ProcessPoolExecutor 多进程并行生成缩略图，充分利用多核。
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()
    count = 0

    print(f"[scan] 开始扫描: {photos_dir}", flush=True)

    DB_BATCH_SIZE = 50  # 每 50 张提交一次，边扫边可见

    async with async_session_factory() as session:
        # 在线程中收集所有图片文件，避免阻塞事件循环
        image_files = await asyncio.to_thread(_collect_image_files, photos_dir)
        total_files = len(image_files)
        print(f"[scan] 发现 {total_files} 个图片文件", flush=True)

        pending: list[Path] = []
        batch_count = 0
        loop = asyncio.get_running_loop()

        async def _process_batch(paths: list[Path]) -> list[tuple[str, str, float, int, int, int]]:
            """多进程处理一批图片，返回成功的结果列表"""
            if not paths:
                return []
            with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                tasks = [
                    loop.run_in_executor(
                        executor, _process_single_image_sync, fp, photos_dir, cache_dir
                    )
                    for fp in paths
                ]
                raw_results = await asyncio.gather(*tasks)
            return [r for r in raw_results if r is not None]

        for full_path in image_files:
            if not full_path.is_file():
                continue

            rel_path = _relative_path(photos_dir, full_path)

            # 检查是否已存在
            result = await session.execute(
                select(Image).where(Image.relative_path == rel_path)
            )
            if result.scalar_one_or_none():
                continue

            pending.append(full_path)

            # 攒够一批则多进程处理
            if len(pending) >= _PROCESS_BATCH_SIZE:
                results = await _process_batch(pending)
                pending = []
                for data in results:
                    filename, rel_path, modified_at, file_size, width, height = data
                    record = Image(
                        filename=filename,
                        relative_path=rel_path,
                        modified_at=modified_at,
                        file_size=file_size,
                        width=width,
                        height=height,
                        filename_natural=natural_sort_key(filename),
                        relative_path_natural=natural_sort_key(rel_path),
                        media_type="image",
                    )
                    session.add(record)
                    count += 1
                    batch_count += 1

                if batch_count >= DB_BATCH_SIZE:
                    await session.commit()
                    print(f"[scan] 进度: {count}/{total_files}", flush=True)
                    batch_count = 0

                await asyncio.sleep(0)  # 让出事件循环

        # 处理剩余 pending
        if pending:
            results = await _process_batch(pending)
            for data in results:
                filename, rel_path, modified_at, file_size, width, height = data
                record = Image(
                    filename=filename,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
                    filename_natural=natural_sort_key(filename),
                    relative_path_natural=natural_sort_key(rel_path),
                    media_type="image",
                )
                session.add(record)
                count += 1
                batch_count += 1

        # 提交剩余
        if batch_count > 0:
            await session.commit()
        print(f"[scan] 图片扫描完成，新增 {count} 条记录", flush=True)

    return count


async def scan_videos(photos_dir: Path, cache_dir: Path) -> int:
    """
    异步扫描 photos 目录中的视频文件，生成缩略图并写入数据库。
    返回新扫描的视频数量。
    使用 ProcessPoolExecutor 多进程并行处理视频（ffprobe/ffmpeg），充分利用多核。
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()
    count = 0

    # 在线程中收集视频文件
    video_files = await asyncio.to_thread(_collect_video_files, photos_dir)
    if not video_files:
        return 0

    print(f"[scan] 发现 {len(video_files)} 个视频文件", flush=True)
    DB_BATCH_SIZE = 20
    # 视频处理更耗时，批大小略小
    _video_batch_size = min(8, _MAX_WORKERS)

    async with async_session_factory() as session:
        pending: list[Path] = []
        batch_count = 0
        loop = asyncio.get_running_loop()

        async def _process_video_batch(paths: list[Path]) -> list[tuple[str, str, float, int, int, int]]:
            """多进程处理一批视频"""
            if not paths:
                return []
            with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                tasks = [
                    loop.run_in_executor(
                        executor, _process_single_video_sync, fp, photos_dir, cache_dir
                    )
                    for fp in paths
                ]
                raw_results = await asyncio.gather(*tasks)
            return [r for r in raw_results if r is not None]

        for full_path in video_files:
            if not full_path.is_file():
                continue

            rel_path = _relative_path(photos_dir, full_path)

            result = await session.execute(
                select(Image).where(Image.relative_path == rel_path)
            )
            if result.scalar_one_or_none():
                continue

            pending.append(full_path)

            if len(pending) >= _video_batch_size:
                results = await _process_video_batch(pending)
                pending = []
                for data in results:
                    filename, rel_path, modified_at, file_size, width, height = data
                    record = Image(
                        filename=filename,
                        relative_path=rel_path,
                        modified_at=modified_at,
                        file_size=file_size,
                        width=width,
                        height=height,
                        filename_natural=natural_sort_key(filename),
                        relative_path_natural=natural_sort_key(rel_path),
                        media_type="video",
                    )
                    session.add(record)
                    count += 1
                    batch_count += 1

                if batch_count >= DB_BATCH_SIZE:
                    await session.commit()
                    print(f"[scan] 视频进度: {count}/{len(video_files)}", flush=True)
                    batch_count = 0

                await asyncio.sleep(0)

        if pending:
            results = await _process_video_batch(pending)
            for data in results:
                filename, rel_path, modified_at, file_size, width, height = data
                record = Image(
                    filename=filename,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
                    filename_natural=natural_sort_key(filename),
                    relative_path_natural=natural_sort_key(rel_path),
                    media_type="video",
                )
                session.add(record)
                count += 1
                batch_count += 1

        if batch_count > 0:
            await session.commit()
        if count:
            print(f"[scan] 视频扫描完成，新增 {count} 条记录", flush=True)

    return count


async def cleanup_database(photos_dir: Path, cache_dir: Path) -> dict:
    """
    数据库清理同步，处理三种不一致：
    1. 幽灵记录：原图已被外部删除 → 移除数据库记录 + 对应缓存
    2. 孤儿缓存：cache 目录中多余的 .webp 文件 → 删除
    3. 缺失缓存：数据库有记录但缩略图丢失 → 重新生成

    返回 {"stale_removed": int, "orphan_cache_removed": int, "cache_regenerated": int}
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()

    stale_removed = 0
    orphan_cache_removed = 0
    cache_regenerated = 0

    print("[cleanup] 开始数据库清理...", flush=True)

    # ── 第 1 步：清除幽灵记录（原图不存在的数据库记录） ──
    async with async_session_factory() as session:
        result = await session.execute(select(Image))
        all_images = list(result.scalars().all())
        print(f"[cleanup] 数据库共 {len(all_images)} 条记录，开始检查原图是否存在...", flush=True)

        # 收集有效记录的 cache 文件名，供第 2 步使用
        valid_cache_names: set[str] = set()

        batch_count = 0
        checked_since_yield = 0
        for img in all_images:
            photo_path = photos_dir / img.relative_path
            if not photo_path.exists():
                # 原图已不存在，删除数据库记录和缓存
                cache_name = _cache_filename(img.relative_path)
                cache_path = cache_dir / cache_name
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
                await session.delete(img)
                stale_removed += 1
                batch_count += 1
                if batch_count >= 100:
                    await session.commit()
                    batch_count = 0
            else:
                valid_cache_names.add(_cache_filename(img.relative_path))
            checked_since_yield += 1
            if checked_since_yield >= 100:
                await asyncio.sleep(0)
                checked_since_yield = 0

        if batch_count > 0:
            await session.commit()

        if stale_removed:
            print(f"[cleanup] 清除 {stale_removed} 条幽灵记录（原图已删除）", flush=True)

    # ── 第 2 步：清除孤儿缓存文件（iterdir 在线程中执行） ──
    def _list_cache_webp(cache_dir: Path) -> list[Path]:
        if not cache_dir.exists():
            return []
        return [f for f in cache_dir.iterdir() if f.suffix == ".webp"]

    if cache_dir.exists():
        cache_files = await asyncio.to_thread(_list_cache_webp, cache_dir)
        for cache_file in cache_files:
            if cache_file.name not in valid_cache_names:
                cache_file.unlink(missing_ok=True)
                orphan_cache_removed += 1

        if orphan_cache_removed:
            print(f"[cleanup] 清除 {orphan_cache_removed} 个孤儿缓存文件", flush=True)

    # ── 第 3 步：补全缺失的缩略图缓存（多进程并行生成） ──
    async with async_session_factory() as session:
        result = await session.execute(select(Image))
        all_images = list(result.scalars().all())

        def _regenerate_one(args: tuple[Path, Path, bool]) -> bool:
            photo_path, cache_path, is_video = args
            if is_video:
                return _generate_video_thumbnail(photo_path, cache_path)
            return _generate_thumbnail(photo_path, cache_path)

        to_regen: list[tuple[Path, Path, bool]] = []
        for img in all_images:
            cache_name = _cache_filename(img.relative_path)
            cache_path = cache_dir / cache_name
            if not cache_path.exists():
                photo_path = photos_dir / img.relative_path
                if photo_path.exists():
                    is_video = getattr(img, "media_type", "image") == "video"
                    to_regen.append((photo_path, cache_path, is_video))

        if to_regen:
            loop = asyncio.get_running_loop()
            batch_size = min(_PROCESS_BATCH_SIZE, len(to_regen))
            with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                for i in range(0, len(to_regen), batch_size):
                    batch = to_regen[i : i + batch_size]
                    tasks = [
                        loop.run_in_executor(executor, _regenerate_one, item)
                        for item in batch
                    ]
                    results = await asyncio.gather(*tasks)
                    cache_regenerated += sum(1 for ok in results if ok)
                    await asyncio.sleep(0)

            if cache_regenerated:
                print(f"[cleanup] 重新生成 {cache_regenerated} 个缺失缓存", flush=True)

    summary = {
        "stale_removed": stale_removed,
        "orphan_cache_removed": orphan_cache_removed,
        "cache_regenerated": cache_regenerated,
    }
    print(f"[cleanup] 清理完成: {summary}", flush=True)
    return summary
