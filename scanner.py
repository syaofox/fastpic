import asyncio
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image as PILImage, ImageFile
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image, async_session_factory
from utils.images import cache_filename
from utils.image_records import create_image_record
from utils.tags import DAMAGED_TAG_NAME, add_tag_to_image, ensure_tag_exists

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".ts"}
THUMBNAIL_WIDTH = 300
# 多进程缩略图：并行度与批大小
_MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)
_PROCESS_BATCH_SIZE = min(16, _MAX_WORKERS * 2)


def _relative_path(photos_dir: Path, full_path: Path) -> str:
    """计算相对路径，统一使用 / 分隔"""
    rel = full_path.relative_to(photos_dir)
    return str(rel).replace("\\", "/")


def _load_image_maybe_truncated(full_path: Path) -> tuple[PILImage.Image, bool]:
    """加载图片，若正常加载失败则尝试截断模式。返回 (img, is_corrupted)。"""
    old_val = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = PILImage.open(full_path)
        img.load()
        return (img, False)
    except OSError:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = PILImage.open(full_path)
            img.load()
            return (img, True)
        except Exception:
            raise
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old_val


def _generate_thumbnail(full_path: Path, cache_path: Path) -> bool:
    """为指定图片生成缩略图，返回是否成功"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        img, _ = _load_image_maybe_truncated(full_path)
        try:
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
        finally:
            img.close()
    except Exception as e:
        print(f"[cache] 生成缩略图失败 {full_path}: {e}", flush=True)
        return False


def get_media_metadata_and_thumbnail(
    full_path: Path, cache_path: Path, is_video: bool
) -> tuple[int, int, float, int, bool] | None:
    """同步获取媒体元数据并生成缩略图，返回 (width, height, modified_at, file_size, is_corrupted)，失败返回 None。
    供 watcher、上传等场景复用。视频的 is_corrupted 恒为 False。"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        modified_at = os.path.getmtime(full_path)
        file_size = os.path.getsize(full_path)
        if is_video:
            width, height = _get_video_dimensions(full_path)
            _generate_video_thumbnail(full_path, cache_path)
            return (width, height, modified_at, file_size, False)
        img, is_corrupted = _load_image_maybe_truncated(full_path)
        try:
            width, height = img.size
            if img.width > THUMBNAIL_WIDTH:
                ratio = THUMBNAIL_WIDTH / img.width
                new_size = (THUMBNAIL_WIDTH, int(img.height * ratio))
                thumb = img.resize(new_size, PILImage.Resampling.LANCZOS)
            else:
                thumb = img.copy()
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            thumb.save(cache_path, "WEBP", quality=85)
            return (width, height, modified_at, file_size, is_corrupted)
        finally:
            img.close()
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
            # ffprobe 可能输出 "width,height" 或 "width\nheight"（部分 .ts 等格式）
            raw = result.stdout.strip().replace(",", " ").replace("\n", " ")
            parts = [p for p in raw.split() if p.isdigit()]
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"[cache] ffprobe 失败 {full_path}: {e}", flush=True)
    return 1920, 1080


def _generate_video_thumbnail(full_path: Path, cache_path: Path) -> bool:
    """使用 ffmpeg 从视频第一帧提取缩略图并转为 WebP"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
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


def _collect_media_files(photos_dir: Path) -> tuple[list[Path], list[Path]]:
    """一次 os.walk 遍历收集图片和视频路径，避免多次 rglob 重复遍历"""
    images: list[Path] = []
    videos: list[Path] = []
    for root, dirs, files in os.walk(photos_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            p = Path(root) / f
            ext = p.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(p)
            elif ext in VIDEO_EXTENSIONS:
                videos.append(p)
    return images, videos


def _process_single_image_sync(
    full_path: Path, photos_dir: Path, cache_dir: Path
) -> tuple[str, str, float, int, int, int, bool] | None:
    """同步处理单张图片：读取尺寸、生成缩略图，返回 (filename, rel_path, modified_at, file_size, width, height, is_corrupted)，失败返回 None"""
    try:
        rel_path = _relative_path(photos_dir, full_path)
        modified_at = os.path.getmtime(full_path)
        file_size = os.path.getsize(full_path)
        img, is_corrupted = _load_image_maybe_truncated(full_path)
        try:
            width, height = img.size
            if img.width > THUMBNAIL_WIDTH:
                ratio = THUMBNAIL_WIDTH / img.width
                new_size = (THUMBNAIL_WIDTH, int(img.height * ratio))
                thumb = img.resize(new_size, PILImage.Resampling.LANCZOS)
            else:
                thumb = img.copy()
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            cache_name = cache_filename(rel_path)
            cache_path = cache_dir / cache_name
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            thumb.save(cache_path, "WEBP", quality=85)
            return (full_path.name, rel_path, modified_at, file_size, width, height, is_corrupted)
        finally:
            img.close()
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
        cache_name = cache_filename(rel_path)
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
        image_files, _ = await asyncio.to_thread(_collect_media_files, photos_dir)
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

        damaged_tag = await ensure_tag_exists(session, DAMAGED_TAG_NAME)

        _EXISTS_CHECK_BATCH = 500  # 批量检查是否已存在，减少 DB 往返
        i = 0
        while i < len(image_files):
            check_batch = [
                p for p in image_files[i : i + _EXISTS_CHECK_BATCH]
                if p.is_file()
            ]
            i += _EXISTS_CHECK_BATCH
            if not check_batch:
                continue
            rel_paths = [_relative_path(photos_dir, p) for p in check_batch]
            result = await session.execute(
                select(Image.relative_path).where(Image.relative_path.in_(rel_paths))
            )
            existing = {r[0] for r in result.fetchall()}
            for full_path in check_batch:
                rel_path = _relative_path(photos_dir, full_path)
                if rel_path in existing:
                    continue
                pending.append(full_path)

            # 攒够一批则多进程处理
            while len(pending) >= _PROCESS_BATCH_SIZE:
                batch_to_process = pending[: _PROCESS_BATCH_SIZE]
                pending = pending[_PROCESS_BATCH_SIZE :]
                results = await _process_batch(batch_to_process)
                for data in results:
                    filename, rel_path, modified_at, file_size, width, height, is_corrupted = data
                    record = create_image_record(
                        filename=filename,
                        relative_path=rel_path,
                        modified_at=modified_at,
                        file_size=file_size,
                        width=width,
                        height=height,
                        media_type="image",
                    )
                    session.add(record)
                    await session.flush()
                    if is_corrupted and damaged_tag:
                        await add_tag_to_image(session, record.id, damaged_tag)
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
                filename, rel_path, modified_at, file_size, width, height, is_corrupted = data
                record = create_image_record(
                    filename=filename,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
                    media_type="image",
                )
                session.add(record)
                await session.flush()
                if is_corrupted and damaged_tag:
                    await add_tag_to_image(session, record.id, damaged_tag)
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

    _, video_files = await asyncio.to_thread(_collect_media_files, photos_dir)
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

        _EXISTS_CHECK_BATCH = 500
        vi = 0
        while vi < len(video_files):
            check_batch = [
                p for p in video_files[vi : vi + _EXISTS_CHECK_BATCH]
                if p.is_file()
            ]
            vi += _EXISTS_CHECK_BATCH
            if not check_batch:
                continue
            rel_paths = [_relative_path(photos_dir, p) for p in check_batch]
            result = await session.execute(
                select(Image.relative_path).where(Image.relative_path.in_(rel_paths))
            )
            existing = {r[0] for r in result.fetchall()}
            for full_path in check_batch:
                rel_path = _relative_path(photos_dir, full_path)
                if rel_path in existing:
                    continue
                pending.append(full_path)

            while len(pending) >= _video_batch_size:
                batch_to_process = pending[: _video_batch_size]
                pending = pending[_video_batch_size :]
                results = await _process_video_batch(batch_to_process)
                for data in results:
                    filename, rel_path, modified_at, file_size, width, height = data
                    record = create_image_record(
                        filename=filename,
                        relative_path=rel_path,
                        modified_at=modified_at,
                        file_size=file_size,
                        width=width,
                        height=height,
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
                record = create_image_record(
                    filename=filename,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
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


def _regenerate_one(args: tuple[Path, Path, bool]) -> bool:
    """根据参数重新生成单个缩略图，供多进程调用（需为模块级函数以便 pickle）"""
    photo_path, cache_path, is_video = args
    if is_video:
        return _generate_video_thumbnail(photo_path, cache_path)
    return _generate_thumbnail(photo_path, cache_path)


_CLEANUP_BATCH_SIZE = 5000  # 分批处理，避免百万级全表加载 OOM


async def cleanup_database(photos_dir: Path, cache_dir: Path) -> dict:
    """
    数据库清理同步，处理三种不一致：
    1. 幽灵记录：原图已被外部删除 → 移除数据库记录 + 对应缓存
    2. 孤儿缓存：cache 目录中多余的 .webp 文件 → 删除
    3. 缺失缓存：数据库有记录但缩略图丢失 → 重新生成

    使用分批处理，支持百万级规模，避免全表加载 OOM。
    返回 {"stale_removed": int, "orphan_cache_removed": int, "cache_regenerated": int}
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()

    stale_removed = 0
    orphan_cache_removed = 0
    cache_regenerated = 0

    print("[cleanup] 开始数据库清理...", flush=True)

    # ── 第 1 步：清除幽灵记录（分批加载，避免 OOM） ──
    valid_cache_names: set[str] = set()
    async with async_session_factory() as session:
        last_id = 0
        total_checked = 0
        while True:
            stmt = (
                select(Image)
                .where(Image.id > last_id)
                .order_by(Image.id)
                .limit(_CLEANUP_BATCH_SIZE)
            )
            result = await session.execute(stmt)
            batch = list(result.scalars().all())
            if not batch:
                break
            total_checked += len(batch)
            if total_checked == len(batch):
                print(f"[cleanup] 数据库共约 {len(batch)}+ 条记录，分批检查原图是否存在...", flush=True)

            batch_count = 0
            for img in batch:
                photo_path = photos_dir / img.relative_path
                if not photo_path.exists():
                    cache_name = cache_filename(img.relative_path)
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
                    valid_cache_names.add(cache_filename(img.relative_path))
                last_id = img.id or last_id

            if batch_count > 0:
                await session.commit()
            await asyncio.sleep(0)

        if stale_removed:
            print(f"[cleanup] 清除 {stale_removed} 条幽灵记录（原图已删除）", flush=True)

    # ── 第 2 步：清除孤儿缓存文件（线程中 rglob 流式处理，不加载全量到内存） ──
    def _remove_orphan_cache(cache_dir: Path, valid: set[str]) -> int:
        """在线程中执行，rglob 迭代器逐个处理，不加载全量"""
        count = 0
        if not cache_dir.exists():
            return 0
        for cache_file in cache_dir.rglob("*.webp"):
            rel = str(cache_file.relative_to(cache_dir)).replace("\\", "/")
            if rel not in valid:
                cache_file.unlink(missing_ok=True)
                count += 1
        return count

    orphan_cache_removed = await asyncio.to_thread(
        _remove_orphan_cache, cache_dir, valid_cache_names
    )
    if orphan_cache_removed:
        print(f"[cleanup] 清除 {orphan_cache_removed} 个孤儿缓存文件", flush=True)

    # ── 第 3 步：补全缺失的缩略图缓存（分批加载 + 多进程生成） ──
    async with async_session_factory() as session:
        last_id = 0
        loop = asyncio.get_running_loop()
        while True:
            stmt = (
                select(Image)
                .where(Image.id > last_id)
                .order_by(Image.id)
                .limit(_CLEANUP_BATCH_SIZE)
            )
            result = await session.execute(stmt)
            batch = list(result.scalars().all())
            if not batch:
                break

            to_regen: list[tuple[Path, Path, bool]] = []
            for img in batch:
                cache_name = cache_filename(img.relative_path)
                cache_path = cache_dir / cache_name
                if not cache_path.exists():
                    photo_path = photos_dir / img.relative_path
                    if photo_path.exists():
                        is_video = getattr(img, "media_type", "image") == "video"
                        to_regen.append((photo_path, cache_path, is_video))
                last_id = img.id or last_id

            if to_regen:
                regen_batch_size = min(_PROCESS_BATCH_SIZE, len(to_regen))
                with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                    for i in range(0, len(to_regen), regen_batch_size):
                        regen_batch = to_regen[i : i + regen_batch_size]
                        tasks = [
                            loop.run_in_executor(executor, _regenerate_one, item)
                            for item in regen_batch
                        ]
                        results = await asyncio.gather(*tasks)
                        cache_regenerated += sum(1 for ok in results if ok)
                        await asyncio.sleep(0)

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
