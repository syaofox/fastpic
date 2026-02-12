import hashlib
import os
from pathlib import Path

from PIL import Image as PILImage
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image, async_session_factory

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
THUMBNAIL_WIDTH = 300


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


async def scan_photos(photos_dir: Path, cache_dir: Path) -> int:
    """
    异步扫描 photos 目录，生成缩略图并写入数据库。
    返回新扫描的图片数量。
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()
    count = 0

    print(f"[scan] 开始扫描: {photos_dir}", flush=True)

    BATCH_SIZE = 50  # 每 50 张提交一次，边扫边可见

    async with async_session_factory() as session:
        # 收集所有图片文件
        image_files: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(photos_dir.rglob(f"*{ext}"))

        total_files = len(image_files)
        print(f"[scan] 发现 {total_files} 个图片文件", flush=True)

        batch_count = 0
        for full_path in image_files:
            if not full_path.is_file():
                continue

            rel_path = _relative_path(photos_dir, full_path)
            modified_at = os.path.getmtime(full_path)
            file_size = os.path.getsize(full_path)

            # 检查是否已存在
            result = await session.execute(
                select(Image).where(Image.relative_path == rel_path)
            )
            if result.scalar_one_or_none():
                continue

            try:
                with PILImage.open(full_path) as img:
                    width, height = img.size
                    img.load()
                    # 生成 300px 宽缩略图（等比缩放）
                    if img.width > THUMBNAIL_WIDTH:
                        ratio = THUMBNAIL_WIDTH / img.width
                        new_size = (THUMBNAIL_WIDTH, int(img.height * ratio))
                        thumb = img.resize(new_size, PILImage.Resampling.LANCZOS)
                    else:
                        thumb = img.copy()

                    # 处理 RGBA 等模式
                    if thumb.mode in ("RGBA", "P"):
                        thumb = thumb.convert("RGB")

                    cache_name = _cache_filename(rel_path)
                    cache_path = cache_dir / cache_name
                    thumb.save(cache_path, "WEBP", quality=85)

                filename = full_path.name
                record = Image(
                    filename=filename,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
                )
                session.add(record)
                count += 1
                batch_count += 1

                # 分批提交
                if batch_count >= BATCH_SIZE:
                    await session.commit()
                    print(f"[scan] 进度: {count}/{total_files}", flush=True)
                    batch_count = 0

            except Exception as e:
                print(f"[scan] 处理失败 {full_path}: {e}", flush=True)
                continue

        # 提交剩余
        if batch_count > 0:
            await session.commit()
        print(f"[scan] 扫描完成，新增 {count} 条记录", flush=True)

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

        if batch_count > 0:
            await session.commit()

        if stale_removed:
            print(f"[cleanup] 清除 {stale_removed} 条幽灵记录（原图已删除）", flush=True)

    # ── 第 2 步：清除孤儿缓存文件 ──
    if cache_dir.exists():
        for cache_file in cache_dir.iterdir():
            if cache_file.suffix == ".webp" and cache_file.name not in valid_cache_names:
                cache_file.unlink(missing_ok=True)
                orphan_cache_removed += 1

        if orphan_cache_removed:
            print(f"[cleanup] 清除 {orphan_cache_removed} 个孤儿缓存文件", flush=True)

    # ── 第 3 步：补全缺失的缩略图缓存 ──
    async with async_session_factory() as session:
        result = await session.execute(select(Image))
        all_images = list(result.scalars().all())

        for img in all_images:
            cache_name = _cache_filename(img.relative_path)
            cache_path = cache_dir / cache_name
            if not cache_path.exists():
                photo_path = photos_dir / img.relative_path
                if photo_path.exists():
                    if _generate_thumbnail(photo_path, cache_path):
                        cache_regenerated += 1

        if cache_regenerated:
            print(f"[cleanup] 重新生成 {cache_regenerated} 个缺失缓存", flush=True)

    summary = {
        "stale_removed": stale_removed,
        "orphan_cache_removed": orphan_cache_removed,
        "cache_regenerated": cache_regenerated,
    }
    print(f"[cleanup] 清理完成: {summary}", flush=True)
    return summary
