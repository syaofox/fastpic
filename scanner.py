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


async def scan_photos(photos_dir: Path, cache_dir: Path) -> int:
    """
    异步扫描 photos 目录，生成缩略图并写入数据库。
    返回新扫描的图片数量。
    """
    photos_dir = photos_dir.resolve()
    cache_dir = cache_dir.resolve()
    count = 0

    async with async_session_factory() as session:
        # 收集所有图片文件
        image_files: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(photos_dir.rglob(f"*{ext}"))

        for full_path in image_files:
            if not full_path.is_file():
                continue

            rel_path = _relative_path(photos_dir, full_path)
            modified_at = os.path.getmtime(full_path)

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
                    width=width,
                    height=height,
                )
                session.add(record)
                count += 1

            except Exception:
                continue

        await session.commit()

    return count
