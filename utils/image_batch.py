"""按路径前缀分批加载 Image 的迭代器，供 move/rename/delete/merge 等复用"""
from collections.abc import AsyncIterator

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image
from .path_utils import path_filter_for_prefix


async def iter_images_by_path_prefix(
    session: AsyncSession,
    prefix: str,
    batch_size: int = 1000,
) -> AsyncIterator[list[Image]]:
    """按路径前缀分批 yield Image 列表，用于大批量文件夹操作。"""
    pf = path_filter_for_prefix(Image.relative_path, prefix)
    last_id = 0
    while True:
        stmt = (
            select(Image)
            .where(pf)
            .where(Image.id > last_id)
            .order_by(Image.id)
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        images = list(result.scalars().all())
        if not images:
            break
        yield images
        last_id = images[-1].id or last_id


async def collect_image_items_by_prefix(
    session: AsyncSession,
    prefix: str,
    src: str,
    batch_size: int = 1000,
) -> list[tuple[int, str, str]]:
    """按路径前缀收集 (id, relative_path, src) 列表，供 merge_folders 使用"""
    items: list[tuple[int, str, str]] = []
    async for batch in iter_images_by_path_prefix(session, prefix, batch_size):
        for img in batch:
            if img.id:
                items.append((img.id, img.relative_path, src))
    return items
