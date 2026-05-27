import time

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models import ImageTag, Tag

HOT_TAGS_CACHE_TTL = 300.0
_hot_tags_cache: list[dict] | None = None
_hot_tags_cache_ts: float = 0


def invalidate_hot_tags_cache() -> None:
    global _hot_tags_cache, _hot_tags_cache_ts
    _hot_tags_cache = None
    _hot_tags_cache_ts = 0


async def get_hot_tags_cached(session: AsyncSession) -> list[dict]:
    global _hot_tags_cache, _hot_tags_cache_ts
    now = time.monotonic()
    if _hot_tags_cache is not None and (now - _hot_tags_cache_ts) < HOT_TAGS_CACHE_TTL:
        return _hot_tags_cache
    tag_stmt = (
        select(Tag.name, func.count(ImageTag.image_id).label("count"))
        .outerjoin(ImageTag, ImageTag.tag_id == Tag.id)
        .group_by(Tag.id, Tag.name)
        .order_by(func.count(ImageTag.image_id).desc(), Tag.name)
        .limit(100)
    )
    tag_result = await session.execute(tag_stmt)
    _hot_tags_cache = [{"name": r[0], "count": r[1] or 0} for r in tag_result.fetchall()]
    _hot_tags_cache_ts = now
    return _hot_tags_cache
