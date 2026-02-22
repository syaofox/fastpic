"""标签相关辅助函数"""
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from models import Tag, ImageTag


DAMAGED_TAG_NAME = "损坏"


async def ensure_tag_exists(session, tag_name: str) -> Tag | None:
    """确保标签存在，返回 Tag 对象。应在添加其他数据前调用。"""
    tag_name = (tag_name or "").strip()
    if not tag_name:
        return None
    result = await session.execute(select(Tag).where(Tag.name == tag_name))
    tag = result.scalar_one_or_none()
    if tag:
        return tag
    tag = Tag(name=tag_name)
    session.add(tag)
    try:
        await session.flush()
        return tag
    except IntegrityError:
        await session.rollback()
        result = await session.execute(select(Tag).where(Tag.name == tag_name))
        return result.scalar_one_or_none()


async def add_tag_to_image(session, image_id: int, tag: Tag) -> bool:
    """为图片添加标签，返回是否成功添加（已存在则返回 False）。tag 需已存在。"""
    existing = await session.execute(
        select(ImageTag).where(ImageTag.image_id == image_id, ImageTag.tag_id == tag.id)
    )
    if existing.scalar_one_or_none() is not None:
        return False
    session.add(ImageTag(image_id=image_id, tag_id=tag.id))
    await session.flush()
    return True
