"""标签 API"""
from sqlalchemy import delete
from sqlmodel import select
from fastapi import APIRouter, Depends, HTTPException

from models import Image, Tag, ImageTag, get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from schemas import AddTagsRequest, RenameTagRequest, MergeTagRequest, BatchDeleteTagsRequest
from utils.path_utils import escape_like

router = APIRouter(prefix="/api", tags=["tags"])


@router.get("/tags")
async def list_tags(
    q: str = "",
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """列出所有标签及图片数量，支持 ?q= 模糊搜索"""
    from sqlalchemy import func

    stmt = (
        select(Tag.name, func.count(ImageTag.image_id).label("count"))
        .outerjoin(ImageTag, ImageTag.tag_id == Tag.id)
        .group_by(Tag.id, Tag.name)
        .order_by(func.count(ImageTag.image_id).desc(), Tag.name)
    )
    if q:
        q_escaped = escape_like(q.strip())
        stmt = stmt.where(Tag.name.ilike(f"%{q_escaped}%", escape="\\"))
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    rows = result.fetchall()
    return {"tags": [{"name": r[0], "count": r[1] or 0} for r in rows]}


@router.put("/tags/{tag_name:path}")
async def rename_tag(
    tag_name: str,
    body: RenameTagRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """重命名标签"""
    tag_name = tag_name.strip()
    new_name = (body.name or "").strip()
    if not tag_name:
        raise HTTPException(status_code=400, detail="原标签名不能为空")
    if not new_name:
        raise HTTPException(status_code=400, detail="新标签名不能为空")
    if new_name == tag_name:
        return {"renamed": True}
    result = await session.execute(select(Tag).where(Tag.name == tag_name))
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    existing = await session.execute(select(Tag).where(Tag.name == new_name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="新名称已存在")
    tag.name = new_name
    await session.commit()
    return {"renamed": True}


@router.delete("/tags/{tag_name:path}")
async def delete_tag(
    tag_name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """删除标签及所有关联"""
    tag_name = tag_name.strip()
    if not tag_name:
        raise HTTPException(status_code=400, detail="标签名不能为空")
    result = await session.execute(select(Tag).where(Tag.name == tag_name))
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    await session.execute(delete(ImageTag).where(ImageTag.tag_id == tag.id))
    await session.delete(tag)
    await session.commit()
    return {"deleted": True}


@router.post("/tags/{tag_name:path}/merge")
async def merge_tag(
    tag_name: str,
    body: MergeTagRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将标签 A 合并到 B：A 下所有图片改为 B，再删除 A"""
    tag_name = tag_name.strip()
    target_name = (body.target or "").strip()
    if not tag_name:
        raise HTTPException(status_code=400, detail="源标签名不能为空")
    if not target_name:
        raise HTTPException(status_code=400, detail="目标标签名不能为空")
    if tag_name == target_name:
        raise HTTPException(status_code=400, detail="源标签与目标标签相同")
    result = await session.execute(select(Tag).where(Tag.name == tag_name))
    source_tag = result.scalar_one_or_none()
    if not source_tag:
        raise HTTPException(status_code=404, detail="源标签不存在")
    target_result = await session.execute(select(Tag).where(Tag.name == target_name))
    target_tag = target_result.scalar_one_or_none()
    if not target_tag:
        raise HTTPException(status_code=404, detail="目标标签不存在")
    img_result = await session.execute(
        select(ImageTag.image_id).where(ImageTag.tag_id == source_tag.id)
    )
    image_ids = [r[0] for r in img_result.fetchall()]
    for image_id in image_ids:
        has_target = await session.execute(
            select(ImageTag).where(
                ImageTag.image_id == image_id,
                ImageTag.tag_id == target_tag.id,
            )
        )
        if has_target.scalar_one_or_none() is None:
            session.add(ImageTag(image_id=image_id, tag_id=target_tag.id))
        await session.execute(
            delete(ImageTag).where(
                ImageTag.image_id == image_id,
                ImageTag.tag_id == source_tag.id,
            )
        )
    await session.delete(source_tag)
    await session.commit()
    return {"merged": True, "images_updated": len(image_ids)}


@router.post("/tags/batch-delete")
async def batch_delete_tags(
    body: BatchDeleteTagsRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """批量删除多个标签"""
    names = [n.strip() for n in (body.names or []) if (n or "").strip()]
    if not names:
        raise HTTPException(status_code=400, detail="请至少选择一个标签")
    deleted = 0
    for tag_name in names:
        result = await session.execute(select(Tag).where(Tag.name == tag_name))
        tag = result.scalar_one_or_none()
        if tag:
            await session.execute(delete(ImageTag).where(ImageTag.tag_id == tag.id))
            await session.delete(tag)
            deleted += 1
    await session.commit()
    return {"deleted": deleted}


@router.get("/images/{image_id:int}/tags")
async def get_image_tags(
    image_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """获取某图片的标签列表"""
    result = await session.execute(select(Image).where(Image.id == image_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    tag_result = await session.execute(
        select(Tag.name)
        .join(ImageTag, ImageTag.tag_id == Tag.id)
        .where(ImageTag.image_id == image_id)
        .order_by(Tag.name)
    )
    tags = [r[0] for r in tag_result.fetchall()]
    return {"tags": tags}


@router.post("/images/{image_id:int}/tags")
async def add_image_tags(
    image_id: int,
    body: AddTagsRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """为图片添加标签"""
    result = await session.execute(select(Image).where(Image.id == image_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    added = 0
    for name in (body.tags or []):
        name = (name or "").strip()
        if not name:
            continue
        tag_result = await session.execute(select(Tag).where(Tag.name == name))
        tag = tag_result.scalar_one_or_none()
        if not tag:
            tag = Tag(name=name)
            session.add(tag)
            await session.flush()
        existing = await session.execute(
            select(ImageTag).where(ImageTag.image_id == image_id, ImageTag.tag_id == tag.id)
        )
        if existing.scalar_one_or_none() is None:
            session.add(ImageTag(image_id=image_id, tag_id=tag.id))
            added += 1
    await session.commit()
    return {"added": added}


@router.delete("/images/{image_id:int}/tags/{tag_name:str}")
async def remove_image_tag(
    image_id: int,
    tag_name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """移除图片的指定标签"""
    tag_name = tag_name.strip()
    if not tag_name:
        raise HTTPException(status_code=400, detail="标签名不能为空")
    result = await session.execute(select(Tag).where(Tag.name == tag_name))
    tag = result.scalar_one_or_none()
    if not tag:
        return {"removed": False}
    link_result = await session.execute(
        select(ImageTag).where(ImageTag.image_id == image_id, ImageTag.tag_id == tag.id)
    )
    link = link_result.scalar_one_or_none()
    if link:
        await session.delete(link)
        await session.commit()
        return {"removed": True}
    return {"removed": False}
