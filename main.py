"""FastPic 应用入口"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime as _dt
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import select
from sqlalchemy import case, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR, STATIC_DIR, PER_PAGE, APP_VERSION
from models import Image, Tag, ImageTag, init_db, get_async_session
from scanner import scan_photos, scan_videos, cleanup_database, _cache_filename
from scan_state import begin_scan, end_scan
from watcher import start_watcher
from app_common import templates
from routers import auth, tags, images, folders, settings
from utils.path_utils import escape_like, path_filter_for_prefix
from utils.folder_tree import get_folder_tree_cached, get_subfolders


async def _background_scan():
    """后台扫描包装：先清理再扫描，捕获并打印异常"""
    begin_scan()
    try:
        await cleanup_database(PHOTOS_DIR, CACHE_DIR)
        n_img = await scan_photos(PHOTOS_DIR, CACHE_DIR)
        n_vid = await scan_videos(PHOTOS_DIR, CACHE_DIR)
        print(f"[scan] 扫描完成，新增 {n_img} 张图片、{n_vid} 个视频")
    except Exception as e:
        import traceback
        print(f"[scan] 扫描失败: {e}")
        traceback.print_exc()
    finally:
        end_scan()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(_background_scan())
    loop = asyncio.get_running_loop()
    observer = start_watcher(PHOTOS_DIR, CACHE_DIR, loop)
    yield
    observer.stop()
    observer.join(timeout=5)


app = FastAPI(lifespan=lifespan)
auth.setup_auth_middleware(app)

app.include_router(auth.router)
app.include_router(tags.router)
app.include_router(images.router)
app.include_router(folders.router)
app.include_router(settings.router)


@app.get("/favicon.ico")
async def favicon():
    """返回网站图标"""
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/x-icon")
    from fastapi import HTTPException
    raise HTTPException(status_code=404)


def _per_page_for_cols(cols: int) -> int:
    cols = max(2, min(8, cols))
    return cols * ((PER_PAGE + cols - 1) // cols)


@app.get("/")
async def index(request: Request, session: AsyncSession = Depends(get_async_session)):
    """返回主页框架"""
    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree, nested_tree, folder_counts = await get_folder_tree_cached(PHOTOS_DIR, rel_paths)
    tag_stmt = (
        select(Tag.name, func.count(ImageTag.image_id).label("count"))
        .outerjoin(ImageTag, ImageTag.tag_id == Tag.id)
        .group_by(Tag.id, Tag.name)
        .order_by(func.count(ImageTag.image_id).desc(), Tag.name)
        .limit(100)
    )
    tag_result = await session.execute(tag_stmt)
    all_tags = [{"name": r[0], "count": r[1] or 0} for r in tag_result.fetchall()]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "folder_tree": folder_tree,
            "nested_tree": nested_tree,
            "folder_counts": folder_counts,
            "all_tags": all_tags,
            "version": APP_VERSION,
        },
    )


@app.get("/api/sidebar-folder-tree")
async def sidebar_folder_tree(request: Request, session: AsyncSession = Depends(get_async_session)):
    """返回侧栏文件夹树 HTML 片段"""
    path = request.query_params.get("path", "")
    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree, nested_tree, folder_counts = await get_folder_tree_cached(PHOTOS_DIR, rel_paths)
    return templates.TemplateResponse(
        "partials/folder_tree.html",
        {"request": request, "nested_tree": nested_tree, "folder_counts": folder_counts, "current_path": path},
    )


@app.get("/gallery")
async def gallery(
    request: Request,
    path: str = "",
    search: str = "",
    mode: str = "folder",
    sort_by: str = "modified_at",
    sort_order: str = "desc",
    page: int = 1,
    cols: int = 4,
    filter_filename: str = "",
    filter_size_min: str = "",
    filter_size_max: str = "",
    filter_date_from: str = "",
    filter_date_to: str = "",
    filter_tag: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片网格 HTML 片段（供 HTMX 调用）"""
    path = (path or "").strip().strip("/")
    mode = "waterfall" if mode == "waterfall" else "folder"
    per_page = _per_page_for_cols(cols)
    sort_columns = {
        "filename": case(
            (Image.filename_natural.is_(None), Image.filename),
            else_=Image.filename_natural,
        ),
        "folder_filename": case(
            (Image.relative_path_natural.is_(None), Image.relative_path),
            else_=Image.relative_path_natural,
        ),
        "modified_at": Image.modified_at,
        "file_size": Image.file_size,
    }
    sort_col = sort_columns.get(sort_by, Image.modified_at)
    sort_order = "asc" if sort_order == "asc" else "desc"
    order_clause = sort_col.asc() if sort_order == "asc" else sort_col.desc()
    stmt = select(Image).order_by(order_clause)
    if path:
        pf = path_filter_for_prefix(Image.relative_path, path)
        stmt = stmt.where(pf)
    else:
        pf = None
    if search:
        stmt = stmt.where(Image.filename.ilike(f"%{search}%"))
    filter_filename = (filter_filename or "").strip()
    has_filters = False
    if filter_filename:
        escaped_fn = escape_like(filter_filename)
        stmt = stmt.where(Image.filename.ilike(f"%{escaped_fn}%", escape="\\"))
        has_filters = True
    _size_min = int(filter_size_min) if filter_size_min and filter_size_min.isdigit() else None
    _size_max = int(filter_size_max) if filter_size_max and filter_size_max.isdigit() else None
    if _size_min is not None:
        stmt = stmt.where(Image.file_size >= _size_min)
        has_filters = True
    if _size_max is not None:
        stmt = stmt.where(Image.file_size <= _size_max)
        has_filters = True
    _date_from_ts = None
    _date_to_ts = None
    if filter_date_from:
        try:
            _date_from_ts = _dt.strptime(filter_date_from, "%Y-%m-%d").timestamp()
        except ValueError:
            pass
    if filter_date_to:
        try:
            _date_to_ts = _dt.strptime(filter_date_to, "%Y-%m-%d").timestamp() + 86399
        except ValueError:
            pass
    if _date_from_ts is not None:
        stmt = stmt.where(Image.modified_at >= _date_from_ts)
        has_filters = True
    if _date_to_ts is not None:
        stmt = stmt.where(Image.modified_at <= _date_to_ts)
        has_filters = True
    filter_tag = (filter_tag or "").strip()
    if filter_tag:
        stmt = stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == filter_tag)
        has_filters = True
    if mode == "folder":
        if path:
            escaped = escape_like(path)
            stmt = stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        elif not filter_tag:
            stmt = stmt.where(~Image.relative_path.like("%/%"))
    count_stmt = select(func.count(Image.id))
    if path:
        count_stmt = count_stmt.where(pf)
    if search:
        count_stmt = count_stmt.where(Image.filename.ilike(f"%{search}%"))
    if filter_filename:
        count_stmt = count_stmt.where(Image.filename.ilike(f"%{escape_like(filter_filename)}%", escape="\\"))
    if _size_min is not None:
        count_stmt = count_stmt.where(Image.file_size >= _size_min)
    if _size_max is not None:
        count_stmt = count_stmt.where(Image.file_size <= _size_max)
    if _date_from_ts is not None:
        count_stmt = count_stmt.where(Image.modified_at >= _date_from_ts)
    if _date_to_ts is not None:
        count_stmt = count_stmt.where(Image.modified_at <= _date_to_ts)
    if filter_tag:
        count_stmt = count_stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == filter_tag)
    if mode == "folder":
        if path:
            escaped = escape_like(path)
            count_stmt = count_stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        elif not filter_tag:
            count_stmt = count_stmt.where(~Image.relative_path.like("%/%"))
    total = (await session.execute(count_stmt)).scalar() or 0
    subfolders = []
    if mode == "folder" and page == 1 and not search and not has_filters and not filter_tag:
        subfolders = await get_subfolders(session, PHOTOS_DIR, path, pf, sort_by, sort_order)
    offset = (page - 1) * per_page
    stmt = stmt.offset(offset).limit(per_page + 1)
    result = await session.execute(stmt)
    images_list = list(result.scalars().all())
    has_next = len(images_list) > per_page
    if has_next:
        images_list = images_list[:per_page]
    image_tags_map: dict[int, list[str]] = {}
    if images_list:
        image_ids = [img.id for img in images_list if img.id]
        if image_ids:
            tag_stmt = (
                select(ImageTag.image_id, Tag.name)
                .join(Tag, Tag.id == ImageTag.tag_id)
                .where(ImageTag.image_id.in_(image_ids))
                .order_by(Tag.name)
            )
            tag_result = await session.execute(tag_stmt)
            for img_id, tag_name in tag_result.fetchall():
                if img_id not in image_tags_map:
                    image_tags_map[img_id] = []
                image_tags_map[img_id].append(tag_name)
    breadcrumb_parts = path.split("/") if path else []
    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "images": images_list,
            "path": path,
            "search": search,
            "mode": mode,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "page": page,
            "per_page": per_page,
            "has_next": has_next,
            "total": total,
            "append": page > 1,
            "subfolders": subfolders,
            "breadcrumb_parts": breadcrumb_parts,
            "filter_filename": filter_filename,
            "filter_size_min": filter_size_min,
            "filter_size_max": filter_size_max,
            "filter_date_from": filter_date_from,
            "filter_date_to": filter_date_to,
            "filter_tag": filter_tag,
            "has_filters": has_filters,
            "cols": cols,
            "image_tags_map": image_tags_map,
        },
    )


@app.get("/api/folder-images")
async def api_folder_images(
    path: str = "",
    mode: str = "folder",
    sort_by: str = "modified_at",
    sort_order: str = "desc",
    filter_filename: str = "",
    filter_size_min: str = "",
    filter_size_max: str = "",
    filter_date_from: str = "",
    filter_date_to: str = "",
    filter_tag: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """获取当前文件夹/模式下的全部图片（用于大图浏览模式）"""
    path = (path or "").strip().strip("/")
    mode = "waterfall" if mode == "waterfall" else "folder"
    sort_columns = {
        "filename": case(
            (Image.filename_natural.is_(None), Image.filename),
            else_=Image.filename_natural,
        ),
        "folder_filename": case(
            (Image.relative_path_natural.is_(None), Image.relative_path),
            else_=Image.relative_path_natural,
        ),
        "modified_at": Image.modified_at,
        "file_size": Image.file_size,
    }
    sort_col = sort_columns.get(sort_by, Image.modified_at)
    sort_order = "asc" if sort_order == "asc" else "desc"
    order_clause = sort_col.asc() if sort_order == "asc" else sort_col.desc()
    stmt = select(Image).order_by(order_clause)
    if path:
        pf = path_filter_for_prefix(Image.relative_path, path)
        stmt = stmt.where(pf)
    if mode == "folder":
        if path:
            escaped = escape_like(path)
            stmt = stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        else:
            stmt = stmt.where(~Image.relative_path.like("%/%"))
    filter_filename = (filter_filename or "").strip()
    if filter_filename:
        stmt = stmt.where(Image.filename.ilike(f"%{escape_like(filter_filename)}%", escape="\\"))
    _size_min = int(filter_size_min) if filter_size_min and filter_size_min.isdigit() else None
    _size_max = int(filter_size_max) if filter_size_max and filter_size_max.isdigit() else None
    if _size_min is not None:
        stmt = stmt.where(Image.file_size >= _size_min)
    if _size_max is not None:
        stmt = stmt.where(Image.file_size <= _size_max)
    _date_from_ts = None
    _date_to_ts = None
    if filter_date_from:
        try:
            _date_from_ts = _dt.strptime(filter_date_from, "%Y-%m-%d").timestamp()
        except ValueError:
            pass
    if filter_date_to:
        try:
            _date_to_ts = _dt.strptime(filter_date_to, "%Y-%m-%d").timestamp() + 86399
        except ValueError:
            pass
    if _date_from_ts is not None:
        stmt = stmt.where(Image.modified_at >= _date_from_ts)
    if _date_to_ts is not None:
        stmt = stmt.where(Image.modified_at <= _date_to_ts)
    filter_tag = (filter_tag or "").strip()
    if filter_tag:
        stmt = stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == filter_tag)
    result = await session.execute(stmt)
    images_list = list(result.scalars().all())
    return {
        "urls": ["/photos/" + img.relative_path for img in images_list],
        "ids": [img.id for img in images_list],
        "media_types": [getattr(img, "media_type", "image") for img in images_list],
    }


@app.get("/debug/path-count")
async def debug_path_count(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """调试：查看指定路径下的图片数量"""
    path = (path or "").strip().strip("/")
    if not path:
        total = (await session.execute(select(func.count(Image.id)))).scalar() or 0
        return {"path": "", "total": total, "note": "path 为空时返回全部"}
    pf = path_filter_for_prefix(Image.relative_path, path)
    total = (await session.execute(select(func.count(Image.id)).where(pf))).scalar() or 0
    result = await session.execute(select(Image.relative_path).where(pf).limit(5))
    sample_paths = [r[0] for r in result.fetchall()]
    return {"path": path, "total": total, "sample_paths": sample_paths}


STATIC_DIR.mkdir(exist_ok=True)
app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")
app.mount("/cache", StaticFiles(directory=str(CACHE_DIR)), name="cache")
