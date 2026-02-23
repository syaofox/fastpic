"""FastPic 应用入口"""
import asyncio
import mimetypes
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from config import PHOTOS_DIR, CACHE_DIR, STATIC_DIR, PER_PAGE, APP_VERSION
from models import Image, Tag, ImageTag, init_db, get_async_session, async_session_factory
from scanner import scan_photos, scan_videos, cleanup_database
from scan_state import begin_scan, end_scan
from watcher import start_watcher
from app_common import templates
from routers import auth, tags, images, folders, settings
from utils.path_utils import normalize_path, path_filter_for_prefix
from utils.folder_tree import get_folder_tree_cached, get_subfolders
from utils.query_builder import (
    get_sort_column,
    parse_filter_params,
    apply_image_filters,
    apply_image_filters_to_count,
)


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


# count 短期缓存（无筛选时），减轻切换文件夹时的重复查询
_COUNT_CACHE_TTL = 60.0
_count_cache: dict[tuple[str, str], tuple[int, float]] = {}


def _get_cached_count(path: str, mode: str) -> int | None:
    key = (path or "", mode)
    entry = _count_cache.get(key)
    if entry is None:
        return None
    total, ts = entry
    if time.monotonic() - ts > _COUNT_CACHE_TTL:
        del _count_cache[key]
        return None
    return total


def _set_cached_count(path: str, mode: str, total: int) -> None:
    key = (path or "", mode)
    _count_cache[key] = (total, time.monotonic())


@app.get("/")
async def index(request: Request, session: AsyncSession = Depends(get_async_session)):
    """返回主页框架"""
    folder_tree, nested_tree, folder_counts = await get_folder_tree_cached(
        PHOTOS_DIR, session=session
    )
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
    folder_tree, nested_tree, folder_counts = await get_folder_tree_cached(
        PHOTOS_DIR, session=session
    )
    return templates.TemplateResponse(
        "partials/folder_tree.html",
        {"request": request, "nested_tree": nested_tree, "folder_counts": folder_counts, "current_path": path},
    )


def _parse_cursor(cursor: str, sort_by: str) -> tuple[float | str | None, int | None]:
    """解析 keyset 游标，返回 (sort_value, id)。支持 modified_at/file_size 数值或字符串。"""
    if not cursor or "_" not in cursor:
        return None, None
    try:
        parts = cursor.rsplit("_", 1)
        if len(parts) != 2:
            return None, None
        val_str, id_str = parts
        row_id = int(id_str)
        if sort_by in ("modified_at", "file_size"):
            return float(val_str), row_id
        return val_str, row_id
    except (ValueError, TypeError):
        return None, None


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
    cursor: str = "",
    filter_filename: str = "",
    filter_size_min: str = "",
    filter_size_max: str = "",
    filter_date_from: str = "",
    filter_date_to: str = "",
    filter_tag: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片网格 HTML 片段（供 HTMX 调用）。支持 cursor 游标分页，百万级时避免 offset 性能问题。"""
    path = normalize_path(path, allow_empty=True) or ""
    valid_modes = ("folder", "list", "waterfall")
    mode = mode if mode in valid_modes else "folder"
    per_page = _per_page_for_cols(cols)
    sort_col = get_sort_column(sort_by)
    sort_order = "asc" if sort_order == "asc" else "desc"
    order_clause = sort_col.asc() if sort_order == "asc" else sort_col.desc()
    stmt = select(Image).order_by(order_clause, Image.id.asc() if sort_order == "asc" else Image.id.desc())
    parsed = parse_filter_params(
        filter_filename, filter_size_min, filter_size_max,
        filter_date_from, filter_date_to, filter_tag,
    )
    stmt, pf, has_filters = apply_image_filters(stmt, path, search, mode, parsed)
    count_stmt = apply_image_filters_to_count(
        select(func.count(Image.id)), path, search, mode, parsed, pf
    )

    use_keyset = bool(cursor) and page > 1 and sort_by in ("modified_at", "file_size")
    cursor_val, cursor_id = _parse_cursor(cursor, sort_by) if cursor else (None, None)
    if use_keyset and cursor_val is not None and cursor_id is not None:
        from sqlalchemy import or_
        sort_col_raw = Image.modified_at if sort_by == "modified_at" else Image.file_size
        if sort_order == "desc":
            stmt = stmt.where(
                or_(
                    sort_col_raw < cursor_val,
                    (sort_col_raw == cursor_val) & (Image.id < cursor_id),
                )
            )
        else:
            stmt = stmt.where(
                or_(
                    sort_col_raw > cursor_val,
                    (sort_col_raw == cursor_val) & (Image.id > cursor_id),
                )
            )
    if not use_keyset:
        offset = (page - 1) * per_page
        stmt = stmt.offset(offset)
    stmt_paged = stmt.limit(per_page + 1)
    need_count = search or has_filters or parsed["filter_tag"] or _get_cached_count(path, mode) is None
    need_subfolders = (
        mode in ("folder", "list")
        and page == 1
        and not search
        and not has_filters
        and not parsed["filter_tag"]
    )

    async def _run_count():
        cached = _get_cached_count(path, mode)
        if cached is not None:
            return cached
        async with async_session_factory() as s:
            t = (await s.execute(count_stmt)).scalar() or 0
            if not search and not has_filters and not parsed["filter_tag"]:
                _set_cached_count(path, mode, t)
            return t

    async def _run_subfolders():
        if not need_subfolders:
            return []
        async with async_session_factory() as s:
            return await get_subfolders(s, PHOTOS_DIR, path, pf, sort_by, sort_order)

    async def _run_images():
        async with async_session_factory() as s:
            result = await s.execute(stmt_paged)
            return list(result.scalars().all())

    if need_count and need_subfolders:
        total, subfolders, images_list = await asyncio.gather(
            _run_count(), _run_subfolders(), _run_images()
        )
    elif need_count:
        total, images_list = await asyncio.gather(_run_count(), _run_images())
        subfolders = []
    elif need_subfolders:
        subfolders, images_list = await asyncio.gather(_run_subfolders(), _run_images())
        total = _get_cached_count(path, mode) or 0
    else:
        total = _get_cached_count(path, mode) or 0
        subfolders = []
        images_list = await _run_images()
    has_next = len(images_list) > per_page
    if has_next:
        images_list = images_list[:per_page]
    next_cursor = ""
    if images_list and has_next and sort_by in ("modified_at", "file_size"):
        last_img = images_list[-1]
        val = last_img.modified_at if sort_by == "modified_at" else (last_img.file_size or 0)
        next_cursor = f"{val}_{last_img.id}"
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
            "next_cursor": next_cursor,
        },
    )


_FOLDER_IMAGES_MAX = 5000  # 大图模式最大返回数量，防止 DoS


@app.get("/api/folder-images")
async def api_folder_images(
    path: str = "",
    search: str = "",
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
    path = normalize_path(path, allow_empty=True) or ""
    valid_modes = ("folder", "list", "waterfall")
    mode = mode if mode in valid_modes else "folder"
    sort_col = get_sort_column(sort_by)
    sort_order = "asc" if sort_order == "asc" else "desc"
    order_clause = sort_col.asc() if sort_order == "asc" else sort_col.desc()
    stmt = (
        select(Image.id, Image.relative_path, Image.media_type)
        .order_by(order_clause)
        .limit(_FOLDER_IMAGES_MAX + 1)
    )
    parsed = parse_filter_params(
        filter_filename, filter_size_min, filter_size_max,
        filter_date_from, filter_date_to, filter_tag,
    )
    stmt, _, _ = apply_image_filters(stmt, path, search, mode, parsed)
    result = await session.execute(stmt)
    rows = result.fetchall()
    truncated = len(rows) > _FOLDER_IMAGES_MAX
    if truncated:
        rows = rows[:_FOLDER_IMAGES_MAX]
    return {
        "urls": ["/photos/" + r.relative_path for r in rows],
        "ids": [r.id for r in rows],
        "media_types": [getattr(r, "media_type", "image") for r in rows],
        "truncated": truncated,
    }


@app.get("/debug/path-count")
async def debug_path_count(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """调试：查看指定路径下的图片数量"""
    path = normalize_path(path, allow_empty=True) or ""
    if not path:
        total = (await session.execute(select(func.count(Image.id)))).scalar() or 0
        return {"path": "", "total": total, "note": "path 为空时返回全部"}
    pf = path_filter_for_prefix(Image.relative_path, path)
    total = (await session.execute(select(func.count(Image.id)).where(pf))).scalar() or 0
    result = await session.execute(select(Image.relative_path).where(pf).limit(5))
    sample_paths = [r[0] for r in result.fetchall()]
    return {"path": path, "total": total, "sample_paths": sample_paths}


class CachedStaticFiles(StaticFiles):
    """带 Cache-Control 头的静态文件服务"""

    def __init__(self, *args, cache_control: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if self.cache_control and hasattr(response, "headers"):
            response.headers.setdefault("Cache-Control", self.cache_control)
        return response


STATIC_DIR.mkdir(exist_ok=True)
mimetypes.add_type("video/mp2t", ".ts")
app.mount(
    "/photos",
    CachedStaticFiles(directory=str(PHOTOS_DIR), cache_control="public, max-age=3600"),
    name="photos",
)
app.mount(
    "/cache",
    CachedStaticFiles(
        directory=str(CACHE_DIR),
        cache_control="public, max-age=86400, immutable",
    ),
    name="cache",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
