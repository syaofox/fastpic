import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image, init_db, get_async_session
from scanner import scan_photos, _cache_filename

PHOTOS_DIR = Path(__file__).parent / "photos"
CACHE_DIR = Path(__file__).parent / "cache"
PER_PAGE = 24

PHOTOS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # 后台异步扫描
    asyncio.create_task(scan_photos(PHOTOS_DIR, CACHE_DIR))
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
templates.env.filters["cache_key"] = _cache_filename
templates.env.filters["urlencode_path"] = lambda s: quote(s or "", safe="")


def _escape_like(value: str) -> str:
    """转义 SQL LIKE 中的 % 和 _，避免被当作通配符"""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _get_folder_tree(rel_paths: list[str]) -> list[list[str]]:
    """从 relative_path 列表提取文件夹树，返回 [['2024'], ['2024','01'], ...]"""
    folders: set[tuple[str, ...]] = set()
    folders.add(())  # 根
    for rp in rel_paths:
        parts = rp.split("/")
        if len(parts) > 1:
            for i in range(1, len(parts)):
                folders.add(tuple(parts[:i]))
    return [list(f) for f in sorted(folders) if f]


def _build_nested_tree(flat_folders: list[list[str]]) -> dict:
    """将扁平文件夹列表转为嵌套树结构，用于可折叠渲染。
    返回格式: {'2024': {'01': {'Jan': {}}, '02': {}}, '2025': {}}"""
    root: dict = {}
    for parts in flat_folders:
        d = root
        for part in parts:
            if part not in d:
                d[part] = {}
            d = d[part]
    return root


@app.get("/")
async def index(request: Request, session: AsyncSession = Depends(get_async_session)):
    """返回主页框架"""
    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree = _get_folder_tree(rel_paths)
    nested_tree = _build_nested_tree(folder_tree)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "folder_tree": folder_tree, "nested_tree": nested_tree},
    )


async def _get_subfolders(session: AsyncSession, path: str, path_filter) -> list[dict]:
    """获取当前路径下的直接子文件夹，每个子文件夹取 4 张代表图"""
    # 查询该路径下所有 relative_path（用于提取子文件夹名）
    subfolder_stmt = select(Image.relative_path)
    if path:
        subfolder_stmt = subfolder_stmt.where(path_filter)
    subfolder_stmt = subfolder_stmt.distinct()
    result = await session.execute(subfolder_stmt)
    all_paths = [r[0] for r in result.fetchall()]

    path_depth = len(path.split("/")) if path else 0
    subfolder_names: set[str] = set()
    for rel in all_paths:
        parts = rel.split("/")
        if len(parts) > path_depth + 1:
            subfolder_names.add(parts[path_depth])

    subfolders: list[dict] = []
    for name in sorted(subfolder_names):
        full_path = f"{path}/{name}" if path else name
        escaped_sub = _escape_like(full_path)
        sub_path_filter = Image.relative_path.like(f"{escaped_sub}/%", escape="\\")
        thumb_stmt = (
            select(Image.relative_path)
            .where(sub_path_filter)
            .order_by(Image.modified_at.desc())
            .limit(4)
        )
        thumb_result = await session.execute(thumb_stmt)
        thumbnails = [r[0] for r in thumb_result.fetchall()]
        subfolders.append({"name": name, "full_path": full_path, "thumbnails": thumbnails})

    return subfolders


@app.get("/gallery")
async def gallery(
    request: Request,
    path: str = "",
    search: str = "",
    mode: str = "folder",
    page: int = 1,
    per_page: int = PER_PAGE,
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片网格 HTML 片段（供 HTMX 调用）
    mode: folder=仅当前层文件+子文件夹, waterfall=递归所有图片,无文件夹无文件名
    """
    from sqlalchemy import func

    # 规范化路径：去除首尾空格和斜杠
    path = (path or "").strip().strip("/")
    mode = "waterfall" if mode == "waterfall" else "folder"

    stmt = select(Image).order_by(Image.modified_at.desc())
    if path:
        escaped = _escape_like(path)
        path_filter = (
            Image.relative_path.like(f"{escaped}/%", escape="\\")
            | (Image.relative_path == path)
        )
        stmt = stmt.where(path_filter)
    else:
        path_filter = None
    if search:
        stmt = stmt.where(Image.filename.ilike(f"%{search}%"))

    # 文件夹模式：仅当前层直接文件，不含子文件夹下的图片
    if mode == "folder":
        if path:
            escaped = _escape_like(path)
            direct_filter = ~Image.relative_path.like(f"{escaped}/%/%", escape="\\")
            stmt = stmt.where(direct_filter)
        else:
            stmt = stmt.where(~Image.relative_path.like("%/%"))

    # 总数
    count_stmt = select(func.count(Image.id))
    if path:
        count_stmt = count_stmt.where(path_filter)
    if search:
        count_stmt = count_stmt.where(Image.filename.ilike(f"%{search}%"))
    if mode == "folder":
        if path:
            count_stmt = count_stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        else:
            count_stmt = count_stmt.where(~Image.relative_path.like("%/%"))
    total = (await session.execute(count_stmt)).scalar() or 0

    # 子文件夹（仅文件夹模式、首页且无搜索时）
    subfolders: list[dict] = []
    if mode == "folder" and page == 1 and not search:
        subfolders = await _get_subfolders(session, path, path_filter)

    # 分页
    offset = (page - 1) * per_page
    stmt = stmt.offset(offset).limit(per_page + 1)
    result = await session.execute(stmt)
    images = list(result.scalars().all())
    has_next = len(images) > per_page
    if has_next:
        images = images[:per_page]

    # 面包屑
    breadcrumb_parts = path.split("/") if path else []

    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "images": images,
            "path": path,
            "search": search,
            "mode": mode,
            "page": page,
            "per_page": per_page,
            "has_next": has_next,
            "total": total,
            "append": page > 1,
            "subfolders": subfolders,
            "breadcrumb_parts": breadcrumb_parts,
        },
    )


@app.get("/debug/path-count")
async def debug_path_count(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """调试：查看指定路径下的图片数量（用于排查路径过滤问题）"""
    from sqlalchemy import func

    path = (path or "").strip().strip("/")
    if not path:
        total = (await session.execute(select(func.count(Image.id)))).scalar() or 0
        return {"path": "", "total": total, "note": "path 为空时返回全部"}

    escaped = _escape_like(path)
    path_filter = (
        Image.relative_path.like(f"{escaped}/%", escape="\\")
        | (Image.relative_path == path)
    )
    total = (
        await session.execute(
            select(func.count(Image.id)).where(path_filter)
        )
    ).scalar() or 0

    # 取前 5 条示例路径
    result = await session.execute(
        select(Image.relative_path).where(path_filter).limit(5)
    )
    sample_paths = [r[0] for r in result.fetchall()]
    return {"path": path, "total": total, "sample_paths": sample_paths}


@app.post("/scan")
async def trigger_scan():
    """手动触发扫描"""
    n = await scan_photos(PHOTOS_DIR, CACHE_DIR)
    return {"scanned": n}


# 静态目录挂载（放在路由之后，mount 会匹配前缀路径）
app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")
app.mount("/cache", StaticFiles(directory=str(CACHE_DIR)), name="cache")
