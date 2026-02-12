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


@app.get("/gallery")
async def gallery(
    request: Request,
    path: str = "",
    search: str = "",
    page: int = 1,
    per_page: int = PER_PAGE,
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片网格 HTML 片段（供 HTMX 调用）"""
    from sqlalchemy import func

    # 规范化路径：去除首尾空格和斜杠
    path = (path or "").strip().strip("/")

    stmt = select(Image).order_by(Image.modified_at.desc())
    if path:
        # 转义 LIKE 通配符 % 和 _，避免错误匹配
        escaped = _escape_like(path)
        path_filter = (
            Image.relative_path.like(f"{escaped}/%", escape="\\")
            | (Image.relative_path == path)
        )
        stmt = stmt.where(path_filter)
    if search:
        stmt = stmt.where(Image.filename.ilike(f"%{search}%"))

    # 总数
    count_stmt = select(func.count(Image.id))
    if path:
        count_stmt = count_stmt.where(path_filter)
    if search:
        count_stmt = count_stmt.where(Image.filename.ilike(f"%{search}%"))
    total = (await session.execute(count_stmt)).scalar() or 0

    # 分页
    offset = (page - 1) * per_page
    stmt = stmt.offset(offset).limit(per_page + 1)
    result = await session.execute(stmt)
    images = list(result.scalars().all())
    has_next = len(images) > per_page
    if has_next:
        images = images[:per_page]

    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "images": images,
            "path": path,
            "search": search,
            "page": page,
            "per_page": per_page,
            "has_next": has_next,
            "total": total,
            "append": page > 1,
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
