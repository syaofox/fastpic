import asyncio
import hmac
import os
import re
import secrets
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image, init_db, get_async_session, natural_sort_key
from scanner import scan_photos, cleanup_database, _cache_filename
from scan_state import begin_scan, end_scan, is_scanning
from watcher import start_watcher

PHOTOS_DIR = Path(__file__).parent / "photos"
CACHE_DIR = Path(__file__).parent / "cache"
PER_PAGE = 24

# ── 访问密码保护 ──
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()
_SESSION_TOKEN = secrets.token_hex(32) if ACCESS_PASSWORD else ""


def _get_version() -> str:
    """从 pyproject.toml 读取版本号"""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if pyproject_path.exists():
        text = pyproject_path.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            return m.group(1)
    return "unknown"


APP_VERSION = _get_version()

PHOTOS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

async def _background_scan():
    """后台扫描包装：先清理再扫描，捕获并打印异常"""
    begin_scan()
    try:
        # 先清理不一致数据
        await cleanup_database(PHOTOS_DIR, CACHE_DIR)
        # 再扫描新增图片
        n = await scan_photos(PHOTOS_DIR, CACHE_DIR)
        print(f"[scan] 扫描完成，新增 {n} 张图片")
    except Exception as e:
        import traceback
        print(f"[scan] 扫描失败: {e}")
        traceback.print_exc()
    finally:
        end_scan()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # 后台异步扫描
    asyncio.create_task(_background_scan())
    # 启动文件系统监听（实时感知 photos 目录变化）
    loop = asyncio.get_running_loop()
    observer = start_watcher(PHOTOS_DIR, CACHE_DIR, loop)
    yield
    # 关闭时停止监听
    observer.stop()
    observer.join(timeout=5)


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
templates.env.filters["cache_key"] = _cache_filename
templates.env.filters["urlencode_path"] = lambda s: quote(s or "", safe="")


# ── 密码保护中间件 ──

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """如果设置了 ACCESS_PASSWORD，则拦截未认证请求并重定向到登录页"""
    if not ACCESS_PASSWORD:
        return await call_next(request)
    path = request.url.path
    # 白名单：登录页和 favicon 不需要验证
    if path in ("/login", "/favicon.ico", "/api/scan-status"):
        return await call_next(request)
    token = request.cookies.get("fp_session")
    if not token or not hmac.compare_digest(token, _SESSION_TOKEN):
        return RedirectResponse(url="/login", status_code=302)
    return await call_next(request)


@app.get("/login")
async def login_page(request: Request):
    """显示登录页面"""
    # 如果未启用密码保护，直接跳转首页
    if not ACCESS_PASSWORD:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/login")
async def login_submit(request: Request):
    """验证密码并设置 session cookie"""
    form = await request.form()
    password = (form.get("password") or "").strip()
    if hmac.compare_digest(password, ACCESS_PASSWORD):
        response = RedirectResponse(url="/", status_code=302)
        # session cookie：不设置 max_age，浏览器关闭即失效
        response.set_cookie(key="fp_session", value=_SESSION_TOKEN, httponly=True, samesite="lax")
        return response
    # 密码错误
    return templates.TemplateResponse("login.html", {"request": request, "error": "密码错误，请重试"})


@app.get("/logout")
async def logout():
    """登出：清除 session cookie"""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="fp_session")
    return response


def _escape_like(value: str) -> str:
    """转义 SQL LIKE 中的 % 和 _，避免被当作通配符"""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _get_folder_tree(rel_paths: list[str]) -> list[list[str]]:
    """从 relative_path 列表 + 文件系统提取文件夹树，返回 [['2024'], ['2024','01'], ...]
    同时扫描文件系统，确保空文件夹也出现在树中。"""
    folders: set[tuple[str, ...]] = set()
    folders.add(())  # 根
    # 从数据库路径提取
    for rp in rel_paths:
        parts = rp.split("/")
        if len(parts) > 1:
            for i in range(1, len(parts)):
                folders.add(tuple(parts[:i]))
    # 从文件系统扫描所有子目录
    def _scan_dirs(base: Path, prefix: tuple[str, ...] = ()):
        if not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                path_tuple = prefix + (child.name,)
                folders.add(path_tuple)
                _scan_dirs(child, path_tuple)
    _scan_dirs(PHOTOS_DIR)
    return [list(f) for f in sorted(folders) if f]


def _compute_folder_counts(rel_paths: list[str]) -> dict[str, int]:
    """从 relative_path 列表计算每个文件夹下的图片总数（含子目录）。
    O(N * D)，N=图片数，D=平均路径深度，无需额外数据库查询。"""
    counts: dict[str, int] = {"": len(rel_paths)}  # 根目录 = 全部图片
    for rp in rel_paths:
        parts = rp.split("/")
        for i in range(1, len(parts)):  # 不含文件名本身
            prefix = "/".join(parts[:i])
            counts[prefix] = counts.get(prefix, 0) + 1
    return counts


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
    folder_counts = _compute_folder_counts(rel_paths)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "folder_tree": folder_tree, "nested_tree": nested_tree, "folder_counts": folder_counts, "version": APP_VERSION},
    )


@app.get("/api/sidebar-folder-tree")
async def sidebar_folder_tree(request: Request, session: AsyncSession = Depends(get_async_session)):
    """返回侧栏文件夹树 HTML 片段，用于无刷新更新"""
    path = request.query_params.get("path", "")
    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree = _get_folder_tree(rel_paths)
    nested_tree = _build_nested_tree(folder_tree)
    folder_counts = _compute_folder_counts(rel_paths)
    return templates.TemplateResponse(
        "partials/folder_tree.html",
        {"request": request, "nested_tree": nested_tree, "folder_counts": folder_counts, "current_path": path},
    )


async def _get_subfolders(
    session: AsyncSession,
    path: str,
    path_filter,
    sort_by: str = "filename",
    sort_order: str = "asc",
) -> list[dict]:
    """获取当前路径下的直接子文件夹，每个子文件夹取 4 张代表图。
    同时扫描文件系统，确保空文件夹也能显示。
    子文件夹按 sort_by/sort_order 参与排序。"""
    from sqlalchemy import func

    # 1) 从数据库中提取子文件夹名
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

    # 2) 从文件系统中扫描实际子目录（捕获空文件夹）
    fs_dir = PHOTOS_DIR / path if path else PHOTOS_DIR
    if fs_dir.is_dir():
        for child in fs_dir.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                subfolder_names.add(child.name)

    # 3) 预计算每个子文件夹的图片数及排序用字段（含递归子目录），O(N)
    subfolder_img_counts: dict[str, int] = {}
    subfolder_max_modified: dict[str, float] = {}
    subfolder_max_size: dict[str, int] = {}
    for rel in all_paths:
        parts = rel.split("/")
        if len(parts) > path_depth + 1:
            sub_name = parts[path_depth]
            subfolder_img_counts[sub_name] = subfolder_img_counts.get(sub_name, 0) + 1

    # 批量查询每个子文件夹的 max(modified_at) 和 max(file_size)
    for name in subfolder_names:
        full_path = f"{path}/{name}" if path else name
        escaped_sub = _escape_like(full_path)
        sub_path_filter = Image.relative_path.like(f"{escaped_sub}/%", escape="\\")
        agg_stmt = select(
            func.max(Image.modified_at).label("max_modified"),
            func.max(Image.file_size).label("max_size"),
        ).where(sub_path_filter)
        agg_result = await session.execute(agg_stmt)
        row = agg_result.fetchone()
        if row and row[0] is not None:
            subfolder_max_modified[name] = row[0]
        if row and row[1] is not None:
            subfolder_max_size[name] = row[1]

    # 4) 为每个子文件夹取缩略图
    subfolders: list[dict] = []
    for name in subfolder_names:
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
        subfolders.append({
            "name": name,
            "full_path": full_path,
            "thumbnails": thumbnails,
            "image_count": subfolder_img_counts.get(name, 0),
            "_sort_key_filename": natural_sort_key(name),
            "_sort_key_folder_filename": natural_sort_key(full_path),
            "_sort_key_modified_at": subfolder_max_modified.get(name, 0.0),
            "_sort_key_file_size": subfolder_max_size.get(name, 0),
        })

    # 5) 按 sort_by / sort_order 排序（filename/folder_filename 已用自然排序键）
    sort_col_map = {
        "filename": "_sort_key_filename",
        "folder_filename": "_sort_key_folder_filename",
        "modified_at": "_sort_key_modified_at",
        "file_size": "_sort_key_file_size",
    }
    key = sort_col_map.get(sort_by, "_sort_key_filename")
    reverse = sort_order == "desc"
    subfolders.sort(key=lambda s: s[key], reverse=reverse)

    return subfolders


def _per_page_for_cols(cols: int) -> int:
    """根据列数计算每页数量，确保能被列数整除，避免最后一行出现空位"""
    cols = max(2, min(8, cols))
    return cols * ((PER_PAGE + cols - 1) // cols)


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
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片网格 HTML 片段（供 HTMX 调用）
    mode: folder=仅当前层文件+子文件夹, waterfall=递归所有图片,无文件夹无文件名
    sort_by: filename / folder_filename / modified_at / file_size
    sort_order: asc / desc
    filter_filename: 文件名包含指定字符串
    filter_size_min/max: 文件大小范围（字节）
    filter_date_from/to: 修改日期范围（ISO 格式日期字符串 YYYY-MM-DD）
    """
    from sqlalchemy import func
    import time as _time
    from datetime import datetime as _dt

    # 规范化路径：去除首尾空格和斜杠
    path = (path or "").strip().strip("/")
    mode = "waterfall" if mode == "waterfall" else "folder"

    # 根据列数计算每页数量，确保能被整除
    per_page = _per_page_for_cols(cols)

    # 排序字段映射（folder_filename: 先按所在文件夹，再按图片名；filename/folder_filename 用自然排序）
    from sqlalchemy import case
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

    # ── 过滤条件 ──
    filter_filename = (filter_filename or "").strip()
    has_filters = False
    if filter_filename:
        escaped_fn = _escape_like(filter_filename)
        stmt = stmt.where(Image.filename.ilike(f"%{escaped_fn}%", escape="\\"))
        has_filters = True

    # 文件大小过滤（字节）
    _size_min = int(filter_size_min) if filter_size_min and filter_size_min.isdigit() else None
    _size_max = int(filter_size_max) if filter_size_max and filter_size_max.isdigit() else None
    if _size_min is not None:
        stmt = stmt.where(Image.file_size >= _size_min)
        has_filters = True
    if _size_max is not None:
        stmt = stmt.where(Image.file_size <= _size_max)
        has_filters = True

    # 修改日期过滤（YYYY-MM-DD → timestamp）
    _date_from_ts = None
    _date_to_ts = None
    if filter_date_from:
        try:
            _date_from_ts = _dt.strptime(filter_date_from, "%Y-%m-%d").timestamp()
        except ValueError:
            pass
    if filter_date_to:
        try:
            # 日期结束为当天 23:59:59
            _date_to_ts = _dt.strptime(filter_date_to, "%Y-%m-%d").timestamp() + 86399
        except ValueError:
            pass
    if _date_from_ts is not None:
        stmt = stmt.where(Image.modified_at >= _date_from_ts)
        has_filters = True
    if _date_to_ts is not None:
        stmt = stmt.where(Image.modified_at <= _date_to_ts)
        has_filters = True

    # 文件夹模式：仅当前层直接文件，不含子文件夹下的图片
    if mode == "folder":
        if path:
            escaped = _escape_like(path)
            direct_filter = ~Image.relative_path.like(f"{escaped}/%/%", escape="\\")
            stmt = stmt.where(direct_filter)
        else:
            stmt = stmt.where(~Image.relative_path.like("%/%"))

    # 总数（需要复制相同的过滤条件）
    count_stmt = select(func.count(Image.id))
    if path:
        count_stmt = count_stmt.where(path_filter)
    if search:
        count_stmt = count_stmt.where(Image.filename.ilike(f"%{search}%"))
    if filter_filename:
        count_stmt = count_stmt.where(Image.filename.ilike(f"%{_escape_like(filter_filename)}%", escape="\\"))
    if _size_min is not None:
        count_stmt = count_stmt.where(Image.file_size >= _size_min)
    if _size_max is not None:
        count_stmt = count_stmt.where(Image.file_size <= _size_max)
    if _date_from_ts is not None:
        count_stmt = count_stmt.where(Image.modified_at >= _date_from_ts)
    if _date_to_ts is not None:
        count_stmt = count_stmt.where(Image.modified_at <= _date_to_ts)
    if mode == "folder":
        if path:
            count_stmt = count_stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        else:
            count_stmt = count_stmt.where(~Image.relative_path.like("%/%"))
    total = (await session.execute(count_stmt)).scalar() or 0

    # 子文件夹（仅文件夹模式、首页且无搜索/过滤时），参与排序且排在文件前
    subfolders: list[dict] = []
    if mode == "folder" and page == 1 and not search and not has_filters:
        subfolders = await _get_subfolders(session, path, path_filter, sort_by, sort_order)

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
            "has_filters": has_filters,
            "cols": cols,
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
    session: AsyncSession = Depends(get_async_session),
):
    """获取当前文件夹/模式下的全部图片（用于大图浏览模式，无分页）"""
    from datetime import datetime as _dt

    path = (path or "").strip().strip("/")
    mode = "waterfall" if mode == "waterfall" else "folder"

    from sqlalchemy import case
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
        escaped = _escape_like(path)
        path_filter = (
            Image.relative_path.like(f"{escaped}/%", escape="\\")
            | (Image.relative_path == path)
        )
        stmt = stmt.where(path_filter)
    if mode == "folder":
        if path:
            escaped = _escape_like(path)
            stmt = stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        else:
            stmt = stmt.where(~Image.relative_path.like("%/%"))

    filter_filename = (filter_filename or "").strip()
    if filter_filename:
        stmt = stmt.where(Image.filename.ilike(f"%{_escape_like(filter_filename)}%", escape="\\"))
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

    result = await session.execute(stmt)
    images = list(result.scalars().all())
    return {
        "urls": ["/photos/" + img.relative_path for img in images],
        "ids": [img.id for img in images],
    }


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


def _format_file_size(size_bytes: int) -> str:
    """将字节数格式化为可读字符串，如 1.2 MB"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


@app.get("/api/image-info/{image_id:int}")
async def get_image_info(
    image_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """获取单张图片的详细信息，用于大图模式下的信息弹框"""
    from datetime import datetime

    result = await session.execute(select(Image).where(Image.id == image_id))
    img = result.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="图片不存在或已被删除")

    full_path = str((PHOTOS_DIR / img.relative_path).resolve())
    modified_dt = datetime.fromtimestamp(img.modified_at)
    modified_str = modified_dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "full_path": full_path,
        "filename": img.filename,
        "relative_path": img.relative_path,
        "resolution": f"{img.width} × {img.height}" if (img.width and img.height) else "—",
        "file_size": _format_file_size(img.file_size or 0),
        "modified_at": modified_str,
    }


@app.get("/api/scan-status")
async def get_scan_status():
    """返回当前是否有扫描任务在进行"""
    return {"scanning": is_scanning()}


@app.post("/scan")
async def trigger_scan():
    """手动触发扫描"""
    begin_scan()
    try:
        n = await scan_photos(PHOTOS_DIR, CACHE_DIR)
        return {"scanned": n}
    finally:
        end_scan()


@app.post("/api/cleanup")
async def trigger_cleanup():
    """手动触发数据库清理同步"""
    result = await cleanup_database(PHOTOS_DIR, CACHE_DIR)
    return result


@app.get("/api/stats")
async def get_stats(session: AsyncSession = Depends(get_async_session)):
    """获取数据库和文件系统统计信息"""
    from sqlalchemy import func

    # 数据库图片总数
    total_images = (await session.execute(select(func.count(Image.id)))).scalar() or 0
    # 数据库总文件大小
    total_size = (await session.execute(select(func.sum(Image.file_size)))).scalar() or 0

    # 文件夹数量（从文件系统统计）
    folder_count = 0
    for dirpath, dirnames, _ in os.walk(PHOTOS_DIR):
        # 排除隐藏文件夹
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        folder_count += len(dirnames)

    # 缓存文件数量和大小
    cache_count = 0
    cache_size = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            if f.suffix == ".webp":
                cache_count += 1
                cache_size += f.stat().st_size

    return {
        "total_images": total_images,
        "total_size": total_size,
        "folder_count": folder_count,
        "cache_count": cache_count,
        "cache_size": cache_size,
        "photos_dir": str(PHOTOS_DIR.resolve()),
        "cache_dir": str(CACHE_DIR.resolve()),
    }


@app.get("/api/search-dirs")
async def search_dirs(
    q: str = "",
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """全局目录搜索：模糊匹配 relative_path 的目录部分（不含文件名）。
    返回去重后的目录路径列表，每项含 path 和 image_count。"""
    q = (q or "").strip()
    if not q:
        return {"dirs": []}

    # 获取所有 relative_path
    result = await session.execute(select(Image.relative_path))
    all_paths = [r[0] for r in result.fetchall()]

    # 提取每条路径的目录部分（去掉文件名），搜集去重目录及其图片计数
    dir_counts: dict[str, int] = {}
    for rp in all_paths:
        parts = rp.rsplit("/", 1)
        if len(parts) == 2:
            dir_path = parts[0]
        else:
            dir_path = ""  # 根目录文件
        dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1

    # 同时包含所有中间路径（让父目录也可以搜到，计数为递归子图片数）
    full_dir_counts: dict[str, int] = {}
    for dir_path, count in dir_counts.items():
        if not dir_path:
            continue
        parts = dir_path.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            full_dir_counts[prefix] = full_dir_counts.get(prefix, 0) + count

    # 同时从文件系统扫描空文件夹
    def _scan_all_dirs(base: Path, prefix: str = ""):
        if not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                child_path = f"{prefix}/{child.name}" if prefix else child.name
                if child_path not in full_dir_counts:
                    full_dir_counts[child_path] = 0
                _scan_all_dirs(child, child_path)
    _scan_all_dirs(PHOTOS_DIR)

    # 模糊匹配：搜索词在目录路径中出现（不区分大小写）
    q_lower = q.lower()
    matched = []
    for dir_path, count in sorted(full_dir_counts.items()):
        if q_lower in dir_path.lower():
            matched.append({"path": dir_path, "image_count": count})
            if len(matched) >= limit:
                break

    return {"dirs": matched}


@app.get("/settings")
async def settings_page(request: Request):
    """设置页面"""
    return templates.TemplateResponse("settings.html", {"request": request})


# ---------- 删除 API ----------

class DeleteImagesRequest(BaseModel):
    ids: list[int]


class DeleteFoldersRequest(BaseModel):
    paths: list[str]


def _delete_image_files(relative_path: str) -> None:
    """删除图片的原始文件和缓存文件"""
    # 删除原图
    photo_path = PHOTOS_DIR / relative_path
    if photo_path.exists():
        photo_path.unlink(missing_ok=True)
    # 删除缓存缩略图
    cache_name = _cache_filename(relative_path)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        cache_path.unlink(missing_ok=True)


@app.post("/api/delete-images")
async def delete_images(
    body: DeleteImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """删除指定 ID 的图片（数据库记录 + 原图 + 缓存）"""
    if not body.ids:
        return {"deleted": 0}

    # 查出所有要删除的图片
    stmt = select(Image).where(Image.id.in_(body.ids))
    result = await session.execute(stmt)
    images = list(result.scalars().all())

    deleted = 0
    for img in images:
        _delete_image_files(img.relative_path)
        await session.delete(img)
        deleted += 1

    await session.commit()
    return {"deleted": deleted}


@app.get("/api/subfolders")
async def get_subfolders(
    path: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """获取指定路径下的直接子文件夹，用于移动目标选择"""
    path = (path or "").strip().strip("/")
    # 安全检查
    if ".." in path or path.startswith("/"):
        return {"subfolders": []}

    if path:
        escaped = _escape_like(path)
        path_filter = (
            Image.relative_path.like(f"{escaped}/%", escape="\\")
            | (Image.relative_path == path)
        )
    else:
        path_filter = None
    subfolders = await _get_subfolders(session, path, path_filter)
    return {
        "subfolders": [
            {"name": s["name"], "full_path": s["full_path"], "image_count": s["image_count"]}
            for s in subfolders
        ]
    }


class MoveImagesRequest(BaseModel):
    ids: list[int]
    target_path: str


@app.post("/api/move-images")
async def move_images(
    body: MoveImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定图片移动到目标文件夹"""
    from scanner import IMAGE_EXTENSIONS, _generate_thumbnail

    if not body.ids:
        return {"moved": 0, "errors": []}

    target_path = (body.target_path or "").strip().strip("/")
    if ".." in target_path or target_path.startswith("/"):
        return {"moved": 0, "errors": ["目标路径不合法"]}

    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    stmt = select(Image).where(Image.id.in_(body.ids))
    result = await session.execute(stmt)
    images = list(result.scalars().all())

    moved = 0
    errors = []
    for img in images:
        src_path = PHOTOS_DIR / img.relative_path
        if not src_path.exists():
            errors.append(f"{img.filename}: 文件不存在")
            continue

        ext = Path(img.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            errors.append(f"{img.filename}: 不支持的格式")
            continue

        # 目标路径：若已在目标目录则跳过
        new_rel = f"{target_path}/{img.filename}" if target_path else img.filename
        if new_rel == img.relative_path:
            continue  # 已在目标位置

        dest_path = target_dir / img.filename
        if dest_path.exists() and dest_path.resolve() != src_path.resolve():
            dest_path = _unique_dest(target_dir, img.filename, ext)
            new_rel = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")

        try:
            shutil.move(str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{img.filename}: {e}")
            continue

        # 删除旧缓存
        old_cache = CACHE_DIR / _cache_filename(img.relative_path)
        if old_cache.exists():
            old_cache.unlink(missing_ok=True)

        # 更新数据库
        img.relative_path = new_rel
        img.filename = dest_path.name
        img.filename_natural = natural_sort_key(dest_path.name)
        img.relative_path_natural = natural_sort_key(new_rel)
        img.modified_at = os.path.getmtime(dest_path)
        img.file_size = dest_path.stat().st_size

        # 生成新缓存
        new_cache = CACHE_DIR / _cache_filename(new_rel)
        _generate_thumbnail(dest_path, new_cache)

        session.add(img)
        moved += 1

    await session.commit()
    return {"moved": moved, "errors": errors}


@app.post("/api/delete-folders")
async def delete_folders(
    body: DeleteFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """删除指定文件夹路径下所有图片（数据库 + 文件系统），并删除文件夹目录"""
    if not body.paths:
        return {"deleted_images": 0, "deleted_folders": 0}

    total_images = 0
    total_folders = 0

    for folder_path in body.paths:
        folder_path = folder_path.strip().strip("/")
        if not folder_path:
            continue

        # 查询该路径下的所有图片（递归）
        escaped = _escape_like(folder_path)
        path_filter = (
            Image.relative_path.like(f"{escaped}/%", escape="\\")
            | (Image.relative_path == folder_path)
        )
        stmt = select(Image).where(path_filter)
        result = await session.execute(stmt)
        images = list(result.scalars().all())

        for img in images:
            _delete_image_files(img.relative_path)
            await session.delete(img)
            total_images += 1

        # 删除文件夹目录本身
        folder_fs_path = PHOTOS_DIR / folder_path
        if folder_fs_path.exists() and folder_fs_path.is_dir():
            shutil.rmtree(folder_fs_path, ignore_errors=True)
            total_folders += 1

    await session.commit()
    return {"deleted_images": total_images, "deleted_folders": total_folders}


# ---------- 创建文件夹 & 上传 API ----------

class CreateFolderRequest(BaseModel):
    path: str  # 当前所在路径
    name: str  # 新文件夹名


@app.post("/api/create-folder")
async def create_folder(body: CreateFolderRequest):
    """在指定路径下创建子文件夹"""
    parent = (body.path or "").strip().strip("/")
    name = body.name.strip().strip("/")
    if not name:
        return {"error": "文件夹名不能为空", "ok": False}
    # 安全检查：不允许 .. 和绝对路径
    if ".." in name or "/" in name or "\\" in name:
        return {"error": "文件夹名不合法", "ok": False}

    folder_path = PHOTOS_DIR / parent / name if parent else PHOTOS_DIR / name
    if folder_path.exists():
        return {"error": "文件夹已存在", "ok": False}

    folder_path.mkdir(parents=True, exist_ok=True)
    rel = f"{parent}/{name}" if parent else name
    print(f"[api] 创建文件夹: {rel}", flush=True)
    return {"ok": True, "path": rel}


@app.post("/api/upload")
async def upload_images(
    path: str = Form(""),
    on_duplicate: str = Form("skip"),
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_async_session),
):
    """上传图片到指定路径
    on_duplicate: skip=跳过重复, rename=重命名, overwrite=覆盖
    重复判断：同目录下文件内容 MD5 相同即视为重复
    写入文件后立即生成缩略图并入库，无需等待 watchdog。
    """
    import hashlib
    from PIL import Image as PILImage
    from scanner import IMAGE_EXTENSIONS, _generate_thumbnail

    target_path = (path or "").strip().strip("/")
    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 预计算目标目录中已有文件的哈希（用于内容去重）
    existing_hashes: dict[str, str] = {}  # md5 -> filename
    if target_dir.is_dir():
        for existing in target_dir.iterdir():
            if existing.is_file() and existing.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    h = hashlib.md5(existing.read_bytes()).hexdigest()
                    existing_hashes[h] = existing.name
                except OSError:
                    pass

    uploaded = 0
    skipped = 0
    errors = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            errors.append(f"{f.filename}: 不支持的格式 {ext}")
            continue

        try:
            content = await f.read()
        except Exception as e:
            errors.append(f"{f.filename}: 读取失败 {e}")
            continue

        # 内容哈希
        content_hash = hashlib.md5(content).hexdigest()
        is_overwrite = False

        # 检查内容是否已存在
        if content_hash in existing_hashes:
            if on_duplicate == "skip":
                skipped += 1
                continue
            elif on_duplicate == "overwrite":
                dest = target_dir / existing_hashes[content_hash]
                is_overwrite = True
            else:
                dest = _unique_dest(target_dir, f.filename, ext)
        else:
            safe_name = Path(f.filename).name
            dest = target_dir / safe_name
            if dest.exists():
                if on_duplicate == "skip":
                    skipped += 1
                    continue
                elif on_duplicate == "overwrite":
                    is_overwrite = True
                else:
                    dest = _unique_dest(target_dir, f.filename, ext)

        try:
            dest.write_bytes(content)
            existing_hashes[content_hash] = dest.name

            # ── 立即入库：生成缩略图 + 写数据库 ──
            rel_path = str(dest.relative_to(PHOTOS_DIR)).replace("\\", "/")

            # 检查是否已有记录（覆盖场景）
            existing_record = (await session.execute(
                select(Image).where(Image.relative_path == rel_path)
            )).scalar_one_or_none()

            try:
                with PILImage.open(dest) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0

            modified_at = os.path.getmtime(dest)
            file_size = os.path.getsize(dest)

            # 生成缩略图
            cache_name = _cache_filename(rel_path)
            cache_path = CACHE_DIR / cache_name
            _generate_thumbnail(dest, cache_path)

            if existing_record:
                # 更新已有记录
                existing_record.filename = dest.name
                existing_record.filename_natural = natural_sort_key(dest.name)
                existing_record.relative_path_natural = natural_sort_key(rel_path)
                existing_record.modified_at = modified_at
                existing_record.file_size = file_size
                existing_record.width = width
                existing_record.height = height
                session.add(existing_record)
            else:
                # 新增记录
                record = Image(
                    filename=dest.name,
                    relative_path=rel_path,
                    modified_at=modified_at,
                    file_size=file_size,
                    width=width,
                    height=height,
                    filename_natural=natural_sort_key(dest.name),
                    relative_path_natural=natural_sort_key(rel_path),
                )
                session.add(record)

            uploaded += 1
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")

    await session.commit()
    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


def _unique_dest(target_dir: Path, filename: str, ext: str) -> Path:
    """生成不冲突的文件路径"""
    safe_name = Path(filename).name
    dest = target_dir / safe_name
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    counter = 1
    while dest.exists():
        dest = target_dir / f"{stem}_{counter}{ext}"
        counter += 1
    return dest


# 静态目录挂载（放在路由之后，mount 会匹配前缀路径）
app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")
app.mount("/cache", StaticFiles(directory=str(CACHE_DIR)), name="cache")
