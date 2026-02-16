import asyncio
import hmac
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import quote

import tempfile
import zipfile
from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Image, Tag, ImageTag, init_db, get_async_session, natural_sort_key
from scanner import scan_photos, scan_videos, cleanup_database, _cache_filename
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
        # 再扫描新增图片和视频
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


# ── 搜索框：简繁 + 拼音匹配（懒加载） ──
_opencc_s2t = None
_opencc_t2s = None


def _get_opencc_s2t():
    global _opencc_s2t
    if _opencc_s2t is None:
        try:
            from opencc import OpenCC
            _opencc_s2t = OpenCC("s2t")
        except Exception:
            _opencc_s2t = False
    return _opencc_s2t


def _get_opencc_t2s():
    global _opencc_t2s
    if _opencc_t2s is None:
        try:
            from opencc import OpenCC
            _opencc_t2s = OpenCC("t2s")
        except Exception:
            _opencc_t2s = False
    return _opencc_t2s


def _to_simplified(s: str) -> str:
    """转为简体（用于匹配）"""
    cc = _get_opencc_t2s()
    if not cc:
        return s
    try:
        return cc.convert(s)
    except Exception:
        return s


def _to_traditional(s: str) -> str:
    """转为繁体（用于匹配）"""
    cc = _get_opencc_s2t()
    if not cc:
        return s
    try:
        return cc.convert(s)
    except Exception:
        return s


def _to_pinyin_lower(s: str) -> str:
    """将中文转为小写无声调拼音并拼接（如 厦门 -> xiamen），非中文保留原样并转小写。"""
    try:
        from pypinyin import lazy_pinyin, Style
        parts = lazy_pinyin(s, style=Style.NORMAL)
        return "".join(p).lower() if (p := [x.strip() for x in parts if x]) else s.lower()
    except Exception:
        return s.lower()


def _search_match(query: str, target: str) -> bool:
    """判断 query 是否匹配 target：支持模糊、简繁、拼音。
    - 原始/小写模糊
    - 简体/繁体互相匹配
    - 拼音匹配（如 xiamen 匹配 厦门）
    """
    if not query or not target:
        return False
    q = query.strip()
    t = target
    # 1) 原始模糊（不区分大小写）
    if q.lower() in t.lower():
        return True
    # 2) 简繁
    try:
        q_s, q_t = _to_simplified(q), _to_traditional(q)
        t_s, t_t = _to_simplified(t), _to_traditional(t)
        if q_s and q_s in t_s:
            return True
        if q_t and q_t in t_t:
            return True
    except Exception:
        pass
    # 3) 拼音：把目标转成拼音再匹配（支持输入拼音搜中文路径）
    try:
        t_py = _to_pinyin_lower(t)
        q_lower = q.lower()
        if q_lower in t_py:
            return True
        # 若 query 含中文，也把 query 转拼音，看是否在 target 的拼音里
        if any("\u4e00" <= c <= "\u9fff" for c in q):
            q_py = _to_pinyin_lower(q)
            if q_py and q_py in t_py:
                return True
    except Exception:
        pass
    return False


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


# ── 文件夹树短期缓存（60 秒），创建/删除文件夹时失效 ──
_FOLDER_TREE_CACHE_TTL = 60.0
_folder_tree_cache: dict | None = None
_folder_tree_cache_lock = asyncio.Lock()


def _invalidate_folder_tree_cache() -> None:
    """创建/删除文件夹后调用，使缓存失效"""
    global _folder_tree_cache
    _folder_tree_cache = None


async def _get_folder_tree_cached(rel_paths: list[str]) -> tuple[list[list[str]], dict, dict[str, int]]:
    """获取 folder_tree、nested_tree、folder_counts，带 60 秒缓存"""
    global _folder_tree_cache
    async with _folder_tree_cache_lock:
        now = time.monotonic()
        if _folder_tree_cache is not None:
            ts = _folder_tree_cache.get("ts", 0)
            if now - ts < _FOLDER_TREE_CACHE_TTL:
                return (
                    _folder_tree_cache["folder_tree"],
                    _folder_tree_cache["nested_tree"],
                    _folder_tree_cache["folder_counts"],
                )
        folder_tree = await asyncio.to_thread(_get_folder_tree, rel_paths)
        nested_tree = _build_nested_tree(folder_tree)
        folder_counts = _compute_folder_counts(rel_paths)
        _folder_tree_cache = {
            "ts": now,
            "folder_tree": folder_tree,
            "nested_tree": nested_tree,
            "folder_counts": folder_counts,
        }
        return folder_tree, nested_tree, folder_counts


@app.get("/")
async def index(request: Request, session: AsyncSession = Depends(get_async_session)):
    """返回主页框架"""
    from sqlalchemy import func

    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree, nested_tree, folder_counts = await _get_folder_tree_cached(rel_paths)

    # 标签列表（按使用数量降序）
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
    """返回侧栏文件夹树 HTML 片段，用于无刷新更新"""
    path = request.query_params.get("path", "")
    result = await session.execute(select(Image.relative_path))
    rel_paths = [r[0] for r in result.fetchall()]
    folder_tree, nested_tree, folder_counts = await _get_folder_tree_cached(rel_paths)
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
    子文件夹按 sort_by/sort_order 参与排序。
    优化：用 1 次批量查询替代 N*2 次查询，避免子目录多时卡顿。"""
    path_depth = len(path.split("/")) if path else 0
    path_prefix = path + "/" if path else ""

    # 1) 一次查询获取所有子文件夹下的图片（relative_path, modified_at, file_size）
    # 条件：路径在 path 下且至少有一层子目录
    batch_stmt = select(Image.relative_path, Image.modified_at, Image.file_size)
    if path:
        escaped = _escape_like(path_prefix)
        batch_stmt = batch_stmt.where(
            Image.relative_path.like(f"{escaped}%", escape="\\"),
            Image.relative_path.like(f"{escaped}%/%", escape="\\"),
        )
    else:
        batch_stmt = batch_stmt.where(Image.relative_path.like("%/%"))
    result = await session.execute(batch_stmt)
    rows = result.fetchall()

    # 2) 在 Python 中按子文件夹分组，计算 count、max、取前 4 张缩略图
    subfolder_data: dict[str, list[tuple[str, float, int]]] = {}
    for rel, mod, size in rows:
        parts = rel.split("/")
        if len(parts) <= path_depth:
            continue
        sub_name = parts[path_depth]
        if sub_name not in subfolder_data:
            subfolder_data[sub_name] = []
        subfolder_data[sub_name].append((rel, mod or 0.0, size or 0))

    # 3) 从文件系统补充空文件夹
    fs_dir = PHOTOS_DIR / path if path else PHOTOS_DIR
    if fs_dir.is_dir():
        for child in fs_dir.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                if child.name not in subfolder_data:
                    subfolder_data[child.name] = []

    # 4) 构建 subfolders 列表
    subfolders: list[dict] = []
    for name, items in subfolder_data.items():
        full_path = f"{path}/{name}" if path else name
        if items:
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            count = len(items)
            max_mod = items_sorted[0][1] if items_sorted else 0.0
            max_size = max((x[2] for x in items), default=0)
            thumbnails = [x[0] for x in items_sorted[:4]]
        else:
            count = 0
            max_mod = 0.0
            max_size = 0
            thumbnails = []
        subfolders.append({
            "name": name,
            "full_path": full_path,
            "thumbnails": thumbnails,
            "image_count": count,
            "_sort_key_filename": natural_sort_key(name),
            "_sort_key_folder_filename": natural_sort_key(full_path),
            "_sort_key_modified_at": max_mod,
            "_sort_key_file_size": max_size,
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
    filter_tag: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片网格 HTML 片段（供 HTMX 调用）
    mode: folder=仅当前层文件+子文件夹, waterfall=递归所有图片,无文件夹无文件名
    sort_by: filename / folder_filename / modified_at / file_size
    sort_order: asc / desc
    filter_filename: 文件名包含指定字符串
    filter_size_min/max: 文件大小范围（字节）
    filter_date_from/to: 修改日期范围（ISO 格式日期字符串 YYYY-MM-DD）
    filter_tag: 按标签过滤
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

    # 标签过滤
    filter_tag = (filter_tag or "").strip()
    if filter_tag:
        stmt = stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == filter_tag)
        has_filters = True

    # 文件夹模式：仅当前层直接文件，不含子文件夹下的图片
    # 例外：按标签筛选且 path 为空时，显示所有带该标签的图片（否则根目录下通常无图）
    if mode == "folder":
        if path:
            escaped = _escape_like(path)
            direct_filter = ~Image.relative_path.like(f"{escaped}/%/%", escape="\\")
            stmt = stmt.where(direct_filter)
        elif not filter_tag:
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
    if filter_tag:
        count_stmt = count_stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == filter_tag)
    if mode == "folder":
        if path:
            count_stmt = count_stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape="\\"))
        elif not filter_tag:
            count_stmt = count_stmt.where(~Image.relative_path.like("%/%"))
    total = (await session.execute(count_stmt)).scalar() or 0

    # 子文件夹（仅文件夹模式、首页且无搜索/过滤时），参与排序且排在文件前
    subfolders: list[dict] = []
    if mode == "folder" and page == 1 and not search and not has_filters and not filter_tag:
        subfolders = await _get_subfolders(session, path, path_filter, sort_by, sort_order)

    # 分页
    offset = (page - 1) * per_page
    stmt = stmt.offset(offset).limit(per_page + 1)
    result = await session.execute(stmt)
    images = list(result.scalars().all())
    has_next = len(images) > per_page
    if has_next:
        images = images[:per_page]

    # 批量查询当前页图片的标签
    image_tags_map: dict[int, list[str]] = {}
    if images:
        image_ids = [img.id for img in images if img.id]
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

    filter_tag = (filter_tag or "").strip()
    if filter_tag:
        stmt = stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == filter_tag)

    result = await session.execute(stmt)
    images = list(result.scalars().all())
    return {
        "urls": ["/photos/" + img.relative_path for img in images],
        "ids": [img.id for img in images],
        "media_types": [getattr(img, "media_type", "image") for img in images],
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

    # 查询该图片的标签
    tag_result = await session.execute(
        select(Tag.name)
        .join(ImageTag, ImageTag.tag_id == Tag.id)
        .where(ImageTag.image_id == image_id)
        .order_by(Tag.name)
    )
    tags = [r[0] for r in tag_result.fetchall()]

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
        "tags": tags,
    }


# ── 标签 API ──


class AddTagsRequest(BaseModel):
    tags: list[str]


@app.get("/api/tags")
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
        q_escaped = _escape_like(q.strip())
        stmt = stmt.where(Tag.name.ilike(f"%{q_escaped}%", escape="\\"))
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    rows = result.fetchall()
    return {"tags": [{"name": r[0], "count": r[1] or 0} for r in rows]}


@app.get("/api/images/{image_id:int}/tags")
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


@app.post("/api/images/{image_id:int}/tags")
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
        # 获取或创建 Tag
        tag_result = await session.execute(select(Tag).where(Tag.name == name))
        tag = tag_result.scalar_one_or_none()
        if not tag:
            tag = Tag(name=name)
            session.add(tag)
            await session.flush()
        # 检查是否已存在
        existing = await session.execute(
            select(ImageTag).where(ImageTag.image_id == image_id, ImageTag.tag_id == tag.id)
        )
        if existing.scalar_one_or_none() is None:
            session.add(ImageTag(image_id=image_id, tag_id=tag.id))
            added += 1
    await session.commit()
    return {"added": added}


@app.delete("/api/images/{image_id:int}/tags/{tag_name:str}")
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


@app.get("/api/scan-status")
async def get_scan_status():
    """返回当前是否有扫描任务在进行"""
    return {"scanning": is_scanning()}


@app.post("/scan")
async def trigger_scan():
    """手动触发扫描"""
    begin_scan()
    try:
        n_img = await scan_photos(PHOTOS_DIR, CACHE_DIR)
        n_vid = await scan_videos(PHOTOS_DIR, CACHE_DIR)
        return {"scanned": n_img + n_vid, "images": n_img, "videos": n_vid}
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

    # 模糊匹配 + 简繁匹配 + 拼音匹配
    matched = []
    for dir_path, count in sorted(full_dir_counts.items()):
        if _search_match(q, dir_path):
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


# ---------- 下载 API ----------


def _resolve_and_validate_relative_path(relative_path: str) -> Path | None:
    """校验 relative_path 在 PHOTOS_DIR 下，返回绝对路径或 None"""
    rel = (relative_path or "").strip().strip("/")
    if not rel or ".." in rel or rel.startswith("/"):
        return None
    full = (PHOTOS_DIR / rel).resolve()
    try:
        full.relative_to(PHOTOS_DIR.resolve())
    except ValueError:
        return None
    return full if full.is_file() else None


@app.get("/api/download/image")
async def download_image(
    id: int | None = None,
    relative_path: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """单图下载：按 id 或 relative_path 返回原图，Content-Disposition: attachment"""
    if id is not None:
        result = await session.execute(select(Image).where(Image.id == id))
        img = result.scalar_one_or_none()
        if not img:
            raise HTTPException(status_code=404, detail="图片不存在")
        rel = img.relative_path
        filename = img.filename
    elif relative_path:
        full = _resolve_and_validate_relative_path(relative_path)
        if not full:
            raise HTTPException(status_code=400, detail="路径不合法或文件不存在")
        rel = relative_path.strip().strip("/")
        filename = full.name
    else:
        raise HTTPException(status_code=400, detail="请提供 id 或 relative_path")
    file_path = PHOTOS_DIR / rel
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
    )


class DownloadZipRequest(BaseModel):
    image_ids: list[int] = []
    folder_paths: list[str] = []


@app.post("/api/download/zip")
async def download_zip(
    body: DownloadZipRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """批量下载：将选中的图片 ID 和文件夹路径（含子目录）打包为 ZIP"""
    rel_paths: set[str] = set()

    if body.image_ids:
        result = await session.execute(select(Image.relative_path).where(Image.id.in_(body.image_ids)))
        for row in result.fetchall():
            rel_paths.add(row[0])

    for raw_path in body.folder_paths or []:
        path = (raw_path or "").strip().strip("/")
        if not path or ".." in path or path.startswith("/"):
            continue
        escaped = _escape_like(path)
        path_filter = (
            Image.relative_path.like(f"{escaped}/%", escape="\\")
            | (Image.relative_path == path)
        )
        result = await session.execute(select(Image.relative_path).where(path_filter))
        for row in result.fetchall():
            rel_paths.add(row[0])

    # 只保留磁盘上存在的文件
    existing = []
    for rp in rel_paths:
        full = PHOTOS_DIR / rp
        if full.is_file():
            existing.append(rp)

    if not existing:
        raise HTTPException(status_code=400, detail="没有可下载的文件")

    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rp in existing:
                full = PHOTOS_DIR / rp
                if full.is_file():
                    zf.write(full, rp)
        background_tasks.add_task(os.unlink, tmp_path)
        return FileResponse(
            path=tmp_path,
            filename="download.zip",
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="download.zip"'},
        )
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail="打包下载失败")


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


def _unique_folder_dest(target_dir: Path, folder_name: str) -> Path:
    """生成不冲突的文件夹路径"""
    dest = target_dir / folder_name
    if not dest.exists():
        return dest
    counter = 1
    while dest.exists():
        dest = target_dir / f"{folder_name} ({counter})"
        counter += 1
    return dest


class MoveFoldersRequest(BaseModel):
    paths: list[str]
    target_path: str


@app.post("/api/move-folders")
async def move_folders(
    body: MoveFoldersRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """将指定文件夹（含子文件夹和图片）移动到目标父目录"""
    from scanner import IMAGE_EXTENSIONS, _generate_thumbnail

    if not body.paths:
        return {"moved": 0, "errors": []}

    target_path = (body.target_path or "").strip().strip("/")
    if ".." in target_path or target_path.startswith("/"):
        return {"moved": 0, "errors": ["目标路径不合法"]}

    target_dir = PHOTOS_DIR / target_path if target_path else PHOTOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    errors = []

    for folder_path in body.paths:
        folder_path = folder_path.strip().strip("/")
        if not folder_path:
            continue

        # 校验：目标不能是源路径本身或其子路径
        if target_path == folder_path or target_path.startswith(folder_path + "/"):
            errors.append(f"{folder_path}: 不能移动到自身或子文件夹内")
            continue

        folder_name = Path(folder_path).name
        would_be_path = f"{target_path}/{folder_name}" if target_path else folder_name
        if would_be_path == folder_path:
            continue  # 已在目标位置

        src_path = PHOTOS_DIR / folder_path
        if not src_path.exists() or not src_path.is_dir():
            errors.append(f"{folder_path}: 文件夹不存在")
            continue

        dest_path = _unique_folder_dest(target_dir, folder_name)
        new_prefix = str(dest_path.relative_to(PHOTOS_DIR)).replace("\\", "/")

        if src_path.resolve() == dest_path.resolve():
            continue  # 已在目标位置

        try:
            shutil.move(str(src_path), str(dest_path))
        except OSError as e:
            errors.append(f"{folder_path}: {e}")
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
            # 计算新相对路径
            if img.relative_path == folder_path:
                # 罕见：图片直接在文件夹路径下（应无此情况）
                suffix = ""
            else:
                suffix = img.relative_path[len(folder_path) :]
            new_rel = new_prefix + suffix

            # 删除旧缓存
            old_cache = CACHE_DIR / _cache_filename(img.relative_path)
            if old_cache.exists():
                old_cache.unlink(missing_ok=True)

            # 更新数据库
            img.relative_path = new_rel
            img.filename = Path(new_rel).name
            img.filename_natural = natural_sort_key(img.filename)
            img.relative_path_natural = natural_sort_key(new_rel)
            new_full = dest_path / suffix.lstrip("/") if suffix else dest_path
            if new_full.exists() and new_full.is_file():
                img.modified_at = os.path.getmtime(new_full)
                img.file_size = os.path.getsize(new_full)
                new_cache = CACHE_DIR / _cache_filename(new_rel)
                _generate_thumbnail(new_full, new_cache)

            session.add(img)
            moved += 1

        print(f"[api] 移动文件夹: {folder_path} → {new_prefix}", flush=True)

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
    if total_folders > 0:
        _invalidate_folder_tree_cache()
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
    _invalidate_folder_tree_cache()
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
