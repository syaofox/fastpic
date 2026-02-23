"""图片查询：排序列、过滤器构建"""
from datetime import datetime as _dt

from sqlmodel import select
from sqlalchemy import case, func, text

from models import Image, ImageTag, Tag
from utils.path_utils import escape_like, LIKE_ESCAPE, path_filter_for_prefix


def _fulltext_search_condition(search: str):
    """FULLTEXT 搜索条件，需 images.filename 有 FULLTEXT 索引。
    短词（<3 字符）或含通配符时回退到 ilike，因 FULLTEXT 可能不索引短词。"""
    q = (search or "").strip()
    if not q or len(q) < 3 or "%" in q or "_" in q:
        return None
    return text("MATCH(images.filename) AGAINST(:ft_q IN NATURAL LANGUAGE MODE)").bindparams(ft_q=q)

IMAGE_SORT_COLUMNS = {
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


def get_sort_column(sort_by: str):
    """获取排序列，默认 modified_at"""
    return IMAGE_SORT_COLUMNS.get(sort_by, Image.modified_at)


def parse_filter_params(
    filter_filename: str = "",
    filter_size_min: str = "",
    filter_size_max: str = "",
    filter_date_from: str = "",
    filter_date_to: str = "",
    filter_tag: str = "",
) -> dict:
    """解析过滤参数，返回用于 apply_image_filters 的字典"""
    _size_min = int(filter_size_min) if filter_size_min and filter_size_min.isdigit() else None
    _size_max = int(filter_size_max) if filter_size_max and filter_size_max.isdigit() else None
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
    return {
        "filter_filename": (filter_filename or "").strip(),
        "_size_min": _size_min,
        "_size_max": _size_max,
        "_date_from_ts": _date_from_ts,
        "_date_to_ts": _date_to_ts,
        "filter_tag": (filter_tag or "").strip(),
    }


def apply_image_filters(
    stmt,
    path: str,
    search: str,
    mode: str,
    parsed: dict,
) -> tuple:
    """对 select 语句应用图片过滤条件。

    返回 (stmt, pf, has_filters)
    - stmt: 应用了 where 条件的语句
    - pf: path 对应的 path_filter 子句（用于 count_stmt 和 get_subfolders），path 为空时为 None
    - has_filters: 是否应用了任意过滤条件
    """
    pf = None
    has_filters = False

    if path:
        pf = path_filter_for_prefix(Image.relative_path, path)
        stmt = stmt.where(pf)
    if search:
        ft = _fulltext_search_condition(search)
        if ft is not None:
            stmt = stmt.where(ft)
        else:
            stmt = stmt.where(Image.filename.ilike(f"%{escape_like(search)}%", escape=LIKE_ESCAPE))

    fn = parsed["filter_filename"]
    if fn:
        stmt = stmt.where(Image.filename.ilike(f"%{escape_like(fn)}%", escape=LIKE_ESCAPE))
        has_filters = True
    if parsed["_size_min"] is not None:
        stmt = stmt.where(Image.file_size >= parsed["_size_min"])
        has_filters = True
    if parsed["_size_max"] is not None:
        stmt = stmt.where(Image.file_size <= parsed["_size_max"])
        has_filters = True
    if parsed["_date_from_ts"] is not None:
        stmt = stmt.where(Image.modified_at >= parsed["_date_from_ts"])
        has_filters = True
    if parsed["_date_to_ts"] is not None:
        stmt = stmt.where(Image.modified_at <= parsed["_date_to_ts"])
        has_filters = True
    if parsed["filter_tag"]:
        stmt = stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == parsed["filter_tag"])
        has_filters = True

    if mode in ("folder", "list"):
        if path:
            escaped = escape_like(path)
            stmt = stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape=LIKE_ESCAPE))
        elif not parsed["filter_tag"]:
            stmt = stmt.where(~Image.relative_path.like("%/%"))

    return stmt, pf, has_filters


def apply_image_filters_to_count(count_stmt, path: str, search: str, mode: str, parsed: dict, pf):
    """对 count 语句应用与 apply_image_filters 相同的过滤条件"""
    if path and pf is not None:
        count_stmt = count_stmt.where(pf)
    if search:
        ft = _fulltext_search_condition(search)
        if ft is not None:
            count_stmt = count_stmt.where(ft)
        else:
            count_stmt = count_stmt.where(
                Image.filename.ilike(f"%{escape_like(search)}%", escape=LIKE_ESCAPE)
            )
    if parsed["filter_filename"]:
        count_stmt = count_stmt.where(
            Image.filename.ilike(f"%{escape_like(parsed['filter_filename'])}%", escape=LIKE_ESCAPE)
        )
    if parsed["_size_min"] is not None:
        count_stmt = count_stmt.where(Image.file_size >= parsed["_size_min"])
    if parsed["_size_max"] is not None:
        count_stmt = count_stmt.where(Image.file_size <= parsed["_size_max"])
    if parsed["_date_from_ts"] is not None:
        count_stmt = count_stmt.where(Image.modified_at >= parsed["_date_from_ts"])
    if parsed["_date_to_ts"] is not None:
        count_stmt = count_stmt.where(Image.modified_at <= parsed["_date_to_ts"])
    if parsed["filter_tag"]:
        count_stmt = count_stmt.join(ImageTag, ImageTag.image_id == Image.id).join(
            Tag, Tag.id == ImageTag.tag_id
        ).where(Tag.name == parsed["filter_tag"])
    if mode in ("folder", "list"):
        if path:
            escaped = escape_like(path)
            count_stmt = count_stmt.where(~Image.relative_path.like(f"{escaped}/%/%", escape=LIKE_ESCAPE))
        elif not parsed["filter_tag"]:
            count_stmt = count_stmt.where(~Image.relative_path.like("%/%"))
    return count_stmt
