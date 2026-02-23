"""路径工具：SQL LIKE 转义、路径校验、路径过滤条件"""
from pathlib import Path

# MariaDB 不支持 ESCAPE '\\' 会报语法错误，使用 !
LIKE_ESCAPE = "!"


def escape_like(value: str) -> str:
    """转义 SQL LIKE 中的 % 和 _"""
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def normalize_path(path: str | None, allow_empty: bool = True) -> str | None:
    """规范化相对路径，非法返回 None。

    - strip 空白和首尾斜杠
    - 含 .. 或以 / 开头视为非法
    - allow_empty=False 时空路径返回 None
    """
    p = (path or "").strip().strip("/")
    if not p and not allow_empty:
        return None
    if ".." in p or p.startswith("/"):
        return None
    return p


def resolve_and_validate_relative_path(
    relative_path: str, photos_dir: Path
) -> Path | None:
    """校验 relative_path 在 photos_dir 下，返回绝对路径或 None"""
    rel = (relative_path or "").strip().strip("/")
    if not rel or ".." in rel or rel.startswith("/"):
        return None
    full = (photos_dir / rel).resolve()
    try:
        full.relative_to(photos_dir.resolve())
    except ValueError:
        return None
    return full if full.is_file() else None


def path_filter_for_prefix(relative_path_column, prefix: str, include_children: bool = True):
    """生成 SQLAlchemy 路径过滤条件：匹配 prefix 及其子路径下的图片。

    relative_path_column: 如 Image.relative_path
    prefix: 路径前缀，如 "2024/01"
    include_children: True 时匹配 prefix 及 prefix/xxx，False 时仅匹配 prefix 本身
    """
    escaped = escape_like(prefix)
    if include_children:
        return (
            relative_path_column.like(f"{escaped}/%", escape=LIKE_ESCAPE)
            | (relative_path_column == prefix)
        )
    return relative_path_column == prefix
