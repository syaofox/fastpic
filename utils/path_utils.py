"""路径工具：SQL LIKE 转义、路径校验、路径过滤条件"""
from pathlib import Path


def escape_like(value: str) -> str:
    """转义 SQL LIKE 中的 % 和 _，避免被当作通配符"""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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
            relative_path_column.like(f"{escaped}/%", escape="\\")
            | (relative_path_column == prefix)
        )
    return relative_path_column == prefix
