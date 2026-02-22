"""应用公共对象：模板、供 main 和 routers 共享"""
from datetime import datetime
from urllib.parse import quote

from fastapi.templating import Jinja2Templates

from config import ROOT
from scanner import _cache_filename

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.filters["cache_key"] = _cache_filename
templates.env.filters["urlencode_path"] = lambda s: quote(s or "", safe="")


def _format_filesize(size: int) -> str:
    """将字节数格式化为可读大小，如 1.2 MB、500 KB"""
    if not size:
        return "0 B"
    if size >= 1048576:
        return f"{size / 1048576:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _format_date(ts: float) -> str:
    """将 Unix 时间戳格式化为 YYYY-MM-DD"""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return ""


templates.env.filters["format_filesize"] = _format_filesize
templates.env.filters["format_date"] = _format_date
