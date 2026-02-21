"""应用公共对象：模板、供 main 和 routers 共享"""
from urllib.parse import quote

from fastapi.templating import Jinja2Templates

from config import ROOT
from scanner import _cache_filename

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.filters["cache_key"] = _cache_filename
templates.env.filters["urlencode_path"] = lambda s: quote(s or "", safe="")
