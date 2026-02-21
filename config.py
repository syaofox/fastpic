"""应用配置"""
import os
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).parent
PHOTOS_DIR = ROOT / "photos"
CACHE_DIR = ROOT / "cache"
STATIC_DIR = ROOT / "static"
PER_PAGE = 24

ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()
SESSION_TOKEN = secrets.token_hex(32) if ACCESS_PASSWORD else ""


def get_version() -> str:
    """从 pyproject.toml 读取版本号"""
    pyproject_path = ROOT / "pyproject.toml"
    if pyproject_path.exists():
        text = pyproject_path.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            return m.group(1)
    return "unknown"


APP_VERSION = get_version()

PHOTOS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
