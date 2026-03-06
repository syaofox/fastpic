"""应用配置"""

import os
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).parent.parent
PHOTOS_DIR = ROOT / "photos"
CACHE_DIR = ROOT / "cache"
STATIC_DIR = ROOT / "app" / "static"
PER_PAGE = 24

# 数据库批量操作
IN_CLAUSE_BATCH_SIZE = 1000  # IN 子句分批，避免 max_allowed_packet
FOLDER_OP_BATCH_SIZE = 1000  # 按路径前缀分批加载
SCAN_DUPLICATES_BATCH_SIZE = 5000


def _parse_size(value: str | None, default_bytes: int) -> int:
    """解析大小字符串为字节数。支持纯数字或带单位：K/KB、M/MB、G/GB（不区分大小写）"""
    if not value or not str(value).strip():
        return default_bytes
    s = str(value).strip().upper()
    m = re.match(r"^(\d+)\s*(K|KB|M|MB|G|GB)?$", s)
    if not m:
        try:
            return int(s)
        except ValueError:
            return default_bytes
    num = int(m.group(1))
    unit = (m.group(2) or "").rstrip("B") or "B"
    if unit in ("K", ""):
        return num * 1024
    if unit == "M":
        return num * 1024 * 1024
    if unit == "G":
        return num * 1024 * 1024 * 1024
    return num


# 上传限制（字节），可通过环境变量覆盖，支持 1000M、5000M 等格式
_MAX_FILE = os.environ.get("MAX_UPLOAD_FILE_SIZE", "100M")
_MAX_TOTAL = os.environ.get("MAX_UPLOAD_TOTAL_SIZE", "500M")
MAX_UPLOAD_FILE_SIZE = _parse_size(_MAX_FILE, 100 * 1024 * 1024)
MAX_UPLOAD_TOTAL_SIZE = _parse_size(_MAX_TOTAL, 500 * 1024 * 1024)

ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()
SESSION_TOKEN = secrets.token_hex(32) if ACCESS_PASSWORD else ""

# 启动时跳过全量 os.walk 扫描，仅做 DB 校验（移除已删除文件的幽灵记录）
# 适用于「几乎无新增文件」的日常使用，可显著加快启动
_SKIP_FULL_SCAN = os.environ.get("SKIP_FULL_SCAN_ON_STARTUP", "").strip().lower()
SKIP_FULL_SCAN_ON_STARTUP = _SKIP_FULL_SCAN in ("1", "true", "yes")


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
