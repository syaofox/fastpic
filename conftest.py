"""pytest 配置：数据库测试使用本地 MariaDB (127.0.0.1:3306)。

运行前需启动开发数据库：
  docker compose -f docker-compose.dev.yml up -d

然后执行：
  uv sync --extra dev
  uv run pytest
"""
import os

# 在导入任何应用代码前设置 MYSQL_HOST，便于数据库测试
if "MYSQL_HOST" not in os.environ or not os.environ.get("MYSQL_HOST", "").strip():
    os.environ.setdefault("MYSQL_HOST", "127.0.0.1")
