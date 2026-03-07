"""数据库相关测试。需先启动 docker compose -f docker-compose.dev.yml up -d"""

import pytest
from sqlalchemy import text

from app.models import Image, sync_engine


def test_models_import():
    """验证 models 可正常导入（MYSQL_HOST 由 conftest 设置）"""
    assert Image.__tablename__ == "images"


def test_db_connection():
    """验证可连接数据库并执行简单查询。MariaDB 未启动时跳过"""
    try:
        with sync_engine.connect() as conn:
            r = conn.execute(text("SELECT 1"))
            assert r.scalar() == 1
    except Exception as e:
        if "Connection refused" in str(e) or "2003" in str(e):
            pytest.skip("MariaDB 未启动，请执行: docker compose -f docker-compose.dev.yml up -d")
        raise
