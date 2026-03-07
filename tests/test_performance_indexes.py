"""数据库性能索引测试。需先启动 docker compose -f docker-compose.dev.yml up -d"""

import pytest
from sqlalchemy import text

from app.models import _run_performance_indexes_migration, sync_engine


def check_db_connection():
    """检查数据库连接是否可用"""
    try:
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        if "Connection refused" in str(e) or "2003" in str(e) or "1045" in str(e):
            return False
        raise


def skip_if_no_db():
    """数据库不可用时跳过测试"""
    if not check_db_connection():
        pytest.skip("MariaDB 未启动或无法访问，请执行: docker compose -f docker-compose.dev.yml up -d")


def test_performance_indexes_exist():
    """验证性能优化索引已创建"""
    skip_if_no_db()
    _run_performance_indexes_migration()

    expected_indexes = [
        ("ix_images_path_media", "images"),
        ("ix_image_tags_tag_id", "image_tags"),
        ("ix_folder_thumbnails_folder", "folder_thumbnails"),
        ("ix_images_mod_size", "images"),
        ("ix_images_filename_media", "images"),
    ]

    with sync_engine.connect() as conn:
        for index_name, table_name in expected_indexes:
            r = conn.execute(
                text(
                    f"SELECT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS "
                    f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{table_name}' "
                    f"AND INDEX_NAME = '{index_name}'"
                )
            )
            result = r.fetchone()
            assert result is not None, f"索引 {index_name} 不存在于 {table_name} 表"


def test_images_path_media_index():
    """验证 ix_images_path_media 索引包含正确字段"""
    skip_if_no_db()
    with sync_engine.connect() as conn:
        r = conn.execute(
            text(
                "SELECT COLUMN_NAME, SEQ_IN_INDEX FROM INFORMATION_SCHEMA.STATISTICS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' "
                "AND INDEX_NAME = 'ix_images_path_media' "
                "ORDER BY SEQ_IN_INDEX"
            )
        )
        columns = [row[0] for row in r.fetchall()]
        assert "relative_path" in columns
        assert "media_type" in columns


def test_images_mod_size_index():
    """验证 ix_images_mod_size 索引用于覆盖 COUNT 查询"""
    skip_if_no_db()
    with sync_engine.connect() as conn:
        r = conn.execute(
            text(
                "SELECT COLUMN_NAME, SEQ_IN_INDEX FROM INFORMATION_SCHEMA.STATISTICS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' "
                "AND INDEX_NAME = 'ix_images_mod_size' "
                "ORDER BY SEQ_IN_INDEX"
            )
        )
        columns = [row[0] for row in r.fetchall()]
        assert "modified_at" in columns
        assert "file_size" in columns


def test_folder_thumbnails_index():
    """验证 ix_folder_thumbnails_folder 索引字段顺序"""
    skip_if_no_db()
    with sync_engine.connect() as conn:
        r = conn.execute(
            text(
                "SELECT COLUMN_NAME, SEQ_IN_INDEX FROM INFORMATION_SCHEMA.STATISTICS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'folder_thumbnails' "
                "AND INDEX_NAME = 'ix_folder_thumbnails_folder' "
                "ORDER BY SEQ_IN_INDEX"
            )
        )
        columns = [row[0] for row in r.fetchall()]
        assert "folder_path" in columns
        assert "display_order" in columns
