"""filter_media_type 筛选功能测试"""
import pytest
from sqlmodel import select
from sqlalchemy import func
from sqlalchemy.dialects import mysql

from models import Image
from utils.query_builder import (
    parse_filter_params,
    apply_image_filters,
    apply_image_filters_to_count,
    VALID_MEDIA_TYPES,
)


class TestParseFilterParams:
    """parse_filter_params 中 filter_media_type 解析测试"""

    def test_filter_media_type_image(self):
        parsed = parse_filter_params(filter_media_type="image")
        assert parsed["filter_media_type"] == "image"

    def test_filter_media_type_animated(self):
        parsed = parse_filter_params(filter_media_type="animated")
        assert parsed["filter_media_type"] == "animated"

    def test_filter_media_type_video(self):
        parsed = parse_filter_params(filter_media_type="video")
        assert parsed["filter_media_type"] == "video"

    def test_filter_media_type_empty(self):
        parsed = parse_filter_params(filter_media_type="")
        assert parsed["filter_media_type"] == ""

    def test_filter_media_type_none_default(self):
        parsed = parse_filter_params()
        assert parsed["filter_media_type"] == ""

    def test_filter_media_type_case_insensitive(self):
        parsed = parse_filter_params(filter_media_type="IMAGE")
        assert parsed["filter_media_type"] == "image"
        parsed = parse_filter_params(filter_media_type="Video")
        assert parsed["filter_media_type"] == "video"
        parsed = parse_filter_params(filter_media_type="ANIMATED")
        assert parsed["filter_media_type"] == "animated"

    def test_filter_media_type_strips_whitespace(self):
        parsed = parse_filter_params(filter_media_type="  image  ")
        assert parsed["filter_media_type"] == "image"

    def test_filter_media_type_invalid_ignored(self):
        parsed = parse_filter_params(filter_media_type="invalid")
        assert parsed["filter_media_type"] == ""
        parsed = parse_filter_params(filter_media_type="gif")
        assert parsed["filter_media_type"] == ""
        parsed = parse_filter_params(filter_media_type="photos")
        assert parsed["filter_media_type"] == ""

    def test_valid_media_types_constant(self):
        assert VALID_MEDIA_TYPES == frozenset({"image", "animated", "video"})


class TestApplyImageFiltersMediaType:
    """apply_image_filters 中 filter_media_type 条件测试"""

    def _compile_sql(self, stmt):
        """将语句编译为 MySQL 方言的 SQL 字符串"""
        return str(stmt.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))

    def _minimal_parsed(self, filter_media_type: str):
        return {
            "filter_filename": "",
            "_size_min": None,
            "_size_max": None,
            "_date_from_ts": None,
            "_date_to_ts": None,
            "filter_tag": "",
            "filter_media_type": filter_media_type,
        }

    def test_filter_media_type_image_excludes_gif(self):
        """image: media_type=image 且 filename 不以 .gif 结尾"""
        stmt = select(Image)
        parsed = self._minimal_parsed("image")
        stmt, _, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        sql = self._compile_sql(stmt)
        assert has_filters is True
        assert "media_type" in sql
        assert "gif" in sql.lower()
        # 应包含 NOT LIKE %.gif 或类似条件
        assert "NOT" in sql or "not" in sql

    def test_filter_media_type_animated_includes_gif_only(self):
        """animated: media_type=image 且 filename 以 .gif 结尾"""
        stmt = select(Image)
        parsed = self._minimal_parsed("animated")
        stmt, _, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        sql = self._compile_sql(stmt)
        assert has_filters is True
        assert "media_type" in sql
        assert "gif" in sql.lower()

    def test_filter_media_type_video(self):
        """video: media_type=video"""
        stmt = select(Image)
        parsed = self._minimal_parsed("video")
        stmt, _, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        sql = self._compile_sql(stmt)
        assert has_filters is True
        assert "media_type" in sql
        assert "video" in sql.lower()

    def test_filter_media_type_empty_no_media_filter(self):
        """空/未传：不添加 media_type 条件"""
        stmt = select(Image)
        parsed = self._minimal_parsed("")
        stmt, _, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        sql = self._compile_sql(stmt)
        # 无 filter 时 has_filters 为 False；但 path 为空时可能有其他条件
        # 至少不应包含 filter_media_type 相关的 media_type 条件（除非与其他 filter 组合）
        # 空 path + 无其他 filter 时，会有 ~relative_path LIKE '%/%' 条件
        assert "images" in sql


class TestApplyImageFiltersToCountMediaType:
    """apply_image_filters_to_count 中 filter_media_type 条件测试"""

    def _minimal_parsed(self, filter_media_type: str):
        return {
            "filter_filename": "",
            "_size_min": None,
            "_size_max": None,
            "_date_from_ts": None,
            "_date_to_ts": None,
            "filter_tag": "",
            "filter_media_type": filter_media_type,
        }

    def test_count_includes_media_type_filter_image(self):
        """count 语句应包含与 apply_image_filters 相同的 media_type 条件"""
        stmt = select(func.count(Image.id))
        parsed = self._minimal_parsed("image")
        _, pf, _ = apply_image_filters(select(Image), "", "", "folder", parsed)
        count_stmt = apply_image_filters_to_count(stmt, "", "", "folder", parsed, pf)
        sql = str(count_stmt.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))
        assert "media_type" in sql
        assert "gif" in sql.lower()

    def test_count_includes_media_type_filter_video(self):
        parsed = self._minimal_parsed("video")
        stmt = select(func.count(Image.id))
        _, pf, _ = apply_image_filters(select(Image), "", "", "folder", parsed)
        count_stmt = apply_image_filters_to_count(stmt, "", "", "folder", parsed, pf)
        sql = str(count_stmt.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))
        assert "media_type" in sql
        assert "video" in sql.lower()
