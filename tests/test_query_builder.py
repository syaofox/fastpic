"""查询构建器测试。"""

from datetime import datetime

from app.models import Image
from app.utils.query_builder import (
    IMAGE_SORT_COLUMNS,
    apply_image_filters,
    apply_image_filters_to_count,
    get_sort_column,
    parse_filter_params,
)


class TestParseFilterParams:
    """parse_filter_params 函数测试"""

    def test_parse_empty_params(self):
        """测试空参数"""
        result = parse_filter_params()
        assert result["filter_filename"] == ""
        assert result["_size_min"] is None
        assert result["_size_max"] is None
        assert result["_date_from_ts"] is None
        assert result["_date_to_ts"] is None
        assert result["filter_tag"] == ""

    def test_parse_filename_filter(self):
        """测试文件名过滤"""
        result = parse_filter_params(filter_filename="test")
        assert result["filter_filename"] == "test"

        result = parse_filter_params(filter_filename="  hello  ")
        assert result["filter_filename"] == "hello"

    def test_parse_size_filters(self):
        """测试大小过滤"""
        result = parse_filter_params(filter_size_min="1024", filter_size_max="1048576")
        assert result["_size_min"] == 1024
        assert result["_size_max"] == 1048576

        result = parse_filter_params(filter_size_min="0")
        assert result["_size_min"] == 0

        result = parse_filter_params(filter_size_min="abc")
        assert result["_size_min"] is None

        result = parse_filter_params(filter_size_max="invalid")
        assert result["_size_max"] is None

    def test_parse_date_filters(self):
        """测试日期过滤"""
        result = parse_filter_params(filter_date_from="2024-01-15", filter_date_to="2024-12-31")
        assert result["_date_from_ts"] is not None
        assert result["_date_to_ts"] is not None

        from_dt = datetime.fromtimestamp(result["_date_from_ts"])
        to_dt = datetime.fromtimestamp(result["_date_to_ts"])
        assert from_dt.year == 2024
        assert from_dt.month == 1
        assert from_dt.day == 15
        assert to_dt.year == 2024
        assert to_dt.month == 12
        assert to_dt.day == 31

    def test_parse_date_filters_invalid(self):
        """测试无效日期"""
        result = parse_filter_params(filter_date_from="invalid-date")
        assert result["_date_from_ts"] is None

        result = parse_filter_params(filter_date_to="2024-13-01")
        assert result["_date_to_ts"] is None

    def test_parse_tag_filter(self):
        """测试标签过滤"""
        result = parse_filter_params(filter_tag="nature")
        assert result["filter_tag"] == "nature"

        result = parse_filter_params(filter_tag="  travel  ")
        assert result["filter_tag"] == "travel"


class TestGetSortColumn:
    """get_sort_column 函数测试"""

    def test_sort_column_filename(self):
        """测试文件名排序"""
        col = get_sort_column("filename")
        assert col is not None

    def test_sort_column_folder_filename(self):
        """测试文件夹文件名排序"""
        col = get_sort_column("folder_filename")
        assert col is not None

    def test_sort_column_modified_at(self):
        """测试修改时间排序"""
        col = get_sort_column("modified_at")
        assert col is not None

    def test_sort_column_file_size(self):
        """测试文件大小排序"""
        col = get_sort_column("file_size")
        assert col is not None

    def test_sort_column_default(self):
        """测试默认排序（modified_at）"""
        col = get_sort_column("unknown_sort")
        assert col is not None

    def test_sort_columns_dict_has_all(self):
        """测试排序列字典包含所有必需列"""
        assert "filename" in IMAGE_SORT_COLUMNS
        assert "folder_filename" in IMAGE_SORT_COLUMNS
        assert "modified_at" in IMAGE_SORT_COLUMNS
        assert "file_size" in IMAGE_SORT_COLUMNS


class TestApplyImageFilters:
    """apply_image_filters 函数测试"""

    def test_apply_filters_no_params(self):
        """测试无过滤参数"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        assert has_filters is False

    def test_apply_filters_filename(self):
        """测试文件名过滤"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params(filter_filename="test.jpg")
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        assert has_filters is True

    def test_apply_filters_size_range(self):
        """测试大小范围过滤"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params(filter_size_min="1000", filter_size_max="1000000")
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        assert has_filters is True

    def test_apply_filters_date_range(self):
        """测试日期范围过滤"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params(filter_date_from="2024-01-01", filter_date_to="2024-12-31")
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        assert has_filters is True

    def test_apply_filters_tag(self):
        """测试标签过滤"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params(filter_tag="nature")
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        assert has_filters is True

    def test_apply_filters_path(self):
        """测试路径过滤"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "2024/01", "", "folder", parsed)
        assert pf is not None
        assert has_filters is False

    def test_apply_filters_search(self):
        """测试搜索（search 参数不设置 has_filters）"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "testquery", "folder", parsed)
        # search 参数不改变 has_filters 的值
        assert has_filters is False

    def test_apply_filters_combined(self):
        """测试组合过滤"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params(
            filter_filename="photo",
            filter_size_min="1000",
            filter_date_from="2024-01-01",
            filter_tag="vacation",
        )
        result_stmt, pf, has_filters = apply_image_filters(stmt, "2024", "trip", "list", parsed)
        assert has_filters is True


class TestApplyImageFiltersToCount:
    """apply_image_filters_to_count 函数测试"""

    def test_count_no_filters(self):
        """测试无过滤条件的 count"""
        from sqlalchemy import func
        from sqlmodel import select

        count_stmt = select(func.count(1))
        parsed = parse_filter_params()
        result = apply_image_filters_to_count(count_stmt, "", "", "folder", parsed, None)
        assert result is not None

    def test_count_with_path_filter(self):
        """测试带路径过滤的 count"""
        from sqlalchemy import func
        from sqlmodel import select

        count_stmt = select(func.count(1))
        parsed = parse_filter_params()
        from app.models import Image
        from app.utils.path_utils import path_filter_for_prefix

        pf = path_filter_for_prefix(Image.relative_path, "2024")
        result = apply_image_filters_to_count(count_stmt, "2024", "", "folder", parsed, pf)
        assert result is not None

    def test_count_with_search(self):
        """测试带搜索的 count"""
        from sqlalchemy import func
        from sqlmodel import select

        count_stmt = select(func.count(1))
        parsed = parse_filter_params()
        result = apply_image_filters_to_count(count_stmt, "", "test", "folder", parsed, None)
        assert result is not None

    def test_count_with_size_filter(self):
        """测试带大小过滤的 count"""
        from sqlalchemy import func
        from sqlmodel import select

        count_stmt = select(func.count(1))
        parsed = parse_filter_params(filter_size_min="1000", filter_size_max="1000000")
        result = apply_image_filters_to_count(count_stmt, "", "", "folder", parsed, None)
        assert result is not None

    def test_count_uses_substring_index(self):
        """测试 path 为空时使用 SUBSTRING_INDEX 而非前导通配符 LIKE"""
        from sqlalchemy import func
        from sqlmodel import select

        count_stmt = select(func.count(1))
        parsed = parse_filter_params()
        result = apply_image_filters_to_count(count_stmt, "", "", "folder", parsed, None)
        sql_str = str(result).lower()
        assert "substring_index" in sql_str
        assert "%/%" not in sql_str

    def test_list_mode_uses_substring_index(self):
        """测试 list 模式 path 为空时也使用 SUBSTRING_INDEX"""
        from sqlalchemy import func
        from sqlmodel import select

        count_stmt = select(func.count(1))
        parsed = parse_filter_params()
        result = apply_image_filters_to_count(count_stmt, "", "", "list", parsed, None)
        sql_str = str(result).lower()
        assert "substring_index" in sql_str


class TestApplyImageFiltersSubstringIndex:
    """apply_image_filters 函数中 SUBSTRING_INDEX 优化测试"""

    def test_folder_mode_uses_substring_index(self):
        """测试 folder 模式 path 为空时不返回任何图片"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "folder", parsed)
        sql_str = str(result_stmt).lower()
        assert "id = " in sql_str and ":id_" in sql_str

    def test_list_mode_uses_substring_index(self):
        """测试 list 模式 path 为空时不返回任何图片"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "", "list", parsed)
        sql_str = str(result_stmt).lower()
        assert "id = " in sql_str and ":id_" in sql_str

    def test_folder_mode_with_path_uses_like(self):
        """测试 folder 模式有 path 时仍使用 LIKE（可利用索引）"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "2024", "", "folder", parsed)
        sql_str = str(result_stmt)
        assert "LIKE" in sql_str

    def test_folder_mode_root_with_search_returns_empty(self):
        """测试 folder 模式根目录有搜索词时仍不返回图片（优先显示文件夹）"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "", "test", "folder", parsed)
        sql_str = str(result_stmt).lower()
        assert "id = " in sql_str and ":id_" in sql_str

    def test_folder_mode_subfolder_returns_direct_images(self):
        """测试 folder 模式子目录返回该目录的直接照片（不含子目录）"""
        from sqlmodel import select

        stmt = select(Image)
        parsed = parse_filter_params()
        result_stmt, pf, has_filters = apply_image_filters(stmt, "2024/01", "", "folder", parsed)
        sql_str = str(result_stmt).lower()
        assert "like" in sql_str
