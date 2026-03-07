"""路径工具测试。"""

from pathlib import Path

from app.utils.path_utils import (
    escape_like,
    invalid_filename,
    normalize_path,
    path_filter_for_prefix,
    relative_path,
    resolve_and_validate_relative_path,
)


class TestNormalizePath:
    """normalize_path 函数测试"""

    def test_normalize_simple_path(self):
        """测试简单路径"""
        assert normalize_path("a/b", allow_empty=True) == "a/b"
        assert normalize_path("folder/file.jpg", allow_empty=True) == "folder/file.jpg"

    def test_normalize_strips_slashes(self):
        """测试去除首尾斜杠"""
        assert normalize_path("/a/b/", allow_empty=True) == "a/b"
        assert normalize_path("///test///", allow_empty=True) == "test"
        assert normalize_path("/2024/01/15/", allow_empty=True) == "2024/01/15"

    def test_normalize_strips_whitespace(self):
        """测试去除空白字符"""
        assert normalize_path("  a/b  ", allow_empty=True) == "a/b"
        assert normalize_path("\ttest\n", allow_empty=True) == "test"

    def test_normalize_empty_path(self):
        """测试空路径处理"""
        assert normalize_path("", allow_empty=True) == ""
        assert normalize_path(None, allow_empty=True) == ""
        assert normalize_path("   ", allow_empty=True) == ""

    def test_normalize_empty_disallowed(self):
        """测试不允许空路径"""
        assert normalize_path("", allow_empty=False) is None
        assert normalize_path(None, allow_empty=False) is None
        assert normalize_path("   ", allow_empty=False) is None

    def test_normalize_blocks_traversal(self):
        """测试路径遍历攻击阻止"""
        assert normalize_path("../etc/passwd", allow_empty=True) is None
        assert normalize_path("foo/../bar", allow_empty=True) is None
        assert normalize_path("a/../../b", allow_empty=True) is None
        assert normalize_path("../folder/file", allow_empty=False) is None
        assert normalize_path("a..b", allow_empty=True) is None  # 安全优先：检测 .. 字符串
        assert normalize_path("a...b", allow_empty=True) is None  # "..." 包含 ".."

    def test_normalize_blocks_absolute_path(self):
        """测试阻止绝对路径"""
        # strip 后以 / 开头会被 strip 调，只保留相对路径
        assert normalize_path("/absolute/path", allow_empty=True) == "absolute/path"
        assert normalize_path("/folder", allow_empty=True) == "folder"

    def test_normalize_edge_cases(self):
        """测试边界情况"""
        assert normalize_path(".", allow_empty=True) == "."
        assert normalize_path("..", allow_empty=True) is None  # 安全优先：检测 .. 字符串
        assert normalize_path("folder.", allow_empty=True) == "folder."
        assert normalize_path(".hidden", allow_empty=True) == ".hidden"
        assert normalize_path("测试/中文路径", allow_empty=True) == "测试/中文路径"


class TestEscapeLike:
    """escape_like 函数测试"""

    def test_escape_percent(self):
        """测试转义百分号"""
        assert escape_like("100%") == "100!%"
        assert escape_like("%test") == "!%test"

    def test_escape_underscore(self):
        """测试转义下划线"""
        assert escape_like("test_") == "test!_"
        assert escape_like("_test") == "!_test"

    def test_escape_exclamation(self):
        """测试转义感叹号"""
        assert escape_like("test!") == "test!!"
        assert escape_like("a!b!c") == "a!!b!!c"

    def test_escape_mixed(self):
        """测试混合转义"""
        # 执行顺序: ! -> % -> _ (每个字符都被转义)
        assert escape_like("100%_test!") == "100!%!_test!!"
        assert escape_like("%_!_test") == "!%!_!!!_test"

    def test_escape_no_special_chars(self):
        """测试无特殊字符"""
        assert escape_like("normal") == "normal"
        assert escape_like("") == ""


class TestInvalidFilename:
    """invalid_filename 函数测试"""

    def test_valid_filenames(self):
        """测试合法文件名"""
        assert invalid_filename("file.jpg") is False
        assert invalid_filename("my photo.png") is False
        assert invalid_filename("测试文件.gif") is False
        assert invalid_filename("2024-01-15.jpg") is False
        assert invalid_filename("file_name.jpg") is False

    def test_invalid_filenames(self):
        """测试非法文件名"""
        assert invalid_filename("") is True
        assert invalid_filename("../file") is True
        assert invalid_filename("file/name.jpg") is True
        assert invalid_filename("file\\name.jpg") is True
        assert invalid_filename("file:name.jpg") is True
        assert invalid_filename("file*name.jpg") is True
        assert invalid_filename("file?name.jpg") is True
        assert invalid_filename('file"name.jpg') is True
        assert invalid_filename("file<name.jpg") is True
        assert invalid_filename("file>name.jpg") is True
        assert invalid_filename("file|name.jpg") is True


class TestPathFilterForPrefix:
    """path_filter_for_prefix 函数测试"""

    def test_filter_include_children(self):
        """测试包含子路径"""
        from app.models import Image

        pf = path_filter_for_prefix(Image.relative_path, "2024/01")
        assert pf is not None

    def test_filter_exclude_children(self):
        """测试仅匹配前缀本身"""
        from app.models import Image

        pf = path_filter_for_prefix(Image.relative_path, "2024/01", include_children=False)
        assert pf is not None

    def test_filter_empty_prefix(self):
        """测试空前缀"""
        from app.models import Image

        pf = path_filter_for_prefix(Image.relative_path, "")
        assert pf is not None


class TestRelativePath:
    """relative_path 函数测试"""

    def test_relative_path_basic(self):
        """测试基本相对路径"""
        photos_dir = Path("/photos")
        full_path = Path("/photos/2024/01/image.jpg")
        result = relative_path(photos_dir, full_path)
        assert result == "2024/01/image.jpg"

    def test_relative_path_windows_separator(self):
        """测试 Windows 路径分隔符"""
        photos_dir = Path("/photos")
        full_path = Path("/photos/folder\\subfolder/image.jpg")
        result = relative_path(photos_dir, full_path)
        assert "\\" in result or "/" in result


class TestResolveAndValidateRelativePath:
    """resolve_and_validate_relative_path 函数测试"""

    def test_valid_relative_path(self, tmp_path):
        """测试合法相对路径"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        test_file = photos_dir / "test.jpg"
        test_file.write_text("content")

        result = resolve_and_validate_relative_path("test.jpg", photos_dir)
        assert result == test_file.resolve()

    def test_invalid_traversal(self, tmp_path):
        """测试非法路径遍历"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        result = resolve_and_validate_relative_path("../etc/passwd", photos_dir)
        assert result is None

    def test_nonexistent_file(self, tmp_path):
        """测试不存在的文件"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        result = resolve_and_validate_relative_path("nonexistent.jpg", photos_dir)
        assert result is None

    def test_absolute_path_rejected(self, tmp_path):
        """测试拒绝绝对路径"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        result = resolve_and_validate_relative_path("/etc/passwd", photos_dir)
        assert result is None
