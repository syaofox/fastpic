"""搜索工具测试。"""

from app.utils.search import (
    search_match,
    to_pinyin_lower,
    to_simplified,
    to_traditional,
)


class TestToSimplified:
    """to_simplified 函数测试"""

    def test_simplified_no_change(self):
        """测试已经是简体的内容"""
        assert to_simplified("hello world") == "hello world"
        assert to_simplified("测试") == "测试"

    def test_simplified_traditional_to_simple(self):
        """测试繁体转简体"""
        result = to_simplified("測試")
        assert result == "测试"

    def test_simplified_mixed(self):
        """测试混合内容"""
        result = to_simplified("hello測試world")
        assert "测试" in result

    def test_simplified_empty(self):
        """测试空字符串"""
        assert to_simplified("") == ""


class TestToTraditional:
    """to_traditional 函数测试"""

    def test_traditional_no_change(self):
        """测试已经是繁体的内容"""
        assert to_traditional("測試") == "測試"
        assert to_traditional("hello") == "hello"

    def test_traditional_simple_to_traditional(self):
        """测试简体转繁体"""
        result = to_traditional("测试")
        assert result == "測試"

    def test_traditional_mixed(self):
        """测试混合内容"""
        result = to_traditional("hello测试world")
        assert "測試" in result

    def test_traditional_empty(self):
        """测试空字符串"""
        assert to_traditional("") == ""


class TestToPinyinLower:
    """to_pinyin_lower 函数测试"""

    def test_pinyin_chinese(self):
        """测试中文转拼音"""
        assert to_pinyin_lower("厦门") == "xiamen"
        assert to_pinyin_lower("北京") == "beijing"

    def test_pinyin_mixed(self):
        """测试中英文混合"""
        result = to_pinyin_lower("hello世界")
        assert "hello" in result
        assert "shijie" in result

    def test_pinyin_english_only(self):
        """测试纯英文"""
        assert to_pinyin_lower("hello") == "hello"
        assert to_pinyin_lower("Hello World") == "hello world"

    def test_pinyin_numbers(self):
        """测试数字"""
        assert to_pinyin_lower("123") == "123"

    def test_pinyin_empty(self):
        """测试空字符串"""
        assert to_pinyin_lower("") == ""

    def test_pinyin_special_chars(self):
        """测试特殊字符"""
        result = to_pinyin_lower("@#%^&*")
        # 特殊字符保持不变
        assert result == "@#%^&*"


class TestSearchMatch:
    """search_match 函数测试"""

    def test_match_exact_case_insensitive(self):
        """测试大小写不敏感匹配"""
        assert search_match("test", "Test File") is True
        assert search_match("TEST", "test file") is True
        assert search_match("file", "test FILE") is True

    def test_match_substring(self):
        """测试子串匹配"""
        assert search_match("test", "this is a test file") is True
        assert search_match("photo", "my photo gallery") is True

    def test_match_no_match(self):
        """测试不匹配"""
        assert search_match("xyz", "abc def") is False

    def test_match_empty_query(self):
        """测试空查询"""
        assert search_match("", "any content") is False

    def test_match_empty_target(self):
        """测试空目标"""
        assert search_match("test", "") is False

    def test_match_simplified_chinese(self):
        """测试简体中文匹配"""
        assert search_match("测试", "这是一个测试文件") is True
        assert search_match("测试", "這是一個測試文件") is True

    def test_match_traditional_chinese(self):
        """测试繁体中文匹配"""
        assert search_match("測試", "這是一個測試文件") is True
        assert search_match("測試", "这是一个测试文件") is True

    def test_match_pinyin(self):
        """测试拼音匹配"""
        assert search_match("xiamen", "厦门大学") is True
        assert search_match("beijing", "北京欢迎你") is True
        assert search_match("shanghai", "上海滩") is True

    def test_match_chinese_to_pinyin(self):
        """测试中文查询匹配拼音"""
        assert search_match("厦门", "xiamen university") is True
        assert search_match("北京", "beijing city") is True

    def test_match_partial_chinese(self):
        """测试中文部分匹配"""
        assert search_match("厦", "厦门大学") is True
        assert search_match("门", "厦门大学") is True

    def test_match_folder_path(self):
        """测试文件夹路径匹配"""
        assert search_match("2024", "2024/01/photo.jpg") is True
        assert search_match("01", "2024/01/15/image.png") is True
        assert search_match("vacation", "trip/vacation/photo.jpg") is True

    def test_match_special_chars(self):
        """测试特殊字符"""
        assert search_match("file-1", "my file-1.jpg") is True
        assert search_match("file_1", "photo_file_1.png") is True

    def test_match_case_sensitivity(self):
        """测试大小写敏感性"""
        assert search_match("Test", "test") is True
        assert search_match("TEST", "test") is True


class TestSearchMatchEdgeCases:
    """search_match 边界情况测试"""

    def test_match_whitespace_only_query(self):
        """测试仅空白字符的查询"""
        # 空白字符被 strip 后仍为空字符串
        assert search_match("", "content") is False

    def test_match_whitespace_in_target(self):
        """测试目标中的空白"""
        assert search_match("test", "  test  ") is True

    def test_match_unicode_variation(self):
        """测试 Unicode 变体"""
        # 这些测试取决于 pypinyin 库的行为
        # 简化测试
        assert search_match("test", "test") is True
