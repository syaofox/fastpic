"""模型相关测试。"""


from app.models import natural_sort_key


class TestNaturalSortKey:
    """natural_sort_key 函数测试"""

    def test_natural_sort_key_basic(self):
        """测试基本数字排序"""
        assert natural_sort_key("file1.txt") == "file0000000001.txt"
        assert natural_sort_key("file2.txt") == "file0000000002.txt"
        assert natural_sort_key("file10.txt") == "file0000000010.txt"
        assert natural_sort_key("file100.txt") == "file0000000100.txt"

    def test_natural_sort_key_multiple_numbers(self):
        """测试多个数字段"""
        assert natural_sort_key("img1_2.jpg") == "img0000000001_0000000002.jpg"
        assert natural_sort_key("a1b2c3.txt") == "a0000000001b0000000002c0000000003.txt"

    def test_natural_sort_key_no_numbers(self):
        """测试无数字的文件名"""
        assert natural_sort_key("hello.txt") == "hello.txt"
        assert natural_sort_key("test_file.png") == "test_file.png"

    def test_natural_sort_key_mixed(self):
        """测试混合内容"""
        assert natural_sort_key("photo2024.jpg") == "photo0000002024.jpg"
        assert natural_sort_key("IMG_1234.JPG") == "IMG_0000001234.JPG"

    def test_natural_sort_key_leading_zeros(self):
        """测试已带前导零的数字"""
        assert natural_sort_key("file007.txt") == "file0000000007.txt"
        assert natural_sort_key("file001.txt") == "file0000000001.txt"

    def test_natural_sort_key_empty(self):
        """测试空字符串"""
        assert natural_sort_key("") == ""

    def test_natural_sort_key_none(self):
        """测试 None"""
        assert natural_sort_key(None) == ""

    def test_natural_sort_key_preserves_non_digit(self):
        """测试非数字字符保持不变"""
        assert natural_sort_key("file-1.txt") == "file-0000000001.txt"
        assert natural_sort_key("file_1.txt") == "file_0000000001.txt"
        assert natural_sort_key("file 1.txt") == "file 0000000001.txt"

    def test_natural_sort_key_long_numbers(self):
        """测试长数字"""
        assert natural_sort_key("file1234567890.txt") == "file1234567890.txt"

    def test_natural_sort_key_sorting_order(self):
        """测试排序顺序"""
        items = ["file10", "file2", "file1", "file20"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["file1", "file2", "file10", "file20"]


class TestModelsImport:
    """模型导入测试"""

    def test_image_model_import(self):
        """验证 Image 模型可导入"""
        from app.models import Image

        assert Image.__tablename__ == "images"

    def test_tag_model_import(self):
        """验证 Tag 模型可导入"""
        from app.models import Tag

        assert Tag.__tablename__ == "tags"

    def test_image_tag_model_import(self):
        """验证 ImageTag 模型可导入"""
        from app.models import ImageTag

        assert ImageTag.__tablename__ == "image_tags"

    def test_path_count_cache_model_import(self):
        """验证 PathCountCache 模型可导入"""
        from app.models import PathCountCache

        assert PathCountCache.__tablename__ == "path_count_cache"

    def test_folder_thumbnail_model_import(self):
        """验证 FolderThumbnail 模型可导入"""
        from app.models import FolderThumbnail

        assert FolderThumbnail.__tablename__ == "folder_thumbnails"
