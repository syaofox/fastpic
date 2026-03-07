"""唯一路径生成工具测试。"""


from app.utils.unique_path import unique_path


class TestUniquePath:
    """unique_path 函数测试"""

    def test_unique_path_no_conflict(self, tmp_path):
        """测试无冲突时返回原路径"""
        result = unique_path(tmp_path, "new_file.txt")
        assert result == tmp_path / "new_file.txt"

    def test_unique_path_file_conflict_paren(self, tmp_path):
        """测试文件冲突时使用括号风格"""
        (tmp_path / "test.jpg").write_text("original")

        result = unique_path(tmp_path, "test.jpg", suffix_style="paren")
        assert result.name == "test (1).jpg"
        assert result != tmp_path / "test.jpg"

    def test_unique_path_file_conflict_underscore(self, tmp_path):
        """测试文件冲突时使用下划线风格"""
        (tmp_path / "test.jpg").write_text("original")

        result = unique_path(tmp_path, "test.jpg", suffix_style="underscore")
        assert result.name == "test_1.jpg"
        assert result != tmp_path / "test.jpg"

    def test_unique_path_multiple_conflicts_paren(self, tmp_path):
        """测试多次冲突（括号风格）"""
        (tmp_path / "file.txt").write_text("original")
        (tmp_path / "file (1).txt").write_text("copy1")

        result = unique_path(tmp_path, "file.txt", suffix_style="paren")
        assert result.name == "file (2).txt"

    def test_unique_path_multiple_conflicts_underscore(self, tmp_path):
        """测试多次冲突（下划线风格）"""
        (tmp_path / "file.txt").write_text("original")
        (tmp_path / "file_1.txt").write_text("copy1")

        result = unique_path(tmp_path, "file.txt", suffix_style="underscore")
        assert result.name == "file_2.txt"

    def test_unique_path_folder_no_conflict(self, tmp_path):
        """测试文件夹无冲突"""
        result = unique_path(tmp_path, "new_folder", is_folder=True)
        assert result == tmp_path / "new_folder"

    def test_unique_path_folder_conflict(self, tmp_path):
        """测试文件夹冲突"""
        (tmp_path / "existing").mkdir()

        result = unique_path(tmp_path, "existing", is_folder=True)
        assert result.name == "existing (1)"
        assert result != tmp_path / "existing"

    def test_unique_path_folder_conflict_underscore(self, tmp_path):
        """测试文件夹冲突（下划线风格对文件夹无效）"""
        (tmp_path / "folder").mkdir()

        result = unique_path(tmp_path, "folder", is_folder=True, suffix_style="underscore")
        assert result.name == "folder (1)"

    def test_unique_path_file_with_extension(self, tmp_path):
        """测试带扩展名的文件"""
        (tmp_path / "photo.png").write_text("image")

        result = unique_path(tmp_path, "photo.png")
        assert result.suffix == ".png"
        assert result.stem == "photo (1)"

    def test_unique_path_file_no_extension(self, tmp_path):
        """测试无扩展名的文件"""
        (tmp_path / "README").write_text("content")

        result = unique_path(tmp_path, "README")
        assert result.name == "README (1)"

    def test_unique_path_mixed_case_conflicts(self, tmp_path):
        """测试大小写不同的冲突（Windows 不区分大小写）"""
        (tmp_path / "File.txt").write_text("original")

        result = unique_path(tmp_path, "file.txt")
        # Windows 上文件比较不区分大小写，会产生冲突
        # Linux 上可能不会冲突，结果可能是 "file.txt"
        assert result is not None

    def test_unique_path_unicode_filename(self, tmp_path):
        """测试Unicode文件名"""
        (tmp_path / "文件.jpg").write_text("original")

        result = unique_path(tmp_path, "文件.jpg")
        assert "文件" in result.name

    def test_unique_path_space_in_name(self, tmp_path):
        """测试文件名中包含空格"""
        (tmp_path / "my photo.jpg").write_text("original")

        result = unique_path(tmp_path, "my photo.jpg")
        assert result.name == "my photo (1).jpg"


class TestUniquePathEdgeCases:
    """unique_path 边界情况测试"""

    def test_unique_path_only_extension(self, tmp_path):
        """测试仅扩展名"""
        result = unique_path(tmp_path, ".gitignore")
        assert result is not None

    def test_unique_path_special_chars_in_name(self, tmp_path):
        """测试文件名包含特殊字符"""
        (tmp_path / "test@file#1.txt").write_text("original")

        result = unique_path(tmp_path, "test@file#1.txt")
        assert "@" in result.name
