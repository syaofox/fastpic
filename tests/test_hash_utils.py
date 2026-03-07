"""哈希工具测试。"""


from app.utils.hash_utils import compute_file_md5, compute_file_md5_by_path


class TestComputeFileMd5ByPath:
    """compute_file_md5_by_path 函数测试"""

    def test_compute_md5_existing_file(self, tmp_path):
        """测试计算已存在文件的 MD5"""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello world")

        result = compute_file_md5_by_path(test_file)
        assert result is not None
        assert len(result) == 32  # MD5 is 32 hex characters

    def test_compute_md5_content_consistency(self, tmp_path):
        """测试相同内容产生相同的 MD5"""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello world")

        result1 = compute_file_md5_by_path(test_file)
        result2 = compute_file_md5_by_path(test_file)
        assert result1 == result2

    def test_compute_md5_different_content(self, tmp_path):
        """测试不同内容产生不同的 MD5"""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_bytes(b"hello")
        file2.write_bytes(b"world")

        result1 = compute_file_md5_by_path(file1)
        result2 = compute_file_md5_by_path(file2)
        assert result1 != result2

    def test_compute_md5_nonexistent_file(self, tmp_path):
        """测试不存在的文件"""
        test_file = tmp_path / "nonexistent.txt"
        assert compute_file_md5_by_path(test_file) is None

    def test_compute_md5_directory(self, tmp_path):
        """测试目录（非文件）"""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        assert compute_file_md5_by_path(test_dir) is None

    def test_compute_md5_large_file(self, tmp_path):
        """测试大文件"""
        test_file = tmp_path / "large.bin"
        # Create a 1MB file
        test_file.write_bytes(b"x" * (1024 * 1024))

        result = compute_file_md5_by_path(test_file)
        assert result is not None
        assert len(result) == 32

    def test_compute_md5_empty_file(self, tmp_path):
        """测试空文件"""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        result = compute_file_md5_by_path(test_file)
        # Empty file has known MD5
        assert result == "d41d8cd98f00b204e9800998ecf8427e"


class TestComputeFileMd5:
    """compute_file_md5 函数测试"""

    def test_compute_md5_valid_path(self, tmp_path):
        """测试计算有效相对路径的 MD5"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        test_file = photos_dir / "image.jpg"
        test_file.write_bytes(b"test content")

        result = compute_file_md5(photos_dir, "image.jpg")
        assert result is not None
        assert len(result) == 32

    def test_compute_md5_nested_path(self, tmp_path):
        """测试嵌套路径"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        sub_dir = photos_dir / "2024"
        sub_dir.mkdir()
        test_file = sub_dir / "image.jpg"
        test_file.write_bytes(b"test content")

        result = compute_file_md5(photos_dir, "2024/image.jpg")
        assert result is not None
        assert len(result) == 32

    def test_compute_md5_nonexistent_relative_path(self, tmp_path):
        """测试不存在的相对路径"""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        result = compute_file_md5(photos_dir, "nonexistent.jpg")
        assert result is None
