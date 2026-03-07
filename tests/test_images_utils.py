"""图片相关工具测试。"""


from app.utils.images import cache_filename, delete_image_files


class TestCacheFilename:
    """cache_filename 函数测试"""

    def test_cache_filename_simple(self):
        """测试简单文件名"""
        result = cache_filename("image.jpg")
        assert result.endswith(".webp")
        assert "/" in result
        parts = result.split("/")
        assert len(parts) == 3

    def test_cache_filename_nested_path(self):
        """测试嵌套路径"""
        result = cache_filename("2024/01/image.jpg")
        assert result.endswith(".webp")
        assert "/" in result
        parts = result.split("/")
        assert len(parts) == 3

    def test_cache_filename_deep_nested(self):
        """测试深层嵌套路径"""
        result = cache_filename("a/b/c/d/e/f/g/image.jpg")
        assert result.endswith(".webp")
        parts = result.split("/")
        assert len(parts) == 3

    def test_cache_filename_consistency(self):
        """测试相同路径产生相同的缓存名"""
        result1 = cache_filename("test/image.jpg")
        result2 = cache_filename("test/image.jpg")
        assert result1 == result2

    def test_cache_filename_different_paths(self):
        """测试不同路径产生不同的缓存名"""
        result1 = cache_filename("image1.jpg")
        result2 = cache_filename("image2.jpg")
        assert result1 != result2

    def test_cache_filename_format(self):
        """测试缓存名格式: hash[:2]/hash[2:4]/hash[4:].webp"""
        result = cache_filename("test.jpg")
        parts = result.split("/")
        assert len(parts) == 3
        assert parts[0].__len__() == 2
        assert parts[1].__len__() == 2
        assert parts[2].endswith(".webp")

    def test_cache_filename_unicode(self):
        """测试 Unicode 文件名"""
        result = cache_filename("测试/图片.jpg")
        assert result.endswith(".webp")
        assert "/" in result


class TestDeleteImageFiles:
    """delete_image_files 函数测试"""

    def test_delete_existing_image(self, tmp_path):
        """测试删除已存在的图片"""
        photos_dir = tmp_path / "photos"
        cache_dir = tmp_path / "cache"
        photos_dir.mkdir()
        cache_dir.mkdir()

        image_path = photos_dir / "test.jpg"
        image_path.write_bytes(b"image content")

        rel_path = "test.jpg"
        from app.utils.images import cache_filename

        cache_name = cache_filename(rel_path)
        cache_file = cache_dir / cache_name
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(b"cache content")

        delete_image_files(rel_path, photos_dir, cache_dir)

        assert not image_path.exists()
        assert not cache_file.exists()

    def test_delete_nonexistent_image(self, tmp_path):
        """测试删除不存在的图片（不应抛出异常）"""
        photos_dir = tmp_path / "photos"
        cache_dir = tmp_path / "cache"
        photos_dir.mkdir()
        cache_dir.mkdir()

        delete_image_files("nonexistent.jpg", photos_dir, cache_dir)

    def test_delete_only_photo_exists(self, tmp_path):
        """测试仅原图存在时删除原图"""
        photos_dir = tmp_path / "photos"
        cache_dir = tmp_path / "cache"
        photos_dir.mkdir()
        cache_dir.mkdir()

        image_path = photos_dir / "test.jpg"
        image_path.write_bytes(b"image content")

        delete_image_files("test.jpg", photos_dir, cache_dir)

        assert not image_path.exists()

    def test_delete_only_cache_exists(self, tmp_path):
        """测试仅缓存存在时删除缓存"""
        photos_dir = tmp_path / "photos"
        cache_dir = tmp_path / "cache"
        photos_dir.mkdir()
        cache_dir.mkdir()

        rel_path = "test.jpg"
        from app.utils.images import cache_filename

        cache_name = cache_filename(rel_path)
        cache_file = cache_dir / cache_name
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(b"cache content")

        delete_image_files(rel_path, photos_dir, cache_dir)

        assert not cache_file.exists()

    def test_delete_nested_path(self, tmp_path):
        """测试删除嵌套路径的图片"""
        photos_dir = tmp_path / "photos"
        cache_dir = tmp_path / "cache"
        photos_dir.mkdir()
        cache_dir.mkdir()

        sub_dir = photos_dir / "2024" / "01"
        sub_dir.mkdir(parents=True)
        image_path = sub_dir / "image.jpg"
        image_path.write_bytes(b"image content")

        delete_image_files("2024/01/image.jpg", photos_dir, cache_dir)

        assert not image_path.exists()
