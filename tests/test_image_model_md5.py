"""Image 模型 md5_hash 字段测试。"""


class TestImageModelMd5Hash:
    """Image 模型 md5_hash 字段测试"""

    def test_image_model_has_md5_hash_field(self):
        """验证 Image 模型有 md5_hash 字段"""
        from app.models import Image

        assert hasattr(Image, "md5_hash")

    def test_image_model_md5_hash_default_none(self):
        """验证 md5_hash 默认值为 None"""
        from app.models import Image

        image = Image(
            filename="test.jpg",
            relative_path="test.jpg",
            modified_at=1234567890.0,
            file_size=1024,
            width=100,
            height=100,
        )
        assert image.md5_hash is None

    def test_image_model_md5_hash_can_be_set(self):
        """验证 md5_hash 可以被设置"""
        from app.models import Image

        image = Image(
            filename="test.jpg",
            relative_path="test.jpg",
            modified_at=1234567890.0,
            file_size=1024,
            width=100,
            height=100,
            md5_hash="d41d8cd98f00b204e9800998ecf8427e",
        )
        assert image.md5_hash == "d41d8cd98f00b204e9800998ecf8427e"

    def test_image_model_md5_hash_indexed(self):
        """验证 md5_hash 字段有索引"""
        from app.models import Image

        columns = Image.__table__.columns
        assert "md5_hash" in columns
        assert columns["md5_hash"].index is True
