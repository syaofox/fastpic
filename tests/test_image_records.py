"""create_image_record 函数 md5_hash 参数测试。"""

from app.utils.image_records import create_image_record


class TestCreateImageRecordMd5:
    """create_image_record 函数测试"""

    def test_create_image_record_without_md5(self):
        """测试不传 md5_hash 参数时，默认应为 None"""
        record = create_image_record(
            filename="test.jpg",
            relative_path="test.jpg",
            modified_at=1234567890.0,
            file_size=1024,
            width=100,
            height=100,
        )
        assert record.md5_hash is None

    def test_create_image_record_with_md5(self):
        """测试传入 md5_hash 参数"""
        record = create_image_record(
            filename="test.jpg",
            relative_path="test.jpg",
            modified_at=1234567890.0,
            file_size=1024,
            width=100,
            height=100,
            md5_hash="d41d8cd98f00b204e9800998ecf8427e",
        )
        assert record.md5_hash == "d41d8cd98f00b204e9800998ecf8427e"

    def test_create_image_record_md5_length(self):
        """测试 md5_hash 长度"""
        record = create_image_record(
            filename="test.jpg",
            relative_path="test.jpg",
            modified_at=1234567890.0,
            file_size=1024,
            width=100,
            height=100,
            md5_hash="abcd1234567890efabcd1234567890ef",
        )
        assert len(record.md5_hash) == 32
