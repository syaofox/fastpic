"""格式化工具测试。"""


from app.utils.format import format_file_size


class TestFormatFileSize:
    """format_file_size 函数测试"""

    def test_format_bytes(self):
        """测试字节"""
        assert format_file_size(0) == "0 B"
        assert format_file_size(1) == "1 B"
        assert format_file_size(100) == "100 B"
        assert format_file_size(1023) == "1023 B"

    def test_format_kilobytes(self):
        """测试千字节"""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(10240) == "10.0 KB"
        assert format_file_size(1048575) == "1024.0 KB"

    def test_format_megabytes(self):
        """测试兆字节"""
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1572864) == "1.5 MB"
        assert format_file_size(10485760) == "10.0 MB"
        assert format_file_size(1073741823) == "1024.0 MB"

    def test_format_gigabytes(self):
        """测试吉字节"""
        assert format_file_size(1073741824) == "1.0 GB"
        assert format_file_size(1610612736) == "1.5 GB"
        assert format_file_size(10737418240) == "10.0 GB"

    def test_format_large_values(self):
        """测试大数值"""
        assert format_file_size(1000000000) == "953.7 MB"
        assert format_file_size(1000000000000) == "931.3 GB"

    def test_format_precision(self):
        """测试精度"""
        result = format_file_size(1536)
        assert ".5" in result
        assert "KB" in result

    def test_format_negative(self):
        """测试负数（可能产生意外结果，取决于实现）"""
        result = format_file_size(-1)
        assert "B" in result
