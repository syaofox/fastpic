"""配置参数测试：验证新增的配置常量"""



class TestConfigConstants:
    """测试 config.py 中新增的配置常量"""

    def test_folder_images_max(self):
        """验证大图模式最大返回数量"""
        from app.config import FOLDER_IMAGES_MAX

        assert FOLDER_IMAGES_MAX == 5000

    def test_default_per_page(self):
        """验证默认每页数量"""
        from app.config import DEFAULT_PER_PAGE

        assert DEFAULT_PER_PAGE == 24

    def test_default_cols(self):
        """验证默认列数"""
        from app.config import DEFAULT_COLS

        assert DEFAULT_COLS == 4

    def test_upload_parallel(self):
        """验证上传并发数"""
        from app.config import UPLOAD_PARALLEL

        assert UPLOAD_PARALLEL == 4

    def test_scan_process_batch_size(self):
        """验证扫描批处理大小"""
        from app.config import SCAN_PROCESS_BATCH_SIZE

        assert SCAN_PROCESS_BATCH_SIZE == 16

    def test_cleanup_batch_size(self):
        """验证数据库清理批处理大小"""
        from app.config import CLEANUP_BATCH_SIZE

        assert CLEANUP_BATCH_SIZE == 5000


class TestConfigUsage:
    """测试配置参数在各模块中的使用"""

    def test_main_uses_folder_images_max(self):
        """验证 main.py 使用 FOLDER_IMAGES_MAX"""
        import app.main as main_module

        assert hasattr(main_module, "FOLDER_IMAGES_MAX")

    def test_scanner_uses_cleanup_batch_size(self):
        """验证 scanner.py 使用 CLEANUP_BATCH_SIZE"""
        import app.services.scanner as scanner_module

        assert hasattr(scanner_module, "CLEANUP_BATCH_SIZE")

    def test_routers_images_uses_upload_parallel(self):
        """验证 images router 使用 UPLOAD_PARALLEL"""
        import app.routers.images as images_router

        assert hasattr(images_router, "UPLOAD_PARALLEL")
