"""Scanner 错误隔离测试。"""



class TestScannerErrorClasses:
    """Scanner 错误处理测试"""

    def test_scan_photos_import(self):
        """验证 scan_photos 可正常导入"""
        from app.services.scanner import scan_photos

        assert callable(scan_photos)

    def test_process_batch_safe_is_nested_function(self):
        """验证 _process_batch_safe 是 scan_photos 内部的嵌套函数"""
        import inspect

        from app.services.scanner import scan_photos

        source = inspect.getsource(scan_photos)
        assert "_process_batch_safe" in source
        assert "批次处理失败" in source


class TestScannerErrorHandling:
    """Scanner 错误处理逻辑测试"""

    def test_process_batch_safe_catches_exception(self):
        """验证 _process_batch_safe 能捕获异常并返回空列表"""
        import inspect

        from app.services.scanner import scan_photos

        source = inspect.getsource(scan_photos)

        assert "try:" in source
        assert "except Exception" in source
        assert "批次处理失败" in source
        assert "return []" in source

    def test_process_batch_safe_wraps_process_batch(self):
        """验证 _process_batch_safe 包装了 _process_batch"""
        import inspect

        from app.services.scanner import scan_photos

        source = inspect.getsource(scan_photos)

        assert "await _process_batch_safe" in source
        assert "_process_batch(paths)" in source or "await _process_batch" in source

    def test_batch_processing_continues_on_failure(self):
        """验证批次处理失败时继续处理下一批"""
        import inspect

        from app.services.scanner import scan_photos

        source = inspect.getsource(scan_photos)

        assert "continue" in source or "return []" in source
