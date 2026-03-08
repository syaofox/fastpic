"""图片信息 API 测试"""

import pytest


class TestImageInfoEndpoint:
    def test_endpoint_exists(self):
        """验证 API 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/image-info/{image_id:int}" in routes

    def test_params(self):
        """验证参数配置"""
        import inspect

        from app.routers.images import get_image_info

        sig = inspect.signature(get_image_info)
        params = sig.parameters

        assert "image_id" in params
        assert "session" in params
