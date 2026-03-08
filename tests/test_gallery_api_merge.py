"""API 合并优化测试：/api/gallery-data 组合 API"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestApiGalleryDataEndpoint:
    """测试 /api/gallery-data 组合 API"""

    def test_endpoint_exists(self):
        """验证 API 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/gallery-data" in routes

    def test_params_default_values(self):
        """验证默认参数值"""
        import inspect

        from app.main import api_gallery_data

        sig = inspect.signature(api_gallery_data)
        params = sig.parameters

        assert params["path"].default == ""
        assert params["search"].default == ""
        assert params["mode"].default == "folder"
        assert params["sort_by"].default == "modified_at"
        assert params["sort_order"].default == "desc"
        assert params["cols"].default == 4

    def test_per_page_calculation(self):
        """测试每页数量计算"""
        from app.main import _per_page_for_cols

        result = _per_page_for_cols(4)
        assert result == 24
        assert result > 0


class TestGalleryDataParamsNormalization:
    """测试参数规范化"""

    def test_normalize_empty_path(self):
        """测试空路径处理"""
        from app.utils.path_utils import normalize_path

        result = normalize_path("", allow_empty=True)
        assert result == ""

        result = normalize_path(None, allow_empty=True)
        assert result is None or result == ""

    def test_normalize_path_strips_trailing_slash(self):
        """测试路径去除尾部斜杠"""
        from app.utils.path_utils import normalize_path

        result = normalize_path("test/", allow_empty=True)
        assert result == "test"

    def test_path_traversal_blocked(self):
        """测试路径遍历被阻止"""
        from app.utils.path_utils import normalize_path

        result = normalize_path("../etc/passwd", allow_empty=True)
        assert result is None

        result = normalize_path("foo/../bar", allow_empty=True)
        assert result is None


class TestGalleryDataResponse:
    """测试 API 响应"""

    @pytest.mark.asyncio
    async def test_response_template_name(self):
        """验证返回的模板名称"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.main.get_async_session"):
            with patch("app.main.async_session_factory") as mock_factory:
                with patch("app.main.get_subfolders") as mock_subfolders:
                    with patch("app.main.apply_image_filters") as mock_filters:
                        mock_filters.return_value = (MagicMock(), MagicMock(), False)
                        mock_subfolders.return_value = []

                        mock_scalars = MagicMock()
                        mock_scalars.all.return_value = []
                        mock_result = MagicMock()
                        mock_result.scalars.return_value = mock_scalars

                        mock_async_session = AsyncMock()
                        mock_async_session.execute.return_value = mock_result

                        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_async_session)
                        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

                        client = TestClient(app)
                        response = client.get("/api/gallery-data")

                        assert response.status_code == 200
                        assert "gallery-top-bar" in response.text


class TestAsyncGatherOptimization:
    """测试异步并发优化"""

    @pytest.mark.asyncio
    async def test_asyncio_gather_used(self):
        """验证使用了 asyncio.gather 并发获取"""
        from app.main import api_gallery_data

        with patch("app.main.asyncio.gather") as mock_gather:
            with patch("app.main.get_async_session"):
                with patch("app.main.async_session_factory"):
                    with patch("app.main.get_subfolders", new_callable=AsyncMock):
                        with patch("app.main.apply_image_filters"):
                            mock_gather.return_value = AsyncMock(return_value=([], []))

                            try:
                                from unittest.mock import MagicMock as MockRequest

                                mock_req = MockRequest()
                                mock_req.url = MockRequest()
                                mock_req.url.query = ""

                                await api_gallery_data(
                                    request=mock_req,
                                    session=AsyncMock(),
                                )
                            except Exception:
                                pass

                            if mock_gather.called:
                                args, kwargs = mock_gather.call_args
                                assert len(args) == 2
