"""search_dirs API 测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSearchDirsEndpoint:
    """测试 /api/search-dirs 端点"""

    def test_endpoint_exists(self):
        """验证 API 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/search-dirs" in routes

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_list(self):
        """空查询返回空列表"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.models.get_async_session"):
            client = TestClient(app)
            response = client.get("/api/search-dirs?q=")

            assert response.status_code == 200
            assert response.json() == {"dirs": []}

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty_list(self):
        """仅空白字符的查询返回空列表"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.models.get_async_session"):
            client = TestClient(app)
            response = client.get("/api/search-dirs?q=%20%20%20")

            assert response.status_code == 200
            assert response.json() == {"dirs": []}

    @pytest.mark.asyncio
    async def test_uses_sql_like_filter(self):
        """验证使用 SQL LIKE 预过滤"""
        from fastapi.testclient import TestClient

        from app.main import app
        from app.models import get_async_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_async_session = AsyncMock()
        mock_async_session.execute.return_value = mock_result

        async def mock_get_session():
            yield mock_async_session

        app.dependency_overrides[get_async_session] = mock_get_session
        try:
            client = TestClient(app)
            response = client.get("/api/search-dirs?q=2024")

            assert response.status_code == 200
            mock_async_session.execute.assert_called_once()
            call_args = mock_async_session.execute.call_args
            sql_query = str(call_args[0][0])
            assert "LIKE" in sql_query
            assert ":like_filter" in sql_query
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_queries_images_table(self):
        """验证查询 images 表"""
        from fastapi.testclient import TestClient

        from app.main import app
        from app.models import get_async_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_async_session = AsyncMock()
        mock_async_session.execute.return_value = mock_result

        async def mock_get_session():
            yield mock_async_session

        app.dependency_overrides[get_async_session] = mock_get_session
        try:
            client = TestClient(app)
            response = client.get("/api/search-dirs?q=test")

            assert response.status_code == 200
            mock_async_session.execute.assert_called_once()
            call_args = mock_async_session.execute.call_args
            sql_query = str(call_args[0][0])
            assert "images" in sql_query
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_uses_substring_index(self):
        """验证使用 SUBSTRING_INDEX 提取路径前缀"""
        from fastapi.testclient import TestClient

        from app.main import app
        from app.models import get_async_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_async_session = AsyncMock()
        mock_async_session.execute.return_value = mock_result

        async def mock_get_session():
            yield mock_async_session

        app.dependency_overrides[get_async_session] = mock_get_session
        try:
            client = TestClient(app)
            response = client.get("/api/search-dirs?q=test")

            assert response.status_code == 200
            call_args = mock_async_session.execute.call_args
            sql_query = str(call_args[0][0])
            assert "SUBSTRING_INDEX" in sql_query
            assert "GROUP BY" in sql_query
        finally:
            app.dependency_overrides.clear()
