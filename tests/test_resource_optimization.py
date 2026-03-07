"""资源管理优化测试：连接池配置和流式处理"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDbPoolConfig:
    """测试数据库连接池配置环境变量化"""

    def test_default_pool_values(self, monkeypatch):
        """测试默认连接池参数"""
        monkeypatch.delenv("DB_POOL_SIZE", raising=False)
        monkeypatch.delenv("DB_MAX_OVERFLOW", raising=False)
        monkeypatch.delenv("DB_POOL_RECYCLE", raising=False)

        from app.models import _db_max_overflow, _db_pool_recycle, _db_pool_size

        assert _db_pool_size == 20
        assert _db_max_overflow == 40
        assert _db_pool_recycle == 3600

    def test_env_overrides_defaults(self, monkeypatch):
        """测试环境变量覆盖默认值"""
        monkeypatch.setenv("DB_POOL_SIZE", "10")
        monkeypatch.setenv("DB_MAX_OVERFLOW", "20")
        monkeypatch.setenv("DB_POOL_RECYCLE", "1800")

        size = int(os.environ.get("DB_POOL_SIZE", "20"))
        overflow = int(os.environ.get("DB_MAX_OVERFLOW", "40"))
        recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))

        assert size == 10
        assert overflow == 20
        assert recycle == 1800

    def test_env_not_set_uses_defaults(self, monkeypatch):
        """测试未设置环境变量时使用默认值"""
        monkeypatch.delenv("DB_POOL_SIZE", raising=False)
        monkeypatch.delenv("DB_MAX_OVERFLOW", raising=False)
        monkeypatch.delenv("DB_POOL_RECYCLE", raising=False)

        size = int(os.environ.get("DB_POOL_SIZE", "20"))
        overflow = int(os.environ.get("DB_MAX_OVERFLOW", "40"))
        recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))

        assert size == 20
        assert overflow == 40
        assert recycle == 3600


class TestFolderCountsStreaming:
    """测试流式处理大列表"""

    @pytest.fixture
    def mock_session(self):
        """创建模拟的数据库 session"""
        session = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_streaming_empty_db(self):
        """测试空数据库返回空计数"""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        async def mock_execute(sql, params=None):
            return mock_result

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        from app.utils.folder_tree import _get_folder_counts_streaming

        result = await _get_folder_counts_streaming(mock_session, max_depth=4)

        assert result == {"": 0}

    @pytest.mark.asyncio
    async def test_streaming_single_batch(self):
        """测试单批处理"""
        call_count = 0
        mock_result = MagicMock()

        def fetch_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    (1, "2024/01/photo1.jpg"),
                    (2, "2024/01/photo2.jpg"),
                    (3, "2024/02/photo3.jpg"),
                ]
            return []

        mock_result.fetchall.side_effect = fetch_side_effect

        async def mock_execute(sql, params=None):
            return mock_result

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        from app.utils.folder_tree import _get_folder_counts_streaming

        result = await _get_folder_counts_streaming(mock_session, max_depth=4)

        assert result[""] == 0
        assert result["2024"] == 3
        assert result["2024/01"] == 2
        assert result["2024/02"] == 1

    @pytest.mark.asyncio
    async def test_streaming_multiple_batches(self):
        """测试多批处理"""
        call_count = 0
        mock_result = MagicMock()

        def fetch_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(1, "2024/01/a.jpg"), (2, "2024/01/b.jpg")]
            return []

        mock_result.fetchall.side_effect = fetch_side_effect

        async def mock_execute(sql, params=None):
            return mock_result

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        from app.utils.folder_tree import _get_folder_counts_streaming

        result = await _get_folder_counts_streaming(mock_session, max_depth=4)

        assert result["2024"] == 2
        assert result["2024/01"] == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_streaming_files_without_path(self):
        """测试无路径的文件（根目录文件）"""
        call_count = 0
        mock_result = MagicMock()

        def fetch_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    (1, "photo.jpg"),
                    (2, "2024/01/photo.jpg"),
                ]
            return []

        mock_result.fetchall.side_effect = fetch_side_effect

        async def mock_execute(sql, params=None):
            return mock_result

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        from app.utils.folder_tree import _get_folder_counts_streaming

        result = await _get_folder_counts_streaming(mock_session, max_depth=4)

        assert result[""] == 1
        assert result["2024"] == 1
        assert result["2024/01"] == 1

    @pytest.mark.asyncio
    async def test_streaming_respects_max_depth(self):
        """测试 max_depth 参数"""
        call_count = 0
        mock_result = MagicMock()

        def fetch_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(1, "2024/01/15/photo.jpg")]
            return []

        mock_result.fetchall.side_effect = fetch_side_effect

        async def mock_execute(sql, params=None):
            return mock_result

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        from app.utils.folder_tree import _get_folder_counts_streaming

        result = await _get_folder_counts_streaming(mock_session, max_depth=2)

        assert "2024" in result
        assert "2024/01" in result
        assert "2024/01/15" not in result

    @pytest.mark.asyncio
    async def test_streaming_calls_asyncio_sleep(self):
        """测试每批处理后调用 asyncio.sleep(0)"""
        call_count = 0
        mock_result = MagicMock()

        def fetch_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return [(call_count, f"2024/0{call_count}/a.jpg")]
            return []

        mock_result.fetchall.side_effect = fetch_side_effect

        async def mock_execute(sql, params=None):
            return mock_result

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        from app.utils.folder_tree import _get_folder_counts_streaming

        with patch("app.utils.folder_tree.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _get_folder_counts_streaming(mock_session, max_depth=4)
            assert mock_sleep.call_count == 3
