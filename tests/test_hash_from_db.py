"""_get_existing_hashes_from_db 函数测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGetExistingHashesFromDb:
    """_get_existing_hashes_from_db 函数测试"""

    @pytest.mark.asyncio
    async def test_get_existing_hashes_from_db_root(self):
        """测试根目录查询"""
        from app.config import PHOTOS_DIR
        from app.routers.images import _get_existing_hashes_from_db

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("abc123", "photo1.jpg"),
            ("def456", "photo2.jpg"),
        ]
        mock_session.execute.return_value = mock_result

        with patch("app.routers.images.async_session_factory") as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            target_dir = PHOTOS_DIR
            result = await _get_existing_hashes_from_db(target_dir)

            assert result == {"abc123": "photo1.jpg", "def456": "photo2.jpg"}

    @pytest.mark.asyncio
    async def test_get_existing_hashes_from_db_with_subdirs(self):
        """测试带子目录查询"""
        from app.config import PHOTOS_DIR
        from app.routers.images import _get_existing_hashes_from_db

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("abc123", "2024/photo1.jpg"),
            ("def456", "2024/photo2.jpg"),
        ]
        mock_session.execute.return_value = mock_result

        with patch("app.routers.images.async_session_factory") as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            target_dir = PHOTOS_DIR / "album"
            subdirs = {"2024", ""}
            result = await _get_existing_hashes_from_db(target_dir, subdirs)

            assert result == {"abc123": "2024/photo1.jpg", "def456": "2024/photo2.jpg"}

    @pytest.mark.asyncio
    async def test_get_existing_hashes_excludes_null_md5(self):
        """测试排除 md5_hash 为 None 的记录"""
        from app.config import PHOTOS_DIR
        from app.routers.images import _get_existing_hashes_from_db

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("abc123", "photo1.jpg"),
            (None, "photo2.jpg"),
            ("def456", "photo3.jpg"),
        ]
        mock_session.execute.return_value = mock_result

        with patch("app.routers.images.async_session_factory") as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            target_dir = PHOTOS_DIR
            result = await _get_existing_hashes_from_db(target_dir)

            assert "abc123" in result
            assert "def456" in result
            assert None not in result

    @pytest.mark.asyncio
    async def test_get_existing_hashes_empty_db(self):
        """测试空数据库返回空字典"""
        from app.config import PHOTOS_DIR
        from app.routers.images import _get_existing_hashes_from_db

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        with patch("app.routers.images.async_session_factory") as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            target_dir = PHOTOS_DIR
            result = await _get_existing_hashes_from_db(target_dir)

            assert result == {}
