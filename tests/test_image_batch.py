"""collect_image_items_by_prefix 函数测试。"""

from unittest.mock import MagicMock, patch

import pytest


class TestCollectImageItemsByPrefix:
    """collect_image_items_by_prefix 测试"""

    @pytest.mark.asyncio
    async def test_collect_returns_md5_hash(self):
        """测试返回包含 md5_hash"""
        mock_img = MagicMock()
        mock_img.id = 1
        mock_img.relative_path = "test.jpg"
        mock_img.md5_hash = "abc123"

        async def fake_iter(*args, **kwargs):
            yield [mock_img]

        with patch("app.utils.image_batch.iter_images_by_path_prefix", return_value=fake_iter()):
            from app.utils.image_batch import collect_image_items_by_prefix

            result = await collect_image_items_by_prefix(MagicMock(), "", "a")

        assert len(result) == 1
        assert result[0] == (1, "test.jpg", "a", "abc123")

    @pytest.mark.asyncio
    async def test_collect_with_null_md5(self):
        """测试 md5_hash 为 None 的情况"""
        mock_img = MagicMock()
        mock_img.id = 2
        mock_img.relative_path = "test2.jpg"
        mock_img.md5_hash = None

        async def fake_iter(*args, **kwargs):
            yield [mock_img]

        with patch("app.utils.image_batch.iter_images_by_path_prefix", return_value=fake_iter()):
            from app.utils.image_batch import collect_image_items_by_prefix

            result = await collect_image_items_by_prefix(MagicMock(), "", "b")

        assert len(result) == 1
        assert result[0] == (2, "test2.jpg", "b", None)
