"""delete-images API 测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDeleteImagesEndpoint:
    """测试 /api/delete-images 端点"""

    def test_endpoint_exists(self):
        """验证 API 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/delete-images" in routes

    def test_empty_ids_returns_zero(self):
        """空 IDs 返回 deleted: 0"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.routers.images.task_state") as mock_task:
            mock_task.start_task.return_value = True
            client = TestClient(app)
            response = client.post("/api/delete-images", json={"ids": []})

            assert response.status_code == 200
            assert response.json() == {"deleted": 0}

    def test_task_running_returns_error(self):
        """已有任务运行时返回错误"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.routers.images.task_state") as mock_task:
            mock_task.start_task.return_value = False
            client = TestClient(app)
            response = client.post("/api/delete-images", json={"ids": [1, 2, 3]})

            assert response.status_code == 200
            assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_batch_delete_files_parallel(self):
        """测试批量删除时文件并行删除"""
        from app.routers.images import delete_images
        from app.schemas import DeleteImagesRequest

        mock_session = AsyncMock()
        mock_img1 = MagicMock()
        mock_img1.id = 1
        mock_img1.relative_path = "test1.jpg"
        mock_img2 = MagicMock()
        mock_img2.id = 2
        mock_img2.relative_path = "test2.jpg"

        with patch("app.routers.images.task_state") as mock_task:
            mock_task.start_task.return_value = True
            mock_task.end_task = MagicMock()

            with patch("app.routers.images.IN_CLAUSE_BATCH_SIZE", 10):
                with patch("app.routers.images.PHOTOS_DIR") as mock_photos:
                    with patch("app.routers.images.CACHE_DIR") as mock_cache:
                        with patch("app.routers.images.cache_filename") as mock_cache_name:
                            with patch("app.routers.images.invalidate_folder_tree_cache"):
                                mock_photos.__truediv__ = lambda self, x: MagicMock()
                                mock_cache.__truediv__ = lambda self, x: MagicMock()

                                mock_result = MagicMock()
                                mock_result.scalars.return_value.all.return_value = [mock_img1, mock_img2]
                                mock_session.execute.return_value = mock_result
                                mock_session.delete = AsyncMock()
                                mock_session.commit = AsyncMock()

                                mock_cache_name.side_effect = lambda p: f"cache_{p}"

                                body = DeleteImagesRequest(ids=[1, 2])
                                result = await delete_images(body, mock_session)

                                assert result["deleted"] == 2
                                mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache_called_on_deletion(self):
        """验证删除后调用 invalidate_folder_tree_cache"""
        from app.routers.images import delete_images
        from app.schemas import DeleteImagesRequest

        mock_session = AsyncMock()
        mock_img = MagicMock()
        mock_img.id = 1
        mock_img.relative_path = "test.jpg"

        with patch("app.routers.images.task_state") as mock_task:
            mock_task.start_task.return_value = True
            mock_task.end_task = MagicMock()

            with patch("app.routers.images.IN_CLAUSE_BATCH_SIZE", 10):
                with patch("app.routers.images.PHOTOS_DIR"):
                    with patch("app.routers.images.CACHE_DIR"):
                        with patch("app.routers.images.cache_filename", return_value="cache_test.jpg"):
                            with patch("app.routers.images.invalidate_folder_tree_cache") as mock_invalidate:
                                mock_result = MagicMock()
                                mock_result.scalars.return_value.all.return_value = [mock_img]
                                mock_session.execute.return_value = mock_result
                                mock_session.delete = AsyncMock()
                                mock_session.commit = AsyncMock()

                                body = DeleteImagesRequest(ids=[1])
                                await delete_images(body, mock_session)

                                mock_invalidate.assert_called_once()
