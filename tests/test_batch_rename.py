"""batch-rename API 测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBatchRenameEndpoint:
    """测试 /api/batch-rename 端点"""

    def test_endpoint_exists(self):
        """验证 API 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/batch-rename" in routes

    def test_empty_request_returns_zero(self):
        """空请求返回 count: 0"""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/api/batch-rename", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["folder_count"] == 0
        assert data["image_count"] == 0
        assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_batch_rename_commits_every_50_images(self):
        """测试每处理 50 张图片后 commit 一次"""
        from app.routers.folders import batch_rename
        from app.schemas import BatchRenameRequest, ImageRenameItem

        mock_session = AsyncMock()
        commit_count = 0

        def track_commit():
            nonlocal commit_count
            commit_count += 1

        mock_session.commit.side_effect = track_commit
        mock_session.add = MagicMock()

        mock_images = []
        for i in range(120):
            img = MagicMock()
            img.id = i + 1
            img.filename = f"test{i + 1}.jpg"
            img.relative_path = f"test{i + 1}.jpg"
            mock_images.append(img)

        with patch("app.routers.folders.BATCH_COMMIT_SIZE", 50):
            with patch("app.routers.folders.PHOTOS_DIR"):
                with patch("app.routers.folders.CACHE_DIR"):
                    with patch("app.routers.folders.IMAGE_EXTENSIONS", {".jpg", ".png"}):
                        with patch("app.routers.folders.VIDEO_EXTENSIONS", set()):
                            with patch("app.routers.folders.invalid_filename", return_value=False):
                                with patch("app.routers.folders.unique_path") as mock_unique:
                                    with patch("app.routers.folders.asyncio") as mock_asyncio:
                                        mock_asyncio.to_thread = AsyncMock()

                                        with patch(
                                            "app.routers.folders.update_image_path_and_regenerate_thumbnail",
                                            new_callable=AsyncMock,
                                        ):
                                            mock_unique.side_effect = lambda *args, **kwargs: MagicMock(
                                                name="unique_path"
                                            )

                                            mock_result = MagicMock()
                                            mock_result.scalars.return_value.all.return_value = mock_images
                                            mock_session.execute.return_value = mock_result

                                            rename_items = [
                                                ImageRenameItem(id=i + 1, new_filename=f"renamed{i + 1}.jpg")
                                                for i in range(120)
                                            ]
                                            body = BatchRenameRequest(image_renames=rename_items)

                                            await batch_rename(body, mock_session)

                                            assert commit_count >= 2

    @pytest.mark.asyncio
    async def test_batch_rename_queries_all_images_at_once(self):
        """测试图片重命名时一次性批量查询所有图片"""
        from app.routers.folders import batch_rename
        from app.schemas import BatchRenameRequest, ImageRenameItem

        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        mock_img = MagicMock()
        mock_img.id = 1
        mock_img.filename = "test.jpg"
        mock_img.relative_path = "test.jpg"

        with patch("app.routers.folders.BATCH_COMMIT_SIZE", 50):
            with patch("app.routers.folders.PHOTOS_DIR"):
                with patch("app.routers.folders.CACHE_DIR"):
                    with patch("app.routers.folders.IMAGE_EXTENSIONS", {".jpg", ".png"}):
                        with patch("app.routers.folders.VIDEO_EXTENSIONS", set()):
                            with patch("app.routers.folders.invalid_filename", return_value=False):
                                with patch("app.routers.folders.unique_path") as mock_unique:
                                    with patch("app.routers.folders.asyncio") as mock_asyncio:
                                        mock_asyncio.to_thread = AsyncMock()

                                        with patch(
                                            "app.routers.folders.update_image_path_and_regenerate_thumbnail",
                                            new_callable=AsyncMock,
                                        ):
                                            mock_unique.side_effect = lambda *args, **kwargs: MagicMock(
                                                name="unique_path"
                                            )

                                            mock_result = MagicMock()
                                            mock_result.scalars.return_value.all.return_value = [mock_img]
                                            mock_session.execute.return_value = mock_result

                                            rename_items = [ImageRenameItem(id=1, new_filename="renamed.jpg")]
                                            body = BatchRenameRequest(image_renames=rename_items)

                                            await batch_rename(body, mock_session)

                                            assert mock_session.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_folder_rename_processes_correctly(self):
        """测试文件夹重命名基本流程"""
        from app.routers.folders import batch_rename
        from app.schemas import BatchRenameRequest, FolderRenameItem

        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        mock_img = MagicMock()
        mock_img.relative_path = "test_folder"

        async def async_iter():
            yield [mock_img]

        with patch("app.routers.folders.PHOTOS_DIR"):
            with patch("app.routers.folders.normalize_path", return_value="test_folder"):
                with patch("app.routers.folders.iter_images_by_path_prefix") as mock_iter:
                    mock_iter.return_value = async_iter()

                    with patch("app.routers.folders.invalidate_folder_tree_cache"):
                        with patch("app.routers.folders.Path") as mock_path_cls:
                            mock_path_instance = MagicMock()
                            mock_path_instance.exists.return_value = True
                            mock_path_instance.is_dir.return_value = True
                            mock_path_instance.resolve.return_value = mock_path_instance
                            mock_path_cls.return_value = mock_path_instance

                            with patch("app.routers.folders.asyncio") as mock_asyncio:
                                mock_asyncio.to_thread = AsyncMock()

                                with patch(
                                    "app.routers.folders.update_image_path_and_regenerate_thumbnail",
                                    new_callable=AsyncMock,
                                ):
                                    rename_items = [FolderRenameItem(path="test_folder", new_name="renamed_folder")]
                                    body = BatchRenameRequest(folder_renames=rename_items)

                                    result = await batch_rename(body, mock_session)

                                    assert result["folder_count"] == 1


class TestBatchCommitSize:
    """测试 BATCH_COMMIT_SIZE 配置"""

    def test_batch_commit_size_exists(self):
        """验证 BATCH_COMMIT_SIZE 配置存在"""
        from app.config import BATCH_COMMIT_SIZE

        assert BATCH_COMMIT_SIZE == 50
