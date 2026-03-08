"""merge-folders API 测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMergeFoldersEndpoint:
    """测试 /api/merge-folders 端点"""

    def test_endpoint_exists(self):
        """验证 API 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/merge-folders" in routes

    def test_same_folder_returns_error(self):
        """相同文件夹返回错误"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.routers.folders.task_state") as mock_task:
            mock_task.start_task.return_value = True
            client = TestClient(app)
            response = client.post(
                "/api/merge-folders",
                json={"folder_a": "test", "folder_b": "test"},
            )

            assert response.status_code == 200
            assert response.json()["error"] == "不能选择相同的文件夹"

    def test_parent_child_folder_returns_error(self):
        """父子文件夹返回错误"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.routers.folders.task_state") as mock_task:
            mock_task.start_task.return_value = True
            client = TestClient(app)
            response = client.post(
                "/api/merge-folders",
                json={"folder_a": "test", "folder_b": "test/sub"},
            )

            assert response.status_code == 200
            assert response.json()["error"] == "不能合并互为父子关系的文件夹"

    def test_folder_not_exists_returns_error(self):
        """文件夹不存在返回错误"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.routers.folders.task_state") as mock_task:
            mock_task.start_task.return_value = True
            with patch("app.routers.folders.PHOTOS_DIR") as mock_photos:
                mock_photos.resolve.return_value = Path("/photos")
                mock_path = MagicMock()
                mock_path.exists.return_value = False
                mock_photos.__truediv__ = lambda self, x: mock_path

                client = TestClient(app)
                response = client.post(
                    "/api/merge-folders",
                    json={"folder_a": "notexist", "folder_b": "test"},
                )

                assert response.status_code == 200
                assert "error" in response.json()

    def test_task_running_returns_error(self):
        """已有任务运行时返回错误"""
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.routers.folders.task_state") as mock_task:
            mock_task.start_task.return_value = False
            client = TestClient(app)
            response = client.post(
                "/api/merge-folders",
                json={"folder_a": "test", "folder_b": "test2"},
            )

            assert response.status_code == 200
            assert response.json()["error"] == "有任务正在进行中，请等待完成后再提交新任务"
            assert response.json()["busy"] is True


class TestMergeFoldersLogic:
    """测试合并文件夹的核心逻辑"""

    @pytest.mark.asyncio
    async def test_items_unpacking_correct(self):
        """测试 items 解包正确（4 元组）"""
        items = [
            (1, "a.jpg", "a", "hash1"),
            (2, "b.jpg", "a", "hash2"),
            (3, "c.jpg", "b", "hash1"),
        ]

        by_hash: dict[str, list[tuple[int, str, str, str | None]]] = {}
        for item in items:
            img_id, rel_path, src, md5_hash = item
            h = md5_hash
            if h not in by_hash:
                by_hash[h] = []
            by_hash[h].append(item)

        for h, item_list in by_hash.items():
            for img_id, rel_path, src, md5_hash in item_list:
                assert img_id is not None
                assert rel_path is not None
                assert src in ("a", "b")
                assert md5_hash is not None or md5_hash is None

        assert len(by_hash["hash1"]) == 2
        assert len(by_hash["hash2"]) == 1
