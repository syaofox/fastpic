"""TaskQueue 任务队列测试。"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class TestQueueTask:
    """测试 QueueTask 数据类"""

    def test_create_task(self):
        """测试创建任务"""
        from app.services.task_queue import QueueTask

        task = QueueTask(task_type="scan", params={"path": "/photos"})
        assert task.task_type == "scan"
        assert task.params == {"path": "/photos"}
        assert task.status == "pending"
        assert task.queue_id is not None

    def test_task_to_dict(self):
        """测试任务序列化"""
        from app.services.task_queue import QueueTask

        task = QueueTask(task_type="scan", params={"path": "/photos"})
        data = task.to_dict()
        assert data["task_type"] == "scan"
        assert data["params"] == {"path": "/photos"}
        assert data["status"] == "pending"
        assert "queue_id" in data

    def test_task_from_dict(self):
        """测试任务反序列化"""
        from app.services.task_queue import QueueTask

        data = {
            "queue_id": "test-id",
            "task_type": "scan",
            "params": {"path": "/photos"},
            "priority": 10,
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
        }
        task = QueueTask.from_dict(data)
        assert task.queue_id == "test-id"
        assert task.task_type == "scan"
        assert task.params == {"path": "/photos"}
        assert task.status == "running"


class TestTaskQueue:
    """测试 TaskQueue 核心功能"""

    @pytest.fixture(autouse=True)
    def setup_temp_queue_file(self, tmp_path):
        """使用临时文件避免污染"""
        from app.services import task_queue as tq_module

        self._original_file = tq_module.QUEUE_FILE
        tq_module.QUEUE_FILE = tmp_path / "test_queue.json"
        yield
        tq_module.QUEUE_FILE = self._original_file

    def test_add_task(self):
        """测试添加任务"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        queue_id = asyncio.run(queue.add_task("scan", {"path": "/photos"}, priority=10))
        assert queue_id is not None

        status = queue.get_status()
        assert "scan" in status
        assert len(status["scan"]["pending"]) == 1
        assert status["scan"]["pending"][0]["task_type"] == "scan"

    def test_add_multiple_tasks_same_type(self):
        """测试同类型多任务排队"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        asyncio.run(queue.add_task("scan", priority=10))
        asyncio.run(queue.add_task("scan", priority=10))
        asyncio.run(queue.add_task("scan", priority=10))

        status = queue.get_status()
        assert status["scan"]["pending_count"] == 3

    def test_add_different_task_types(self):
        """测试不同类型任务可同时存在"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        asyncio.run(queue.add_task("scan", priority=10))
        asyncio.run(queue.add_task("cleanup", priority=10))
        asyncio.run(queue.add_task("upload", priority=10))

        status = queue.get_status()
        assert "scan" in status
        assert "cleanup" in status
        assert "upload" in status

    def test_get_task_status(self):
        """测试获取指定任务状态"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        queue_id = asyncio.run(queue.add_task("scan", priority=10))

        task_status = queue.get_task_status(queue_id)
        assert task_status is not None
        assert task_status["queue_id"] == queue_id

    def test_get_task_status_not_found(self):
        """测试获取不存在的任务"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        task_status = queue.get_task_status("non-existent-id")
        assert task_status is None

    def test_cancel_pending_task(self):
        """测试取消待执行任务"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        queue_id = asyncio.run(queue.add_task("scan", priority=10))

        result = queue.cancel_task(queue_id)
        assert result is True

        status = queue.get_status()
        assert status["scan"]["pending_count"] == 0

    def test_cancel_already_cancelled(self):
        """测试取消已取消的任务"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        queue_id = asyncio.run(queue.add_task("scan", priority=10))
        queue.cancel_task(queue_id)

        result = queue.cancel_task(queue_id)
        assert result is False


class TestTaskQueuePersistence:
    """测试队列持久化"""

    @pytest.fixture(autouse=True)
    def setup_temp_queue_file(self, tmp_path):
        """使用临时文件"""
        from app.services import task_queue as tq_module

        self._original_file = tq_module.QUEUE_FILE
        tq_module.QUEUE_FILE = tmp_path / "test_queue_persist.json"
        yield
        tq_module.QUEUE_FILE = self._original_file

    def test_persistence(self):
        """测试任务持久化到文件"""
        from app.services import task_queue as tq_module
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        asyncio.run(queue.add_task("scan", {"path": "/photos"}, priority=10))

        assert tq_module.QUEUE_FILE.exists()

        with open(tq_module.QUEUE_FILE, encoding="utf-8") as f:
            data = json.load(f)
            assert "scan" in data["tasks"]
            assert len(data["tasks"]["scan"]) == 1


class TestTaskQueueHandlers:
    """测试任务处理器注册"""

    @pytest.fixture(autouse=True)
    def setup_temp_queue_file(self, tmp_path):
        """使用临时文件"""
        from app.services import task_queue

        self._original_file = task_queue.QUEUE_FILE
        task_queue.QUEUE_FILE = tmp_path / "test_queue_handler.json"
        yield
        task_queue.QUEUE_FILE = self._original_file

    def test_register_handler(self):
        """测试注册任务处理器"""
        from app.services.task_queue import QueueTask, TaskQueue

        async def mock_handler(task: QueueTask) -> dict:
            return {"result": "ok"}

        queue = TaskQueue()
        queue.register_handler("test-task", mock_handler)

        assert "test-task" in queue._task_handlers


class TestQueueAPI:
    """测试队列 API 端点"""

    def test_queue_status_endpoint_exists(self):
        """验证队列状态端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/queue-status" in routes

    def test_queue_cancel_endpoint_exists(self):
        """验证队列取消端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/api/queue-cancel" in routes


class TestSettingsQueueIntegration:
    """测试设置页面队列集成"""

    def test_scan_endpoint_returns_queue_info(self):
        """测试扫描端点返回队列信息"""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/scan")

        assert response.status_code == 200
        data = response.json()
        assert "queue_id" in data
        assert "status" in data

    def test_cleanup_endpoint_returns_queue_info(self):
        """测试清理端点返回队列信息"""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/api/cleanup")

        assert response.status_code == 200
        data = response.json()
        assert "queue_id" in data
        assert "status" in data

    def test_full_sync_endpoint_returns_queue_info(self):
        """测试完整同步端点返回队列信息"""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/api/full-sync")

        assert response.status_code == 200
        data = response.json()
        assert "queue_id" in data
        assert "status" in data

    def test_scan_duplicates_endpoint_returns_queue_info(self):
        """测试扫描重复文件端点返回队列信息"""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/api/scan-duplicates")

        assert response.status_code == 200
        data = response.json()
        assert "queue_id" in data
        assert "status" in data

    def test_merge_folders_endpoint_returns_queue_info(self):
        """测试合并文件夹端点返回队列信息"""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/api/merge-folders",
            json={"folder_a": "test1", "folder_b": "test2", "target": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "queue_id" in data
        assert "status" in data
