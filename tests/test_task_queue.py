"""TaskQueue 任务队列测试。"""

import asyncio

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

    def test_task_default_values(self):
        """测试任务默认属性"""
        from app.services.task_queue import QueueTask

        task = QueueTask()
        assert task.task_type == ""
        assert task.params == {}
        assert task.priority == 5
        assert task.status == "pending"
        assert task.result is None
        assert task.error is None
        assert task.progress_percent == 0.0


class TestTaskQueue:
    """测试 TaskQueue 核心功能"""

    def setup_method(self):
        """每个测试前重置单例"""
        from app.services.task_queue import TaskQueue

        TaskQueue.reset_instance()

    def test_add_task(self):
        """测试添加任务"""
        from app.services.task_queue import TaskQueue

        queue = TaskQueue()
        queue_id = asyncio.run(queue.add_task("scan", {"path": "/photos"}, priority=10))
        assert queue_id is not None

        status = queue.get_status()
        assert "scan" in status
        assert len(status["scan"]["pending"]) == 1

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


class TestTaskQueueHandlers:
    """测试任务处理器注册"""

    def test_register_handler(self):
        """测试注册任务处理器"""
        from app.services.task_queue import QueueTask, TaskQueue

        async def mock_handler(task: QueueTask) -> dict:
            return {"result": "ok"}

        queue = TaskQueue()
        queue.register_handler("test-task", mock_handler)

        assert "test-task" in queue._task_handlers

    def test_register_handler_creates_semaphore(self):
        """测试注册处理器时创建信号量"""
        from app.services.task_queue import TaskQueue

        async def mock_handler(task) -> dict:
            return {"result": "ok"}

        queue = TaskQueue()
        queue.register_handler("test-task-2", mock_handler)

        assert "test-task-2" in queue._semaphores
        assert queue._semaphores["test-task-2"]._value == 2


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
