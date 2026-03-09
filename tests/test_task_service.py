"""task_service 测试。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.task_service import TaskService, TaskContext


class TestTaskContext:
    """测试 TaskContext 数据类"""

    def test_task_context_creation(self):
        """创建 TaskContext"""
        ctx = TaskContext(task_type="scan", title="扫描中", total_items=100)
        assert ctx.task_type == "scan"
        assert ctx.title == "扫描中"
        assert ctx.total_items == 100
        assert ctx.processed_items == 0

    @pytest.mark.asyncio
    async def test_broadcast_methods(self):
        """测试广播方法"""
        ctx = TaskContext(task_type="scan", title="扫描中", total_items=100)

        with patch("app.services.task_service.broadcaster") as mock_broadcaster:
            mock_broadcaster.broadcast_task_start = AsyncMock()
            await ctx.broadcast_start()

            mock_broadcaster.broadcast_task_start.assert_called_once_with("scan", "扫描中", 100)


class TestTaskService:
    """测试 TaskService 类"""

    def test_task_service_creation(self):
        """创建 TaskService"""
        service = TaskService()
        assert service._handlers == {}

    def test_register_handler(self):
        """注册任务处理器"""
        service = TaskService()

        @service.register("test-task")
        async def handler(ctx, params):
            return {"result": "ok"}

        assert "test-task" in service._handlers

    @pytest.mark.asyncio
    async def test_execute_unknown_task(self):
        """执行未知任务类型"""
        service = TaskService()

        with patch("app.services.task_service.task_state") as mock_state:
            mock_state.start_task.return_value = True

            result = await service.execute("unknown-task", "测试", {}, 10)
            assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_busy_task(self):
        """执行时任务繁忙"""
        service = TaskService()

        with patch("app.services.task_service.task_state") as mock_state:
            mock_state.start_task.return_value = False

            result = await service.execute("scan", "扫描", {}, 10)
            assert result == {"error": "有任务正在进行中"}

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """任务执行成功"""
        service = TaskService()

        @service.register("test-task")
        async def handler(ctx, params):
            return {"processed": 10}

        with patch("app.services.task_service.task_state") as mock_state:
            with patch("app.services.task_service.broadcaster") as mock_broadcaster:
                mock_state.start_task.return_value = True
                mock_state.end_task = MagicMock()  # 同步函数
                mock_broadcaster.broadcast_task_start = AsyncMock()
                mock_broadcaster.broadcast_task_complete = AsyncMock()

                result = await service.execute("test-task", "测试任务", {}, 10)

                assert result == {"processed": 10}
                mock_broadcaster.broadcast_task_start.assert_called_once()
                mock_broadcaster.broadcast_task_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """任务执行失败"""
        service = TaskService()

        @service.register("test-task")
        async def handler(ctx, params):
            raise ValueError("测试错误")

        with patch("app.services.task_service.task_state") as mock_state:
            with patch("app.services.task_service.broadcaster") as mock_broadcaster:
                mock_state.start_task.return_value = True
                mock_state.fail_task = MagicMock()  # 同步函数
                mock_broadcaster.broadcast_task_start = AsyncMock()
                mock_broadcaster.broadcast_task_error = AsyncMock()

                result = await service.execute("test-task", "测试任务", {}, 10)

                assert "error" in result
                assert "测试错误" in result["error"]
                mock_broadcaster.broadcast_task_error.assert_called_once()

    def test_format_result_message(self):
        """测试结果消息格式化"""
        service = TaskService()

        assert service._format_result_message("scan", {"scanned": 100}) == "扫描完成，发现 100 个文件"
        assert service._format_result_message("cleanup", {"stale_removed": 50}) == "清理完成，移除 50 条记录"
        assert service._format_result_message("delete-images", {"deleted": 10}) == "已删除 10 项"
        assert service._format_result_message("move-images", {"moved": 5}) == "已移动 5 项"
        assert service._format_result_message("unknown", {}) == "操作完成"
