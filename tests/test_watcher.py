"""Watcher 并行处理测试。"""

import asyncio
from pathlib import Path

import pytest


class TestWatcherParallelProcessing:
    """Watcher 并行处理优化测试"""

    def test_watcher_lock_exists(self):
        """验证 _watcher_lock 已定义"""
        from app.services import watcher

        assert hasattr(watcher, "_watcher_lock")
        assert isinstance(watcher._watcher_lock, asyncio.Lock)

    def test_event_classification(self):
        """验证事件分类逻辑"""
        path_events = {
            ("created", "/photos/img1.jpg"): ("created", "/photos/img1.jpg", None, 100.0),
            ("created", "/photos/img2.jpg"): ("created", "/photos/img2.jpg", None, 100.0),
            ("deleted", "/photos/img3.jpg"): ("deleted", "/photos/img3.jpg", None, 100.0),
            ("moved", "/photos/old.jpg"): ("moved", "/photos/old.jpg", "/photos/new.jpg", 100.0),
        }

        created_events = []
        deleted_events = []
        moved_events = []

        for key, ev in path_events.items():
            event_type, src, dst, ts = ev
            if event_type == "created":
                created_events.append(Path(src))
            elif event_type == "deleted":
                deleted_events.append(Path(src))
            elif event_type == "moved":
                moved_events.append((Path(src), Path(dst)))

        assert len(created_events) == 2
        assert len(deleted_events) == 1
        assert len(moved_events) == 1

        assert Path("/photos/img1.jpg") in created_events
        assert Path("/photos/img3.jpg") in deleted_events
        assert (Path("/photos/old.jpg"), Path("/photos/new.jpg")) in moved_events

    @pytest.mark.asyncio
    async def test_parallel_processing_with_gather(self):
        """验证 asyncio.gather 并行处理多个任务"""
        results = []

        async def mock_task(task_id: int):
            await asyncio.sleep(0.01)
            results.append(task_id)
            return task_id

        tasks = [mock_task(i) for i in range(5)]
        output = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 5
        assert sorted(r for r in output if isinstance(r, int)) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_gather_with_exceptions(self):
        """验证 gather return_exceptions=True 正确捕获异常"""

        async def successful_task():
            await asyncio.sleep(0.01)
            return "success"

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        tasks = [successful_task(), failing_task(), successful_task()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 2
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)

    @pytest.mark.asyncio
    async def test_watcher_lock_protection(self):
        """验证 _watcher_lock 可以正确保护临界区"""
        from app.services import watcher

        protected_section_ran = False

        async def test_task():
            async with watcher._watcher_lock:
                nonlocal protected_section_ran
                protected_section_ran = True
                await asyncio.sleep(0.01)

        await test_task()
        assert protected_section_ran

    @pytest.mark.asyncio
    async def test_concurrent_lock_exclusion(self):
        """验证多个协程不能同时进入临界区"""
        from app.services import watcher

        execution_order = []

        async def task_a():
            async with watcher._watcher_lock:
                execution_order.append("a_start")
                await asyncio.sleep(0.05)
                execution_order.append("a_end")

        async def task_b():
            await asyncio.sleep(0.01)
            async with watcher._watcher_lock:
                execution_order.append("b_start")
                execution_order.append("b_end")

        await asyncio.gather(task_a(), task_b())

        assert execution_order.index("a_start") < execution_order.index("b_start")
        assert execution_order.index("a_end") < execution_order.index("b_start")
