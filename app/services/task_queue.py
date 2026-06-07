"""
任务队列服务：支持按类型并发、取消操作。

设计要点：
- 按任务类型分别排队（scan/upload/delete/rename 等）
- 同类型任务串行执行，不同类型可并发
- 内存存储，重启后任务丢失
- 支持任务取消
"""

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.services import task_state

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 2


@dataclass
class QueueTask:
    """队列任务项"""

    queue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    progress_percent: float = 0.0
    current_operation: str = ""
    processed_items: int = 0
    total_items: int = 0


class TaskQueue:
    """
    任务队列管理器

    使用方式：
    queue = TaskQueue()
    queue_id = await queue.add_task("scan", {"path": "/photos"})
    status = await queue.get_status()
    await queue.cancel_task(queue_id)
    """

    _instance: "TaskQueue | None" = None
    _initialized = False

    def __new__(cls) -> "TaskQueue":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._task_handlers: dict[str, Callable[[QueueTask], Coroutine[Any, Any, dict[str, Any]]]] = {}
        self._running_tasks: dict[str, QueueTask] = {}
        self._task_handles: dict[str, asyncio.Task] = {}
        self._worker_lock = asyncio.Lock()
        self._tasks: dict[str, list[QueueTask]] = {}
        self._running: dict[str, QueueTask] = {}
        self._notify_cond: asyncio.Condition = asyncio.Condition()
        self._initialized = True

    def register_handler(
        self,
        task_type: str,
        handler: Callable[[QueueTask], Coroutine[Any, Any, dict[str, Any]]],
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """注册任务处理器"""
        self._task_handlers[task_type] = handler
        self._semaphores[task_type] = asyncio.Semaphore(max_concurrent)

    def _ensure_worker(self, task_type: str) -> None:
        """确保 Worker 正在运行（延迟启动，仅在有事件循环时）"""
        if task_type not in self._workers or self._workers[task_type].done():
            handler = self._task_handlers.get(task_type)
            if handler:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    return
                self._workers[task_type] = loop.create_task(self._run_worker(task_type, handler))

    async def _run_worker(
        self,
        task_type: str,
        handler: Callable[[QueueTask], Coroutine[Any, Any, dict[str, Any]]],
    ) -> None:
        """Worker 循环：等待任务并执行"""
        while True:
            async with self._notify_cond:
                await self._notify_cond.wait_for(
                    lambda: bool(self._tasks.get(task_type))
                )
            async with self._worker_lock:
                pending = self._tasks.get(task_type, [])
                if not pending:
                    continue

                task = pending.pop(0)
                self._tasks[task_type] = pending

                task.status = "running"
                task.started_at = datetime.now().isoformat()
                self._running[task_type] = task
                self._running_tasks[task.queue_id] = task

            task_state.start_task(task_type)

            sem = self._semaphores.get(task_type)
            if sem:
                async with sem:
                    handle: asyncio.Task | None = None
                    try:
                        handle = asyncio.ensure_future(handler(task))
                        self._task_handles[task.queue_id] = handle
                        result = await handle
                        task.status = "completed"
                        task.result = result
                        task.progress_percent = 100.0
                        task.current_operation = "已完成"
                    except asyncio.CancelledError:
                        task.status = "cancelled"
                        task.error = "任务被取消"
                        if handle is not None and not handle.done():
                            handle.cancel()
                    except Exception as e:
                        logger.exception(f"[queue] {task_type} 任务失败: {e}")
                        task.status = "failed"
                        task.error = str(e)
                    finally:
                        task.finished_at = datetime.now().isoformat()
                        if task.status == "completed":
                            task_state.end_task(task.result or {})
                        else:
                            task_state.fail_task(task.error or "未知错误")
                        async with self._worker_lock:
                            if task_type in self._running:
                                del self._running[task_type]
                            if task.queue_id in self._running_tasks:
                                del self._running_tasks[task.queue_id]
                            if task.queue_id in self._task_handles:
                                del self._task_handles[task.queue_id]

    async def add_task(
        self,
        task_type: str,
        params: dict[str, Any] | None = None,
        priority: int = 5,
    ) -> str:
        """添加任务到队列，返回 queue_id"""
        params = params or {}
        task = QueueTask(
            task_type=task_type,
            params=params,
            priority=priority,
        )

        async with self._worker_lock:
            if task_type not in self._tasks:
                self._tasks[task_type] = []
            self._tasks[task_type].append(task)

        self._ensure_worker(task_type)
        async with self._notify_cond:
            self._notify_cond.notify_all()
        logger.info(f"[queue] 添加任务: {task_type}, queue_id={task.queue_id}")
        return task.queue_id

    def get_status(self) -> dict[str, Any]:
        """获取队列状态"""
        result: dict[str, Any] = {}

        for task_type, pending in self._tasks.items():
            running = self._running.get(task_type)
            result[task_type] = {
                "running": running.__dict__ if running else None,
                "pending": [p.__dict__ for p in pending],
                "pending_count": len(pending),
            }

        for task_type, running_task in self._running.items():
            if task_type not in result:
                result[task_type] = {
                    "running": running_task.__dict__,
                    "pending": [],
                    "pending_count": 0,
                }

        return result

    def get_task_status(self, queue_id: str) -> dict[str, Any] | None:
        """获取指定任务状态"""
        for pending_list in self._tasks.values():
            for task in pending_list:
                if task.queue_id == queue_id:
                    return {
                        "queue_id": task.queue_id,
                        "task_type": task.task_type,
                        "status": task.status,
                        "progress_percent": task.progress_percent,
                        "current_operation": task.current_operation,
                    }
        for task in self._running.values():
            if task.queue_id == queue_id:
                return {
                    "queue_id": task.queue_id,
                    "task_type": task.task_type,
                    "status": task.status,
                    "progress_percent": task.progress_percent,
                    "current_operation": task.current_operation,
                }
        return None

    def cancel_task(self, queue_id: str) -> bool:
        """取消任务"""
        for task_type, pending in list(self._tasks.items()):
            for i, task in enumerate(pending):
                if task.queue_id == queue_id:
                    task.status = "cancelled"
                    task.finished_at = datetime.now().isoformat()
                    task.error = "任务被取消"
                    self._tasks[task_type].pop(i)
                    logger.info(f"[queue] 取消待执行任务: {queue_id}")
                    return True

        for task_type, task in list(self._running.items()):
            if task.queue_id == queue_id:
                task.status = "cancelled"
                task.finished_at = datetime.now().isoformat()
                task.error = "任务被取消"
                del self._running[task_type]

                handle = self._task_handles.pop(queue_id, None)
                if handle is not None and not handle.done():
                    handle.cancel()

                if queue_id in self._running_tasks:
                    t = self._running_tasks[queue_id]
                    t.status = "cancelled"
                logger.info(f"[queue] 取消运行中任务: {queue_id}")
                return True

        return False

    def clear_completed(self) -> None:
        """清理已完成任务"""
        self._tasks = {k: v for k, v in self._tasks.items() if v}
        self._running = {}

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（仅用于测试）"""
        cls._instance = None
        cls._initialized = False


async def get_task_queue() -> TaskQueue:
    """获取任务队列实例"""
    return TaskQueue()
