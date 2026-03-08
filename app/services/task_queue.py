"""
任务队列服务：支持按类型并发、持久化、取消操作。

设计要点：
- 按任务类型分别排队（scan/upload/delete/rename 等）
- 同类型任务串行执行，不同类型可并发
- 文件持久化，重启后可恢复
- 支持任务取消
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services import task_state

logger = logging.getLogger(__name__)

QUEUE_FILE = Path("task_queue.json")
_lock = threading.Lock()

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "task_type": self.task_type,
            "params": self.params,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "progress_percent": self.progress_percent,
            "current_operation": self.current_operation,
            "processed_items": self.processed_items,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueTask":
        return cls(**data)


@dataclass
class QueueState:
    """队列状态"""

    tasks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    running: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "running": self.running,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueState":
        return cls(
            tasks=data.get("tasks", {}),
            running=data.get("running", {}),
        )


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
        self._worker_lock = asyncio.Lock()
        self._initialized = True

        self._ensure_data_dir()
        self._load_state()

    def _ensure_data_dir(self) -> None:
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _read_state(self) -> QueueState:
        self._ensure_data_dir()
        if not QUEUE_FILE.exists():
            return QueueState()
        try:
            with open(QUEUE_FILE, encoding="utf-8") as f:
                data = json.load(f)
                return QueueState.from_dict(data)
        except (OSError, json.JSONDecodeError):
            return QueueState()

    def _write_state(self, state: QueueState) -> None:
        self._ensure_data_dir()
        data = state.to_dict()
        tmp_file = None
        try:
            fd, tmp_file = tempfile.mkstemp(dir=QUEUE_FILE.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, QUEUE_FILE)
        except Exception:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)
            raise

    def _load_state(self) -> None:
        """加载状态并恢复运行中的任务"""
        state = self._read_state()
        for task_type, task_data in state.running.items():
            if task_data.get("status") == "running":
                task = QueueTask.from_dict(task_data)
                self._running_tasks[task.queue_id] = task

    def register_handler(
        self,
        task_type: str,
        handler: Callable[[QueueTask], Coroutine[Any, Any, dict[str, Any]]],
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """注册任务处理器"""

        def wrapper():
            return self._run_worker(task_type, handler)

        self._task_handlers[task_type] = handler
        self._semaphores[task_type] = asyncio.Semaphore(max_concurrent)
        self._ensure_worker(task_type)

    def _ensure_worker(self, task_type: str) -> None:
        """确保 Worker 正在运行"""
        if task_type not in self._workers or self._workers[task_type].done():
            handler = self._task_handlers.get(task_type)
            if handler:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                self._workers[task_type] = loop.create_task(self._run_worker(task_type, handler))

    async def _run_worker(
        self,
        task_type: str,
        handler: Callable[[QueueTask], Coroutine[Any, Any, dict[str, Any]]],
    ) -> None:
        """Worker 循环：等待任务并执行"""
        while True:
            await asyncio.sleep(0.1)
            async with self._worker_lock:
                state = self._read_state()
                pending = state.tasks.get(task_type, [])
                if not pending:
                    continue

                task_data = pending[0]
                task = QueueTask.from_dict(task_data)

                task.status = "running"
                task.started_at = datetime.now().isoformat()
                state.tasks[task_type] = pending[1:]
                state.running[task_type] = task.to_dict()
                self._write_state(state)

                self._running_tasks[task.queue_id] = task

            task_state.start_task(task_type)

            sem = self._semaphores.get(task_type)
            if sem:
                async with sem:
                    try:
                        result = await handler(task)
                        task.status = "completed"
                        task.result = result
                        task.progress_percent = 100.0
                        task.current_operation = "已完成"
                    except asyncio.CancelledError:
                        task.status = "cancelled"
                        task.error = "任务被取消"
                        raise
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
                            state = self._read_state()
                            if task_type in state.running:
                                del state.running[task_type]
                            self._write_state(state)
                            if task.queue_id in self._running_tasks:
                                del self._running_tasks[task.queue_id]

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
            state = self._read_state()
            if task_type not in state.tasks:
                state.tasks[task_type] = []
            state.tasks[task_type].append(task.to_dict())
            self._write_state(state)

        self._ensure_worker(task_type)
        logger.info(f"[queue] 添加任务: {task_type}, queue_id={task.queue_id}")
        return task.queue_id

    def get_status(self) -> dict[str, Any]:
        """获取队列状态"""
        state = self._read_state()
        result: dict[str, Any] = {}

        for task_type, pending in state.tasks.items():
            result[task_type] = {
                "running": state.running.get(task_type),
                "pending": pending,
                "pending_count": len(pending),
            }

        for task_type, running_data in state.running.items():
            if task_type not in result:
                result[task_type] = {
                    "running": running_data,
                    "pending": [],
                    "pending_count": 0,
                }

        return result

    def get_task_status(self, queue_id: str) -> dict[str, Any] | None:
        """获取指定任务状态"""
        state = self._read_state()
        for pending_list in state.tasks.values():
            for task_data in pending_list:
                if task_data.get("queue_id") == queue_id:
                    return task_data
        for running_data in state.running.values():
            if running_data.get("queue_id") == queue_id:
                return running_data
        return None

    def cancel_task(self, queue_id: str) -> bool:
        """取消任务"""
        state = self._read_state()

        for task_type, pending in state.tasks.items():
            for i, task_data in enumerate(pending):
                if task_data.get("queue_id") == queue_id:
                    task_data["status"] = "cancelled"
                    task_data["finished_at"] = datetime.now().isoformat()
                    task_data["error"] = "任务被取消"
                    state.tasks[task_type].pop(i)
                    self._write_state(state)
                    logger.info(f"[queue] 取消待执行任务: {queue_id}")
                    return True

        for task_type, running_data in list(state.running.items()):
            if running_data.get("queue_id") == queue_id:
                running_data["status"] = "cancelled"
                running_data["finished_at"] = datetime.now().isoformat()
                running_data["error"] = "任务被取消"
                del state.running[task_type]
                self._write_state(state)

                if queue_id in self._running_tasks:
                    task = self._running_tasks[queue_id]
                    task.status = "cancelled"
                logger.info(f"[queue] 取消运行中任务: {queue_id}")
                return True

        return False

    def clear_completed(self) -> None:
        """清理已完成任务"""
        state = self._read_state()
        state.tasks = {k: v for k, v in state.tasks.items() if v}
        state.running = {}
        self._write_state(state)


async def get_task_queue() -> TaskQueue:
    """获取任务队列实例"""
    return TaskQueue()
