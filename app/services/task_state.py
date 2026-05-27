"""
任务状态服务：用于设置页面费时操作的状态展示，
防止页面刷新后 UI 状态丢失导致重复提交。

支持任务队列，一次只能执行一个费时操作。
通过 WebSocket 实时推送进度。
"""

import asyncio
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.services.message_broadcaster import broadcaster

TASK_TITLES = {
    "scan": "正在扫描媒体文件",
    "cleanup": "正在清理数据库",
    "full-sync": "正在完整同步",
    "scan-duplicates": "正在扫描重复文件",
    "upload": "正在上传文件",
    "delete": "正在删除文件",
}

_progress_callbacks: list[Callable[[dict], None]] = []
_queue: asyncio.Queue | None = None
_lock = threading.Lock()


def _get_queue() -> asyncio.Queue | None:
    global _queue
    if _queue is None:
        try:
            asyncio.get_running_loop()
            _queue = asyncio.Queue()
        except RuntimeError:
            pass
    return _queue


async def _get_queue_async() -> asyncio.Queue:
    """异步获取或创建队列"""
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


def register_progress_callback(callback: Callable[[dict], None]) -> None:
    """注册进度回调函数"""
    if callback not in _progress_callbacks:
        _progress_callbacks.append(callback)


def unregister_progress_callback(callback: Callable[[dict], None]) -> None:
    """注销进度回调函数"""
    if callback in _progress_callbacks:
        _progress_callbacks.remove(callback)


async def emit_progress(**kwargs) -> None:
    """发送进度更新到所有订阅者"""
    data = {"timestamp": datetime.now().isoformat(), **kwargs}
    for callback in _progress_callbacks:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception:
            pass
    q = await _get_queue_async()
    if not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await q.put(data)

    try:
        task_type = kwargs.get("task_type", "")
        processed = kwargs.get("processed_items", 0)
        total = kwargs.get("total_items", 0)
        operation = kwargs.get("current_operation", "")
        if task_type:
            await broadcaster.broadcast_task_progress(task_type, processed, total, operation)
    except Exception:
        pass


@dataclass
class TaskState:
    task_id: str | None = None
    task_type: str | None = None
    title: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    current_operation: str | None = None
    progress_percent: float = 0.0
    total_items: int = 0
    processed_items: int = 0


_state: TaskState = TaskState()


def get_queue_status() -> dict[str, Any]:
    """获取任务队列状态"""
    global _state
    is_running = _state.finished_at is None and _state.task_type is not None

    result = {
        "is_running": is_running,
        "current_task": _state.task_type,
    }

    if _state.task_type and _state.started_at:
        result["started_at"] = _state.started_at

    if is_running:
        result["status"] = "running"
    else:
        result["status"] = "idle"

    if _state.current_operation:
        result["current_operation"] = _state.current_operation
    if _state.progress_percent:
        result["progress_percent"] = _state.progress_percent
    if _state.total_items:
        result["total_items"] = _state.total_items
    if _state.processed_items:
        result["processed_items"] = _state.processed_items

    return result


def is_busy() -> bool:
    """检查是否有任务正在进行"""
    global _state
    return _state.finished_at is None and _state.task_type is not None


def start_task(task_type: str, total_items: int = 0, title: str | None = None) -> bool:
    """标记任务开始，返回是否成功启动（如果忙则返回 False）"""
    global _state

    if title is None:
        title = TASK_TITLES.get(task_type, "正在处理...")

    with _lock:
        if is_busy():
            return False
        _state = TaskState(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            title=title,
            started_at=datetime.now().isoformat(),
            total_items=total_items,
        )
    return True


def update_progress(
    current_operation: str | None = None,
    progress_percent: float | None = None,
    processed_items: int | None = None,
    total_items: int | None = None,
) -> None:
    """更新任务进度"""
    global _state
    with _lock:
        _state = TaskState(
            task_id=_state.task_id,
            task_type=_state.task_type,
            started_at=_state.started_at,
            finished_at=_state.finished_at,
            result=_state.result,
            error=_state.error,
            current_operation=current_operation if current_operation is not None else _state.current_operation,
            progress_percent=progress_percent if progress_percent is not None else _state.progress_percent,
            total_items=total_items if total_items is not None else _state.total_items,
            processed_items=processed_items if processed_items is not None else _state.processed_items,
        )


async def async_update_progress(
    current_operation: str | None = None,
    progress_percent: float | None = None,
    processed_items: int | None = None,
    total_items: int | None = None,
) -> None:
    """异步更新任务进度（同步状态 + WebSocket 推送）"""
    update_progress(current_operation, progress_percent, processed_items, total_items)
    await emit_progress(
        current_operation=current_operation,
        progress_percent=progress_percent,
        processed_items=processed_items,
        total_items=total_items,
    )


def end_task(result: dict[str, Any]) -> None:
    """标记任务成功结束"""
    global _state
    with _lock:
        _state = TaskState(
            task_id=_state.task_id,
            task_type=_state.task_type,
            title=_state.title,
            started_at=_state.started_at,
            finished_at=datetime.now().isoformat(),
            result=result,
            current_operation="已完成",
            progress_percent=100.0,
        )


def fail_task(error: str) -> None:
    """标记任务失败"""
    global _state
    with _lock:
        _state = TaskState(
            task_id=_state.task_id,
            task_type=_state.task_type,
            title=_state.title,
            started_at=_state.started_at,
            finished_at=datetime.now().isoformat(),
            error=error,
            current_operation="任务失败",
        )


def get_status() -> dict[str, Any] | None:
    """获取当前任务状态（标准化格式）"""
    global _state
    if _state.task_type is None:
        return None
    is_running = _state.finished_at is None
    result = {
        "task_id": _state.task_id,
        "task_type": _state.task_type,
        "title": _state.title or TASK_TITLES.get(_state.task_type, "正在处理..."),
        "started_at": _state.started_at,
        "is_running": is_running,
    }
    if _state.current_operation:
        result["current_operation"] = _state.current_operation
    if _state.progress_percent:
        result["progress_percent"] = _state.progress_percent
    if _state.total_items:
        result["total_items"] = _state.total_items
    if _state.processed_items:
        result["processed_items"] = _state.processed_items
    if _state.error:
        result["error"] = _state.error
    if _state.result:
        result["result"] = _state.result
    if _state.finished_at:
        result["finished_at"] = _state.finished_at
    return result


def get_last_result() -> dict[str, Any] | None:
    """获取上一次任务结果"""
    global _state
    if _state.finished_at and (_state.result is not None or _state.error is not None):
        return {
            "task_type": _state.task_type,
            "finished_at": _state.finished_at,
            "result": _state.result,
            "error": _state.error,
            "is_running": False,
        }
    return None


def clear() -> None:
    """清除任务状态"""
    global _state
    with _lock:
        _state = TaskState()
