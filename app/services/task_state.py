"""
任务状态服务：用于持久化设置页面费时操作的状态，
防止页面刷新后 UI 状态丢失导致重复提交。

支持任务队列，一次只能执行一个费时操作。
支持 SSE 实时推送进度。
"""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

STATE_FILE = Path("task_state.json")
_lock = threading.Lock()

_progress_callbacks: list[Callable[[dict], None]] = []
_queue: asyncio.Queue | None = None


def _get_queue() -> asyncio.Queue:
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
    q = _get_queue()
    if not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await q.put(data)


def get_queue_for_sse() -> asyncio.Queue:
    """获取 SSE 队列"""
    return _get_queue()


@dataclass
class TaskState:
    task_type: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    current_operation: str | None = None
    progress_percent: float = 0.0
    total_items: int = 0
    processed_items: int = 0


def _ensure_data_dir() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_state() -> TaskState:
    _ensure_data_dir()
    if not STATE_FILE.exists():
        return TaskState()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return TaskState(
                task_type=data.get("task_type"),
                started_at=data.get("started_at"),
                finished_at=data.get("finished_at"),
                result=data.get("result"),
                error=data.get("error"),
                current_operation=data.get("current_operation"),
                progress_percent=data.get("progress_percent", 0.0),
                total_items=data.get("total_items", 0),
                processed_items=data.get("processed_items", 0),
            )
    except (json.JSONDecodeError, IOError):
        return TaskState()


def _write_state(state: TaskState) -> None:
    _ensure_data_dir()
    data = {
        "task_type": state.task_type,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "result": state.result,
        "error": state.error,
        "current_operation": state.current_operation,
        "progress_percent": state.progress_percent,
        "total_items": state.total_items,
        "processed_items": state.processed_items,
    }
    with _lock:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def get_queue_status() -> dict[str, Any]:
    """获取任务队列状态"""
    state = _read_state()
    is_running = state.finished_at is None and state.task_type is not None

    result = {
        "is_running": is_running,
        "current_task": state.task_type,
    }

    if state.task_type and state.started_at:
        result["started_at"] = state.started_at

    if is_running:
        result["status"] = "running"
    else:
        result["status"] = "idle"

    if state.current_operation:
        result["current_operation"] = state.current_operation
    if state.progress_percent:
        result["progress_percent"] = state.progress_percent
    if state.total_items:
        result["total_items"] = state.total_items
    if state.processed_items:
        result["processed_items"] = state.processed_items

    return result


def is_busy() -> bool:
    """检查是否有任务正在进行"""
    state = _read_state()
    return state.finished_at is None and state.task_type is not None


def start_task(task_type: str, total_items: int = 0) -> bool:
    """标记任务开始，返回是否成功启动（如果忙则返回 False）"""
    if is_busy():
        return False

    state = TaskState(
        task_type=task_type,
        started_at=datetime.now().isoformat(),
        total_items=total_items,
    )
    _write_state(state)
    return True


def update_progress(
    current_operation: str | None = None,
    progress_percent: float | None = None,
    processed_items: int | None = None,
    total_items: int | None = None,
) -> None:
    """更新任务进度"""
    current = _read_state()
    state = TaskState(
        task_type=current.task_type,
        started_at=current.started_at,
        finished_at=current.finished_at,
        result=current.result,
        error=current.error,
        current_operation=current_operation if current_operation is not None else current.current_operation,
        progress_percent=progress_percent if progress_percent is not None else current.progress_percent,
        total_items=total_items if total_items is not None else current.total_items,
        processed_items=processed_items if processed_items is not None else current.processed_items,
    )
    _write_state(state)


async def async_update_progress(
    current_operation: str | None = None,
    progress_percent: float | None = None,
    processed_items: int | None = None,
    total_items: int | None = None,
) -> None:
    """异步更新任务进度（同步文件 + SSE 推送）"""
    update_progress(current_operation, progress_percent, processed_items, total_items)
    await emit_progress(
        current_operation=current_operation,
        progress_percent=progress_percent,
        processed_items=processed_items,
        total_items=total_items,
    )


def end_task(result: dict[str, Any]) -> None:
    """标记任务成功结束"""
    current = _read_state()
    state = TaskState(
        task_type=current.task_type,
        started_at=current.started_at,
        finished_at=datetime.now().isoformat(),
        result=result,
        current_operation="已完成",
        progress_percent=100.0,
    )
    _write_state(state)


def fail_task(error: str) -> None:
    """标记任务失败"""
    current = _read_state()
    state = TaskState(
        task_type=current.task_type,
        started_at=current.started_at,
        finished_at=datetime.now().isoformat(),
        error=error,
        current_operation="任务失败",
    )
    _write_state(state)


def get_status() -> dict[str, Any] | None:
    """获取当前任务状态"""
    state = _read_state()
    if state.task_type is None:
        return None
    is_running = state.finished_at is None
    result = {
        "task_type": state.task_type,
        "started_at": state.started_at,
        "is_running": is_running,
    }
    if state.current_operation:
        result["current_operation"] = state.current_operation
    if state.progress_percent:
        result["progress_percent"] = state.progress_percent
    if state.total_items:
        result["total_items"] = state.total_items
    if state.processed_items:
        result["processed_items"] = state.processed_items
    return result


def get_last_result() -> dict[str, Any] | None:
    """获取上一次任务结果"""
    state = _read_state()
    if state.finished_at and (state.result is not None or state.error is not None):
        return {
            "task_type": state.task_type,
            "finished_at": state.finished_at,
            "result": state.result,
            "error": state.error,
            "is_running": False,
        }
    return None


def clear() -> None:
    """清除任务状态"""
    _write_state(TaskState())
