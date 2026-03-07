"""
任务状态服务：用于持久化设置页面费时操作的状态，
防止页面刷新后 UI 状态丢失导致重复提交。

支持任务队列，一次只能执行一个费时操作。
"""

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_FILE = Path("task_state.json")
_lock = threading.Lock()


@dataclass
class TaskState:
    task_type: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


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

    return result


def is_busy() -> bool:
    """检查是否有任务正在进行"""
    state = _read_state()
    return state.finished_at is None and state.task_type is not None


def start_task(task_type: str) -> bool:
    """标记任务开始，返回是否成功启动（如果忙则返回 False）"""
    if is_busy():
        return False

    state = TaskState(
        task_type=task_type,
        started_at=datetime.now().isoformat(),
    )
    _write_state(state)
    return True


def end_task(result: dict[str, Any]) -> None:
    """标记任务成功结束"""
    current = _read_state()
    state = TaskState(
        task_type=current.task_type,
        started_at=current.started_at,
        finished_at=datetime.now().isoformat(),
        result=result,
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
    )
    _write_state(state)


def get_status() -> dict[str, Any] | None:
    """获取当前任务状态"""
    state = _read_state()
    if state.task_type is None:
        return None
    is_running = state.finished_at is None
    return {
        "task_type": state.task_type,
        "started_at": state.started_at,
        "is_running": is_running,
    }


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
