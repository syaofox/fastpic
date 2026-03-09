"""
队列 API 路由：任务队列管理接口
"""

from pydantic import BaseModel

from app.services.task_queue import TaskQueue


class AddTaskRequest(BaseModel):
    task_type: str
    params: dict | None = None
    priority: int = 5


class CancelTaskRequest(BaseModel):
    queue_id: str


async def add_to_queue(task_type: str, params: dict | None = None, priority: int = 5) -> str:
    """添加任务到队列"""
    queue = TaskQueue()
    return await queue.add_task(task_type, params, priority)


async def get_queue_status() -> dict:
    """获取队列状态"""
    queue = TaskQueue()
    return queue.get_status()


async def get_task_status(queue_id: str) -> dict | None:
    """获取指定任务状态"""
    queue = TaskQueue()
    return queue.get_task_status(queue_id)


async def cancel_task(queue_id: str) -> bool:
    """取消任务"""
    queue = TaskQueue()
    return queue.cancel_task(queue_id)
