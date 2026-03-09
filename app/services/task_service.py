from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services import task_state
from app.services.message_broadcaster import broadcaster


@dataclass
class TaskContext:
    task_type: str
    title: str
    total_items: int = 0
    processed_items: int = 0

    async def broadcast_start(self):
        await broadcaster.broadcast_task_start(self.task_type, self.title, self.total_items)

    async def broadcast_progress(self, processed: int, operation: str = ""):
        self.processed_items = processed
        await broadcaster.broadcast_task_progress(self.task_type, processed, self.total_items, operation)

    async def broadcast_complete(self, result: dict[str, Any], message: str = ""):
        await broadcaster.broadcast_task_complete(self.task_type, result, message)

    async def broadcast_error(self, error: str):
        await broadcaster.broadcast_task_error(self.task_type, error)


class TaskService:
    def __init__(self):
        self._handlers: dict[str, Callable[[TaskContext, Any], Awaitable[dict]]] = {}

    def register(self, task_type: str):
        def decorator(func: Callable[[TaskContext, Any], Awaitable[dict]]):
            self._handlers[task_type] = func
            return func

        return decorator

    async def execute(self, task_type: str, title: str, params: Any = None, total_items: int = 0) -> dict[str, Any]:
        if not task_state.start_task(task_type, total_items, title):
            return {"error": "有任务正在进行中"}

        context = TaskContext(task_type=task_type, title=title, total_items=total_items)

        try:
            await context.broadcast_start()

            handler = self._handlers.get(task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task_type}")

            result = await handler(context, params)

            result_message = self._format_result_message(task_type, result)
            await context.broadcast_complete(result, result_message)
            task_state.end_task(result)

            return result

        except Exception as e:
            error_msg = str(e)
            await context.broadcast_error(error_msg)
            task_state.fail_task(error_msg)
            return {"error": error_msg}

    def _format_result_message(self, task_type: str, result: dict[str, Any]) -> str:
        messages = {
            "scan": f"扫描完成，发现 {result.get('scanned', 0)} 个文件",
            "cleanup": f"清理完成，移除 {result.get('stale_removed', 0)} 条记录",
            "full-sync": f"同步完成，发现 {result.get('images_added', 0) + result.get('videos_added', 0)} 个新文件",
            "upload": f"已上传 {result.get('uploaded', 0)} 个文件",
            "delete-images": f"已删除 {result.get('deleted', 0)} 项",
            "delete-folders": f"已删除 {result.get('deleted_folders', 0)} 个文件夹",
            "move-images": f"已移动 {result.get('moved', 0)} 项",
            "move-folders": f"已移动 {result.get('moved', 0)} 个文件夹",
        }
        return messages.get(task_type, "操作完成")


task_service = TaskService()
