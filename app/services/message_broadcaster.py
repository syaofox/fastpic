import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any


class MessageType(StrEnum):
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    GALLERY_UPDATE = "gallery_update"
    NOTIFICATION = "notification"
    SCAN_STATUS = "scan_status"


class MessageBroadcaster:
    def __init__(self):
        from app.routers.websocket import manager

        self._manager = manager

    def _create_message(self, msg_type: MessageType, payload: dict[str, Any]) -> dict:
        return {
            "type": msg_type.value,
            "payload": payload,
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        }

    async def broadcast_task_start(self, task_type: str, title: str, total_items: int = 0):
        message = self._create_message(
            MessageType.TASK_START, {"task_type": task_type, "title": title, "total_items": total_items}
        )
        await self._manager.broadcast(message)

    async def broadcast_task_progress(
        self, task_type: str, processed_items: int, total_items: int, current_operation: str = ""
    ):
        message = self._create_message(
            MessageType.TASK_PROGRESS,
            {
                "task_type": task_type,
                "processed_items": processed_items,
                "total_items": total_items,
                "progress_percent": int((processed_items / total_items) * 100) if total_items > 0 else 0,
                "current_operation": current_operation,
            },
        )
        await self._manager.broadcast(message)

    async def broadcast_task_complete(self, task_type: str, result: dict[str, Any], result_message: str = ""):
        message = self._create_message(
            MessageType.TASK_COMPLETE, {"task_type": task_type, "result": result, "result_message": result_message}
        )
        await self._manager.broadcast(message)

    async def broadcast_task_error(self, task_type: str, error: str):
        message = self._create_message(MessageType.TASK_ERROR, {"task_type": task_type, "error": error})
        await self._manager.broadcast(message)

    async def broadcast_gallery_update(self, affected_path: str = "", action: str = "update"):
        message = self._create_message(MessageType.GALLERY_UPDATE, {"affected_path": affected_path, "action": action})
        await self._manager.broadcast(message)

    async def broadcast_scan_status(self, scanning: bool):
        message = self._create_message(MessageType.SCAN_STATUS, {"scanning": scanning})
        await self._manager.broadcast(message)

    async def broadcast_notification(self, msg: str, level: str = "info"):
        broadcast_msg = self._create_message(MessageType.NOTIFICATION, {"message": msg, "level": level})
        await self._manager.broadcast(broadcast_msg)


broadcaster = MessageBroadcaster()
