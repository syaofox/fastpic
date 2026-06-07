"""
TaskManager：持久化任务管理服务。

所有耗时操作（上传、移动、重命名、删除、扫描等）通过 TaskManager 创建
和更新任务记录，写入 tasks 表 + WebSocket 实时推送。

设计要点：
- 数据库持久化，页面刷新后进度不丢失
- 支持多个并发任务（不同 task_type）
- 通过 WebSocket 广播增量更新
- 兼容旧的 task_state.py（短期共存）
"""

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from app.models import Task
from app.services.message_broadcaster import broadcaster


class TaskManager:
    def _now(self) -> float:
        return datetime.now().timestamp()

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    async def create_task(
        self,
        session: AsyncSession,
        task_type: str,
        title: str = "",
        total_items: int = 0,
    ) -> Task:
        tid = self._new_id()
        now = self._now()
        task = Task(
            id=tid,
            task_type=task_type,
            title=title or task_type,
            status="pending",
            total_items=total_items,
            created_at=now,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)

        await broadcaster.broadcast_task_start(
            task_id=tid, task_type=task_type, title=task.title, total_items=total_items
        )
        return task

    async def start_task(self, task_id: str, session: AsyncSession) -> Task | None:
        task = await session.get(Task, task_id)
        if task is None:
            return None
        now = self._now()
        task.status = "running"
        task.started_at = now
        session.add(task)
        await session.commit()
        await session.refresh(task)

        await broadcaster.broadcast_task_progress(
            task_id=task_id,
            task_type=task.task_type,
            processed_items=task.completed_items,
            total_items=task.total_items,
            current_operation=task.current_operation,
        )
        return task

    async def update_progress(
        self,
        task_id: str,
        session: AsyncSession,
        processed_items: int | None = None,
        total_items: int | None = None,
        current_operation: str | None = None,
    ) -> Task | None:
        task = await session.get(Task, task_id)
        if task is None:
            return None

        if processed_items is not None:
            task.completed_items = processed_items
        if total_items is not None:
            task.total_items = total_items
        if current_operation is not None:
            task.current_operation = current_operation

        total = task.total_items or 1
        task.progress_percent = round((task.completed_items / total) * 100, 1)

        session.add(task)
        await session.commit()
        await session.refresh(task)

        await broadcaster.broadcast_task_progress(
            task_id=task_id,
            task_type=task.task_type,
            processed_items=task.completed_items,
            total_items=task.total_items,
            current_operation=task.current_operation,
        )
        return task

    async def complete_task(
        self,
        task_id: str,
        session: AsyncSession,
        result_summary: str = "",
    ) -> Task | None:
        task = await session.get(Task, task_id)
        if task is None:
            return None

        now = self._now()
        task.status = "completed"
        task.finished_at = now
        task.progress_percent = 100.0
        task.current_operation = "已完成"
        if result_summary:
            task.result_summary = result_summary
        session.add(task)
        await session.commit()
        await session.refresh(task)

        await broadcaster.broadcast_task_complete(
            task_id=task_id,
            task_type=task.task_type,
            result={"completed_items": task.completed_items},
            result_message=result_summary,
        )
        return task

    async def fail_task(
        self,
        task_id: str,
        session: AsyncSession,
        error_message: str = "",
    ) -> Task | None:
        task = await session.get(Task, task_id)
        if task is None:
            return None

        now = self._now()
        task.status = "failed"
        task.finished_at = now
        task.error_message = error_message
        task.current_operation = "任务失败"
        session.add(task)
        await session.commit()
        await session.refresh(task)

        await broadcaster.broadcast_task_error(
            task_id=task_id,
            task_type=task.task_type,
            error=error_message,
        )
        return task

    async def cancel_task(self, task_id: str, session: AsyncSession) -> bool:
        task = await session.get(Task, task_id)
        if task is None or task.status in ("completed", "failed", "cancelled"):
            return False

        now = self._now()
        task.status = "cancelled"
        task.finished_at = now
        task.error_message = "任务被取消"
        task.current_operation = "已取消"
        session.add(task)
        await session.commit()

        await broadcaster.broadcast_task_complete(
            task_id=task_id,
            task_type=task.task_type,
            result={},
            result_message="任务已取消",
        )
        return True

    async def get_active_tasks(self, session: AsyncSession) -> list[Task]:
        stmt = select(Task).where(Task.status.in_(["pending", "running"])).order_by(Task.created_at)  # pyright: ignore
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_task_history(self, session: AsyncSession, limit: int = 50) -> list[Task]:
        stmt = (
            select(Task)
            .where(Task.status.in_(["completed", "failed", "cancelled"]))  # pyright: ignore
            .order_by(Task.finished_at.desc())  # pyright: ignore
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_task(self, task_id: str, session: AsyncSession) -> Task | None:
        return await session.get(Task, task_id)

    async def count_active_tasks(self, session: AsyncSession) -> int:
        stmt = select(func.count(Task.id)).where(Task.status.in_(["pending", "running"]))  # pyright: ignore
        result = await session.execute(stmt)
        return result.scalar() or 0

    async def cleanup_completed(self, session: AsyncSession, before: float | None = None) -> int:
        stmt = select(Task).where(Task.status.in_(["completed", "failed", "cancelled"]))  # pyright: ignore
        if before is not None:
            stmt = stmt.where(Task.finished_at < before)  # pyright: ignore
        result = await session.execute(stmt)
        tasks = list(result.scalars().all())
        count = len(tasks)
        for t in tasks:
            await session.delete(t)
        await session.commit()
        return count


task_manager = TaskManager()
