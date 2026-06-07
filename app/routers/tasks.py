"""任务列表 API：获取活跃/历史任务、清理历史"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import get_async_session
from app.services.task_manager import task_manager

router = APIRouter(prefix="/api", tags=["tasks"])


def _task_to_dict(task):
    return {
        "id": task.id,
        "task_type": task.task_type,
        "title": task.title,
        "status": task.status,
        "progress_percent": task.progress_percent,
        "current_operation": task.current_operation,
        "total_items": task.total_items,
        "completed_items": task.completed_items,
        "error_message": task.error_message,
        "result_summary": task.result_summary,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


@router.get("/tasks")
async def get_tasks(session: AsyncSession = Depends(get_async_session)):
    """获取所有活跃任务 + 最近完成/失败的历史任务"""
    active = await task_manager.get_active_tasks(session)
    history = await task_manager.get_task_history(session, limit=20)
    active_count = await task_manager.count_active_tasks(session)
    return {
        "active": [_task_to_dict(t) for t in active],
        "history": [_task_to_dict(t) for t in history],
        "active_count": active_count,
    }


@router.get("/tasks/{task_id}")
async def get_task_detail(
    task_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """获取单个任务详情"""
    task = await task_manager.get_task(task_id, session)
    if task is None:
        return {"error": "任务不存在"}
    return _task_to_dict(task)


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """取消一个待执行或运行中的任务"""
    ok = await task_manager.cancel_task(task_id, session)
    return {"ok": ok}


@router.post("/tasks/cleanup")
async def cleanup_tasks(session: AsyncSession = Depends(get_async_session)):
    """清理所有已完成/失败/取消的历史任务"""
    count = await task_manager.cleanup_completed(session)
    return {"deleted": count}
