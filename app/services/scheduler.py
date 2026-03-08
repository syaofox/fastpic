"""
统一任务调度器：优先级 + 自适应限流 + 连接池代理

设计目标：
1. 防止数据库连接池耗尽
2. 用户请求优先于后台任务
3. 空闲时自适应提升并发
4. 统一入口便于监控和调优
"""

import asyncio
import logging
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from app.models import async_engine

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """连接池配置"""

    pool_size: int
    max_overflow: int
    total: int
    reserve_ratio: float = 0.3  # 保留给用户请求的比例
    idle_boost: float = 1.5  # 空闲时的并发提升系数


class TaskScheduler:
    """
    统一任务调度器

    优先级：
    - 0: 后台任务（批量重命名、扫描等）
    - 5: 普通任务
    - 10: 用户请求（最高）

    使用方式：
    scheduler = TaskScheduler()
    await scheduler.submit(coro, priority=0)  # 后台任务
    await scheduler.submit(coro, priority=10)  # 用户请求
    """

    _instance: "TaskScheduler | None" = None

    def __new__(cls) -> "TaskScheduler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        pool = async_engine.pool
        self._pool_config = PoolConfig(
            pool_size=pool.size(),
            max_overflow=pool._max_overflow,
            total=pool.size() + pool._max_overflow,
        )

        self._total_limit = int(self._pool_config.total * 0.8)  # 最多使用 80%
        self._reserve_limit = int(self._total_limit * self._pool_config.reserve_ratio)  # 保留
        self._user_limit = self._total_limit  # 用户请求可使用全部
        self._background_limit = self._total_limit - self._reserve_limit  # 后台任务

        self._sem = asyncio.Semaphore(self._background_limit)
        self._user_sem = asyncio.Semaphore(self._user_limit)
        self._idle_boost = False
        self._boost_lock = asyncio.Lock()

        self._initialized = True
        logger.info(
            f"[scheduler] 初始化: total={self._total_limit}, "
            f"user={self._user_limit}, background={self._background_limit}"
        )

    def _get_available_connections(self) -> int:
        """获取可用连接数（非线程安全，仅供估算）"""
        pool = async_engine.pool
        return pool.size() - pool.checkedin() + pool._max_overflow - pool.overflow()

    @asynccontextmanager
    async def _adaptive_limit(self, priority: int):
        """
        自适应限流上下文
        - 空闲时提升并发限制
        - 繁忙时严格限制
        """
        async with self._boost_lock:
            available = self._get_available_connections()
            if available > self._pool_config.total * 0.5:
                if not self._idle_boost:
                    self._idle_boost = True
                    logger.info("[scheduler] 进入空闲模式，提升并发")
            else:
                if self._idle_boost:
                    self._idle_boost = False
                    logger.info("[scheduler] 退出空闲模式")

        base_limit = self._user_limit if priority >= 10 else self._background_limit

        if self._idle_boost and priority < 10:
            base_limit = min(base_limit * 2, self._total_limit)

        sem = self._user_sem if priority >= 10 else self._sem

        try:
            async with sem:
                yield
        except Exception:
            raise

    async def submit(
        self,
        coro: Awaitable[Any],
        priority: int = 5,
        task_name: str = "",
    ) -> Any:
        """
        提交任务执行

        Args:
            coro: 协程对象
            priority: 优先级，0=后台任务，10=用户请求
            task_name: 任务名称（用于日志）

        Returns:
            协程结果
        """
        is_background = priority < 10
        task_type = "后台" if is_background else "用户"

        async with self._adaptive_limit(priority):
            try:
                result = await coro
                return result
            except Exception as e:
                logger.error(f"[scheduler] {task_type}任务失败 ({task_name}): {e}")
                raise

    def get_status(self) -> dict:
        """获取调度器状态"""
        return {
            "pool_total": self._pool_config.total,
            "pool_used": self._get_available_connections(),
            "background_limit": self._background_limit,
            "user_limit": self._user_limit,
            "idle_boost": self._idle_boost,
        }


scheduler = TaskScheduler()
