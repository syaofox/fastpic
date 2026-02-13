"""
扫描状态：供 main、watcher 等模块共享，用于前端显示「正在扫描」提示。
"""

_scanning_count = 0


def begin_scan() -> None:
    """标记扫描开始"""
    global _scanning_count
    _scanning_count += 1


def end_scan() -> None:
    """标记扫描结束"""
    global _scanning_count
    _scanning_count -= 1


def is_scanning() -> bool:
    """当前是否有扫描任务在进行"""
    return _scanning_count > 0
