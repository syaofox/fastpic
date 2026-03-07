
HOT_TAGS_CACHE_TTL = 300.0
_hot_tags_cache: list[dict] | None = None
_hot_tags_cache_ts: float = 0


def invalidate_hot_tags_cache() -> None:
    global _hot_tags_cache, _hot_tags_cache_ts
    _hot_tags_cache = None
    _hot_tags_cache_ts = 0
