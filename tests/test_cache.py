"""缓存相关测试。"""


from app.utils.cache_utils import (
    HOT_TAGS_CACHE_TTL,
    invalidate_hot_tags_cache,
)
from app.utils.folder_tree import _FOLDER_TREE_CACHE_TTL


class TestHotTagsCache:
    """热门标签缓存测试"""

    def test_cache_constants(self):
        """验证缓存 TTL 常量"""
        assert HOT_TAGS_CACHE_TTL == 300.0

    def test_folder_tree_cache_ttl(self):
        """验证文件夹树缓存 TTL 已更新为 5 分钟"""
        assert _FOLDER_TREE_CACHE_TTL == 300.0

    def test_invalidate_hot_tags_cache(self):
        """验证缓存失效函数"""
        from app.utils import cache_utils

        cache_utils._hot_tags_cache = [{"name": "test", "count": 1}]
        cache_utils._hot_tags_cache_ts = 999999

        invalidate_hot_tags_cache()

        assert cache_utils._hot_tags_cache is None
        assert cache_utils._hot_tags_cache_ts == 0

    def test_invalidate_multiple_times(self):
        """验证多次失效不会出错"""
        from app.utils import cache_utils

        cache_utils._hot_tags_cache = [{"name": "test", "count": 1}]
        cache_utils._hot_tags_cache_ts = 999999

        invalidate_hot_tags_cache()
        invalidate_hot_tags_cache()
        invalidate_hot_tags_cache()

        assert cache_utils._hot_tags_cache is None
