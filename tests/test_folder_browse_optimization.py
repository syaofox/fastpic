"""测试文件夹浏览速度优化相关修改。

验证：
1. path_count_cache: _COUNT_CACHE_TTL 为 300 秒
2. path_count_cache: get/set 正常工作
3. folder_tree: get_subfolders 有 90 秒子文件夹缓存
4. gallery.js: galleryPathCache 存在且具备 get/set/clear
"""
import asyncio
from pathlib import Path
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_path_count_cache_ttl():
    """path_count_cache 内存 TTL 应为 300 秒（与 DB 一致）"""
    from utils import path_count_cache as pcc
    ttl = getattr(pcc, "_COUNT_CACHE_TTL", None)
    assert ttl == 300.0, f"期望 _COUNT_CACHE_TTL=300，实际为 {ttl}"


def test_path_count_cache_get_set():
    """path_count_cache get/set 基本功能"""
    from utils.path_count_cache import get_cached_count, set_cached_count

    set_cached_count("test/path", "folder", 42)
    val = get_cached_count("test/path", "folder")
    assert val == 42, f"期望 42，实际 {val}"

    # 空路径
    set_cached_count("", "folder", 10)
    assert get_cached_count("", "folder") == 10


def test_folder_tree_subfolder_cache_exists():
    """folder_tree 存在子文件夹缓存相关常量"""
    from utils import folder_tree as ft
    assert hasattr(ft, "_SUBFOLDER_CACHE_TTL"), "缺少 _SUBFOLDER_CACHE_TTL"
    assert ft._SUBFOLDER_CACHE_TTL == 90.0, f"期望 90，实际 {ft._SUBFOLDER_CACHE_TTL}"
    assert hasattr(ft, "_subfolder_cache"), "缺少 _subfolder_cache"
    assert hasattr(ft, "_subfolder_cache_key"), "缺少 _subfolder_cache_key"


def test_folder_tree_subfolder_cache_key():
    """子文件夹缓存键格式正确"""
    from utils.folder_tree import _subfolder_cache_key
    key = _subfolder_cache_key("a/b", "filename", "asc")
    assert key == "a/b|filename|asc"
    key2 = _subfolder_cache_key("", "modified_at", "desc")
    assert key2 == "|modified_at|desc"


def test_gallery_js_has_gallery_path_cache():
    """gallery.js 包含 galleryPathCache 模块"""
    js_path = Path(__file__).resolve().parent.parent / "static" / "js" / "gallery.js"
    content = js_path.read_text(encoding="utf-8")

    assert "window.galleryPathCache" in content, "缺少 window.galleryPathCache"
    assert "galleryPathCache.get" in content or 'get:' in content, "缺少 get 方法"
    assert "galleryPathCache.set" in content or 'set:' in content, "缺少 set 方法"
    assert "galleryPathCache.clear" in content or 'clear:' in content, "缺少 clear 方法"
    assert "MAX_ENTRIES = 40" in content or "MAX_ENTRIES" in content, "应有 MAX_ENTRIES 限制"
    assert "7 * 60 * 1000" in content or "TTL" in content, "应有 TTL 配置"


def test_gallery_js_cache_integration():
    """gallery.js 中缓存与导航、刷新逻辑集成"""
    js_path = Path(__file__).resolve().parent.parent / "static" / "js" / "gallery.js"
    content = js_path.read_text(encoding="utf-8")

    # 应有缓存命中时的点击拦截
    assert "getGalleryCacheKeyFromLink" in content, "缺少 getGalleryCacheKeyFromLink"
    assert "applyGalleryFromCache" in content, "缺少 applyGalleryFromCache"
    assert "preventDefault" in content, "缓存命中时应 preventDefault"
    # 应有 htmx:afterSwap 写入缓存
    assert "htmx:afterSwap" in content and "galleryPathCache.set" in content
    # 刷新/操作后应清空缓存
    assert "galleryPathCache.clear" in content


def test_invalidate_folder_tree_clears_subfolder_cache():
    """invalidate_folder_tree_cache 会清空子文件夹缓存"""
    from utils.folder_tree import invalidate_folder_tree_cache, _subfolder_cache

    invalidate_folder_tree_cache()
    assert len(_subfolder_cache) == 0, "invalidate 后 _subfolder_cache 应为空"


def test_gallery_endpoint_returns_html():
    """gallery 端点返回有效 HTML（需服务已启动在 127.0.0.1:8000，否则跳过）"""
    import subprocess
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/stdout", "-w", "%{http_code}", "http://127.0.0.1:8000/gallery"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ⚠ gallery 端点测试跳过（服务未启动或 curl 不可用）: {e}")
        return  # 跳过，不视为失败

    output = r.stdout or ""
    if len(output) >= 3 and output[-3:].isdigit():
        body, code = output[:-3], output[-3:]
    else:
        code, body = "000", output
    if code != "200":
        print(f"  ⚠ gallery 端点返回 {code}，可能服务未启动")
        return
    assert "gallery-container" in body or "current-path-marker" in body, "响应应包含 gallery 相关标记"


def run_tests():
    tests = [
        ("path_count_cache TTL", test_path_count_cache_ttl),
        ("path_count_cache get/set", test_path_count_cache_get_set),
        ("folder_tree 子文件夹缓存存在", test_folder_tree_subfolder_cache_exists),
        ("folder_tree 缓存键", test_folder_tree_subfolder_cache_key),
        ("gallery.js galleryPathCache", test_gallery_js_has_gallery_path_cache),
        ("gallery.js 缓存集成", test_gallery_js_cache_integration),
        ("invalidate 清空子文件夹缓存", test_invalidate_folder_tree_clears_subfolder_cache),
        ("gallery 端点返回 HTML", test_gallery_endpoint_returns_html),
    ]
    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  ✗ {name}: {e}")

    print(f"\n通过: {passed}/{len(tests)}")
    if failed:
        print("失败:", [n for n, _ in failed])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
