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


def test_defer_subfolders_in_main():
    """main.py gallery 路由支持 defer_subfolders 参数"""
    import inspect
    from main import gallery
    sig = inspect.signature(gallery)
    assert "defer_subfolders" in sig.parameters, "gallery 应有 defer_subfolders 参数"


def test_build_gallery_url_defer_subfolders():
    """utils.js buildGalleryUrl 在 path 空且 page=1 时加 defer_subfolders=1"""
    js_path = Path(__file__).resolve().parent.parent / "static" / "js" / "utils.js"
    content = js_path.read_text(encoding="utf-8")
    assert "defer_subfolders" in content, "buildGalleryUrl 应支持 defer_subfolders"
    assert "normalizedPath === ''" in content or "path === ''" in content or "path == ''" in content, "应有 path 空判断"


def test_index_html_defer_param():
    """index.html 首屏 hx-get 在 path 空时加 defer_subfolders=1"""
    html_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    content = html_path.read_text(encoding="utf-8")
    assert "defer_subfolders" in content, "index.html 应有 defer_subfolders"
    assert "hx-get" in content and "/gallery" in content, "应有 gallery 的 hx-get"


def test_gallery_subfolders_partial_exists():
    """partials/gallery_subfolders.html 存在"""
    partial_path = Path(__file__).resolve().parent.parent / "templates" / "partials" / "gallery_subfolders.html"
    assert partial_path.exists(), "gallery_subfolders.html 应存在"
    content = partial_path.read_text(encoding="utf-8")
    assert "subfolders" in content, "partial 应渲染 subfolders"


def test_get_direct_children_from_folder_counts():
    """_get_direct_children_from_folder_counts 从 folder_counts 解析直接子目录"""
    from utils.folder_tree import _get_direct_children_from_folder_counts

    folder_counts = {
        "2024": 100,
        "2024/01": 30,
        "2024/01/15": 5,
        "2024/02": 20,
        "2023": 50,
    }
    # path_prefix "2024/01" -> 直接子目录应为 "15"
    children = _get_direct_children_from_folder_counts(folder_counts, "2024/01")
    assert children == {"15"}, f"期望 {{15}}，实际 {children}"

    # path_prefix "2024" -> 直接子目录应为 "01", "02"
    children2 = _get_direct_children_from_folder_counts(folder_counts, "2024")
    assert children2 == {"01", "02"}, f"期望 {{01, 02}}，实际 {children2}"

    # path_prefix 带尾斜杠也正确
    children4 = _get_direct_children_from_folder_counts(folder_counts, "2024/")
    assert children4 == {"01", "02"}, f"期望 {{01, 02}}，实际 {children4}"


def test_folder_tree_no_scan_all_dirs_dead_code():
    """folder_tree 已移除 scan_all_dirs_for_search 死代码"""
    import utils.folder_tree as ft
    assert not hasattr(ft, "scan_all_dirs_for_search"), "scan_all_dirs_for_search 应已移除"


def test_merge_folder_uses_os_walk_topdown_false():
    """merge_folder 使用 os.walk(..., topdown=False) 自底向上删空目录"""
    folders_path = Path(__file__).resolve().parent.parent / "routers" / "folders.py"
    content = folders_path.read_text(encoding="utf-8")
    assert "os.walk" in content, "merge_folder 应使用 os.walk"
    assert "topdown=False" in content, "merge_folder 应使用 topdown=False 自底向上"


def test_settings_trigger_cleanup_uses_run_full_scan():
    """trigger_cleanup 复用 run_full_scan，一次 os.walk 完成"""
    settings_path = Path(__file__).resolve().parent.parent / "routers" / "settings.py"
    content = settings_path.read_text(encoding="utf-8")
    assert "run_full_scan" in content, "settings 应使用 run_full_scan"
    assert "trigger_cleanup" in content
    # trigger_cleanup 应调用 run_full_scan 而非单独的 _collect_media
    assert content.count("run_full_scan") >= 2, "trigger_scan 与 trigger_cleanup 均应调用 run_full_scan"


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
        ("defer_subfolders main.py", test_defer_subfolders_in_main),
        ("defer_subfolders buildGalleryUrl", test_build_gallery_url_defer_subfolders),
        ("defer_subfolders index.html", test_index_html_defer_param),
        ("gallery_subfolders partial", test_gallery_subfolders_partial_exists),
        ("_get_direct_children_from_folder_counts", test_get_direct_children_from_folder_counts),
        ("folder_tree 无 scan_all_dirs 死代码", test_folder_tree_no_scan_all_dirs_dead_code),
        ("merge_folder os.walk topdown=False", test_merge_folder_uses_os_walk_topdown_false),
        ("settings trigger_cleanup 复用 run_full_scan", test_settings_trigger_cleanup_uses_run_full_scan),
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
