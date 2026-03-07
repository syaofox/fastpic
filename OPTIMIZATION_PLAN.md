# FastPic 性能优化方案

> 生成日期: 2026-03-07
> 目的: 记录项目可优化点及具体实现方案

---

## 1. 数据库性能优化

### 1.1 缺少复合索引

**问题分析:**
- `Image` 表的查询经常需要同时按 `relative_path` 和 `media_type` 过滤
- `ImageTag` 表的 `tag_id` 索引在按标签筛选时效率低
- `FolderThumbnail` 表的查询缺少索引

**优化方案:**
```sql
-- 在 Image 表添加复合索引
CREATE INDEX ix_images_path_media ON images(relative_path(255), media_type);
CREATE INDEX ix_images_filename_natural ON images(filename_natural(255));

-- 在 ImageTag 表添加索引
CREATE INDEX ix_image_tags_tag_id ON image_tags(tag_id);

-- 在 FolderThumbnail 表添加索引
CREATE INDEX ix_folder_thumbnails_folder ON folder_thumbnails(folder_path, display_order);
```

**实现方式:** 在 `models.py` 的 `_run_fulltext_migration()` 类似的迁移函数中添加

---

### 1.2 N+1 查询问题

**问题分析:**
- `main.py:342-356` 中先获取图片列表，再为每张图片单独查询标签
- 当图片数量多时会产生大量数据库往返

**当前代码:**
```python
# main.py:342-356
image_tags_map: dict[int, list[str]] = {}
if images_list:
    image_ids = [img.id for img in images_list if img.id]
    if image_ids:
        tag_stmt = (
            select(ImageTag.image_id, Tag.name)
            .join(Tag, Tag.id == ImageTag.tag_id)
            .where(ImageTag.image_id.in_(image_ids))
            .order_by(Tag.name)
        )
        tag_result = await session.execute(tag_stmt)
        for img_id, tag_name in tag_result.fetchall():
            if img_id not in image_tags_map:
                image_tags_map[img_id] = []
            image_tags_map[img_id].append(tag_name)
```

**优化方案:**
此代码已经是批量查询，但可以进一步优化：
1. 使用 `selectinload` 在查询图片时预加载标签
2. 或使用一个查询同时获取图片和标签（需要改写查询逻辑）

**优先级:** 中（当前实现已是批量查询，提升有限）

---

### 1.3 COUNT 查询优化

**问题分析:**
- 当有筛选条件时，`COUNT(*)` 需要扫描全表
- 对百万级图片库影响明显

**优化方案:**
1. 使用覆盖索引（Covering Index）
2. 对常用筛选条件添加复合索引
3. 考虑使用近似计数（如预先计算的统计表）

**具体实现:**
```sql
-- 为常用筛选条件添加覆盖索引
CREATE INDEX ix_images_mod_size ON images(modified_at, file_size);
CREATE INDEX ix_images_filename_media ON images(filename(255), media_type);
```

---

## 2. 代码重复/冗余

### 2.1 scanner.py 图片处理逻辑重复

**问题分析:**
- `_process_single_image_sync` (L241-278) 和 `get_media_metadata_and_thumbnail` (L80-122)
- 两者的图片加载、缩略图生成逻辑高度相似

**当前代码对比:**

`get_media_metadata_and_thumbnail` (L80-122):
```python
def get_media_metadata_and_thumbnail(full_path: Path, cache_path: Path, is_video: bool):
    # 1. 检查缓存是否新鲜
    # 2. 加载图片/视频
    # 3. 生成缩略图
    # 4. 返回元数据
```

`_process_single_image_sync` (L241-278):
```python
def _process_single_image_sync(full_path: Path, photos_dir: Path, cache_dir: Path):
    # 1. 获取相对路径
    # 2. 加载图片
    # 3. 生成缩略图（重复逻辑）
    # 4. 返回元数据
```

**优化方案:**
重构 `_process_single_image_sync` 使其复用 `get_media_metadata_and_thumbnail`:
```python
def _process_single_image_sync(full_path: Path, photos_dir: Path, cache_dir: Path):
    rel_path = _relative_path(photos_dir, full_path)
    cache_name = cache_filename(rel_path)
    cache_path = cache_dir / cache_name
    
    # 复用已有函数
    data = get_media_metadata_and_thumbnail(full_path, cache_path, is_video=False)
    if data is None:
        return None
    
    width, height, modified_at, file_size, is_corrupted = data
    return (full_path.name, rel_path, modified_at, file_size, width, height, is_corrupted)
```

**注意:** 需要将 `get_media_metadata_and_thumbnail` 改为同步函数或确保线程安全

---

### 2.2 重复的相对路径处理函数

**问题分析:**
- `_relative_path` 在 `scanner.py` (L31-34) 和 `watcher.py` (L30) 都有定义
- 应该提取到公共工具模块

**优化方案:**
将 `scanner.py` 中的 `_relative_path` 移到 `app/utils/path_utils.py`:
```python
# app/utils/path_utils.py 添加
def relative_path(photos_dir: Path, full_path: Path) -> str:
    """计算相对路径，统一使用 / 分隔"""
    rel = full_path.relative_to(photos_dir)
    return str(rel).replace("\\", "/")
```

然后在 `scanner.py` 和 `watcher.py` 中导入使用

---

## 3. 缓存策略优化

### 3.1 热门标签缓存

**问题分析:**
- `main.py:129-137` 首页每次都查询热门标签
- 标签数据变化不频繁，可以缓存

**优化方案:**
在 `app/utils/tags.py` 或新建缓存模块:
```python
# 添加热门标签缓存
_hot_tags_cache: list[dict] | None = None
_hot_tags_cache_ts: float = 0
HOT_TAGS_CACHE_TTL = 300  # 5分钟

def get_hot_tags_cached() -> list[dict]:
    global _hot_tags_cache, _hot_tags_cache_ts
    now = time.monotonic()
    if _hot_tags_cache and (now - _hot_tags_cache_ts) < HOT_TAGS_CACHE_TTL:
        return _hot_tags_cache
    # 重新查询...
    return _hot_tags_cache

def invalidate_hot_tags_cache():
    global _hot_tags_cache, _hot_tags_cache_ts
    _hot_tags_cache = None
    _hot_tags_cache_ts = 0
```

在标签增删改时调用 `invalidate_hot_tags_cache()`

---

### 3.2 分级缓存策略

**问题分析:**
- 当前 `_FOLDER_TREE_CACHE_TTL = 60s` 相对较短
- 频繁操作时缓存频繁失效

**优化方案:**
1. 增加缓存 TTL（如 5 分钟）
2. 实现写时失效（Write-Through）：操作后主动更新缓存而非等待过期

```python
# 优化后的缓存结构
_folder_tree_cache: dict | None = None
_folder_tree_cache_lock = asyncio.Lock()

# 写时失效：操作后主动更新
def invalidate_and_refresh_cache(photos_dir: Path, session):
    global _folder_tree_cache
    # 清除缓存
    _folder_tree_cache = None
    # 立即重新计算（可选，视负载而定）
```

---

## 4. 并发处理优化

### 4.1 watcher 事件并行处理

**问题分析:**
- `_drain_queue` (watcher.py:213-286) 逐个 await 处理事件
- 大量文件变化时效率低

**当前代码 (watcher.py:264-276):**
```python
for key, ev in path_events.items():
    event_type, src, dst, ts = ev
    try:
        if event_type == "created":
            await _process_created(photos_dir, cache_dir, Path(src))
        elif event_type == "deleted":
            await _process_deleted(photos_dir, cache_dir, Path(src))
        elif event_type == "moved":
            await _process_moved(photos_dir, cache_dir, Path(src), Path(dst))
        processed += 1
    except Exception as e:
        print(f"[watcher] 处理事件失败 ({event_type} {src}): {e}")
```

**优化方案:**
```python
# 分类事件
created_events = []
deleted_events = []
moved_events = []

for key, ev in path_events.items():
    event_type, src, dst, ts = ev
    if event_type == "created":
        created_events.append(Path(src))
    elif event_type == "deleted":
        deleted_events.append(Path(src))
    elif event_type == "moved":
        moved_events.append((Path(src), Path(dst)))

# 并行处理
tasks = []
for src in created_events:
    tasks.append(_process_created(photos_dir, cache_dir, src))
for src in deleted_events:
    tasks.append(_process_deleted(photos_dir, cache_dir, src))
for src, dst in moved_events:
    tasks.append(_process_moved(photos_dir, cache_dir, src, dst))

if tasks:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed = sum(1 for r in results if not isinstance(r, Exception))
```

**注意:** 需要确保数据库操作的并发安全，可能需要增加锁

---

### 4.2 缩略图生成并发度调优

**问题分析:**
- `_PROCESS_BATCH_SIZE = min(16, _MAX_WORKERS * 2)` 相对保守
- 多核机器未能充分利用

**优化方案:**
```python
# scanner.py
# 根据 CPU 核心数动态调整
_MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)
_PROCESS_BATCH_SIZE = min(32, _MAX_WORKERS * 2)  # 从 16 提升到 32
```

**注意:** 需监控 I/O 等待，避免过高的并发导致磁盘 I/O 瓶颈

---

## 5. 错误处理增强

### 5.1 上传错误处理细化

**问题分析:**
- `images.py:459-473` 统一作为 Exception 处理
- 无法区分不同错误类型

**优化方案:**
```python
# 细化错误类型
class UploadError(Exception):
    """上传基础错误"""
    pass

class FileSizeExceededError(UploadError):
    """文件大小超限"""
    pass

class DuplicateFileError(UploadError):
    """文件重复"""
    pass

# 在 _process_one 中抛出具体错误
if len(content) > MAX_UPLOAD_FILE_SIZE:
    raise FileSizeExceededError(f"单文件超过大小限制")

if content_hash in existing_hashes:
    raise DuplicateFileError(f"文件已存在")

# 在结果处理中区分
for r in results:
    if isinstance(r, FileSizeExceededError):
        errors.append(f"文件大小超限: {str(r)}")
    elif isinstance(r, DuplicateFileError):
        skipped += 1
    elif isinstance(r, Exception):
        errors.append(str(r))
```

---

### 5.2 scanner 错误隔离与恢复

**问题分析:**
- 当前某批处理失败可能导致整个扫描中断
- 缺少单个文件失败的隔离机制

**优化方案:**
```python
async def scan_photos(...):
    # 在 _process_batch 中添加错误隔离
    async def _process_batch_safe(paths: list[Path]) -> list[tuple]:
        results = []
        for batch in chunked(paths, 100):  # 每100个一批
            try:
                batch_results = await _process_batch(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.warning(f"批次处理失败，继续下一批: {e}")
                continue  # 跳过失败批次，不中断整体流程
        return results
```

---

## 6. 资源管理优化

### 6.1 连接池配置环境变量化

**问题分析:**
- `models.py:27-29` 硬编码连接池参数
- 不同环境需要不同配置

**优化方案:**
```python
# models.py
_db_pool_size = int(os.environ.get("DB_POOL_SIZE", "20"))
_db_max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "40"))
_db_pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))

# 在 .env.example 中添加
# DB_POOL_SIZE=20
# DB_MAX_OVERFLOW=40
# DB_POOL_RECYCLE=3600
```

---

### 6.2 流式处理大列表

**问题分析:**
- `folder_tree.py` 中 `_get_folder_tree_from_db_batched` 一次性加载所有路径
- 百万级图片时内存压力大

**优化方案:**
```python
async def _get_folder_tree_streaming(session, photos_dir: Path):
    """使用生成器流式处理，避免一次性加载"""
    folder_counts = {}
    
    # 分批查询，每批处理完立即yield
    batch_size = 10000
    last_id = 0
    
    while True:
        stmt = (
            select(Image.id, Image.relative_path)
            .where(Image.id > last_id)
            .order_by(Image.id)
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        
        if not rows:
            break
            
        for row in rows:
            # 处理单条记录
            process_path(row.relative_path, folder_counts)
            last_id = row.id
            
        # 每批处理完让出事件循环
        await asyncio.sleep(0)
    
    return folder_counts
```

---

## 7. 前端/HTMX 优化

### 7.1 API 合并

**问题分析:**
- gallery 页面需要分别请求图片和子文件夹
- 可以合并减少请求数

**优化方案:**
新建组合 API:
```python
@app.get("/api/gallery-data")
async def api_gallery_data(
    path: str = "",
    # ... 其他参数
    session: AsyncSession = Depends(get_async_session),
):
    """返回图片和子文件夹的组合数据"""
    # 复用现有逻辑
    images = await get_images(...)
    subfolders = await get_subfolders(...)
    
    return {
        "images": [serialize_image(img) for img in images],
        "subfolders": subfolders,
    }
```

---

## 8. 代码质量提升

### 8.1 添加返回类型注解

**需要添加类型注解的函数:**
- `app/utils/images.py` 的所有函数
- `app/utils/unique_path.py` 的所有函数
- 各个 router 中的部分函数

**示例:**
```python
# images.py
def cache_filename(relative_path: str) -> str:
    ...

def delete_image_files(relative_path: str, photos_dir: Path, cache_dir: Path) -> None:
    ...
```

---

### 8.2 配置参数提取

**问题分析:**
- 硬编码的魔数分散在各处
- 难以统一管理

**优化方案:**
在 `config.py` 中集中定义:
```python
# 图片浏览配置
FOLDER_IMAGES_MAX = 5000
DEFAULT_PER_PAGE = 24
DEFAULT_COLS = 4

# 缓存配置
FOLDER_TREE_CACHE_TTL = 60.0
SUBFOLDER_CACHE_TTL = 90.0

# 并发配置
UPLOAD_PARALLEL = 4
SCAN_PROCESS_BATCH_SIZE = 16

# watcher 配置
WATCHER_DEBOUNCE_SECONDS = 3.0
WATCHER_POLL_INTERVAL = 2.0
```

---

## 优化优先级建议

| 优先级 | 优化项 | 预期收益 | 工作量 |
|--------|--------|----------|--------|
| P0 | 添加数据库索引 | 高 | 低 |
| P0 | 修复代码重复 | 中 | 中 |
| P1 | watcher 并发优化 | 高 | 中 |
| P1 | 热门标签缓存 | 中 | 低 |
| P2 | 流式处理 | 中 | 高 |
| P2 | API 合并 | 中 | 中 |
| P3 | 类型注解 | 低 | 高 |
| P3 | 配置提取 | 低 | 中 |

---

## 执行检查清单

- [ ] 1.1 添加数据库索引
- [ ] 2.1 合并图片处理函数
- [ ] 2.2 提取公共函数
- [ ] 3.1 热门标签缓存
- [ ] 3.2 缓存策略优化
- [ ] 4.1 watcher 并发优化
- [ ] 4.2 调整并发参数
- [ ] 5.1 错误处理细化
- [ ] 6.1 环境变量配置化
- [ ] 6.2 流式处理实现
- [ ] 7.1 API 合并
- [ ] 8.1 类型注解
- [ ] 8.2 配置参数提取
