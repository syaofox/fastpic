# FastPic 性能优化指南

本文档记录项目的性能瓶颈及修复方案，用于指导后续改进。

## 目录

- [严重问题](#严重问题)
- [中等问题](#中等问题)
- [轻微问题](#轻微问题)
- [待办事项清单](#待办事项清单)

---

## 严重问题

### 1. 上传时 MD5 哈希计算耗时过长

**文件**: `app/routers/images.py`  
**位置**: 第 65-122 行，第 336-340 行

#### 问题描述

上传图片时，需要计算图库中已有文件的 MD5 哈希来检测重复文件。在大型图库（数万张图片）中，这会导致长时间等待，因为需要读取所有现有文件内容计算哈希。

```python
# 当前实现：读取所有现有文件计算哈希
existing_hashes = await asyncio.to_thread(
    _compute_existing_hashes_for_subdirs, target_dir, subdirs, media_extensions
)
```

#### 影响

- 上传开始前有 30 秒到数分钟的等待时间
- 在网络存储或机械硬盘上尤为明显
- 用户体验差

#### 修复方案

**方案 A：使用快速去重策略**

1. 首先按文件大小分组，相同大小的文件才计算哈希
2. 对大小相同的文件，再比较修改时间
3. 仅当前两者都匹配时才计算完整 MD5

```python
# app/utils/hash_utils.py 新增

def compute_quick_hash(path: Path) -> str | None:
    """快速哈希：文件大小 + 修改时间 + 首尾各 1KB 内容"""
    if not path.is_file():
        return None
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = int(stat.st_mtime)
        
        h = hashlib.md5()
        h.update(str(size).encode())
        h.update(str(mtime).encode())
        
        # 取首尾各 1KB
        with open(path, "rb") as f:
            h.update(f.read(1024))
            f.seek(-1024, 2)
            h.update(f.read(1024))
        
        return h.hexdigest()
    except OSError:
        return None
```

**方案 B：使用数据库预存哈希**

1. 扫描时将 MD5 哈希存入数据库
2. 上传时只需查询数据库，无需读取文件

```python
# app/models.py Image 表新增字段
md5_hash: str | None = Field(default=None, sa_column=Column(String(32), index=True))
```

#### 实现步骤

1. 创建 `app/utils/quick_hash.py` 实现快速哈希
2. 修改 `_compute_existing_hashes` 使用快速哈希预筛
3. 仅对快速哈希冲突的文件计算完整 MD5
4. 可选：在扫描时计算并存储 MD5

---

## 中等问题

### 2. 搜索在应用层而非数据库层

**文件**: `app/utils/search.py`、`app/routers/folders.py`  
**位置**: 第 48-76 行，第 771-788 行

#### 问题描述

`search_match` 函数在 Python 中进行简繁转换、拼音匹配，然后在内存中对所有 folder_counts 进行遍历匹配。这对大量数据非常慢。

```python
# 当前：获取所有数据后在 Python 中匹配
full_dir_counts = dict(await get_folder_counts_for_search(session))
for dir_path, count in sorted(full_dir_counts.items()):
    if search_match(q, dir_path):  # Python 遍历匹配
        matched.append({"path": dir_path, "image_count": count})
```

#### 修复方案

**分层过滤策略**

1. 第一层：使用 SQL LIKE 预过滤，大幅减少候选集
2. 第二层：对小量候选进行简繁/拼音匹配

```python
# app/routers/folders.py 修改

@app.get("/search-dirs")
async def search_dirs(
    q: str = "",
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    q = (q or "").strip()
    if not q:
        return {"dirs": []}
    
    # 第一层：使用 SQL 预过滤（利用索引）
    escaped = escape_like(q)
    prefix_filter = f"{escaped}%"
    
    sql = text("""
        SELECT SUBSTRING_INDEX(relative_path, '/', 1) AS prefix, COUNT(*) AS cnt 
        FROM images 
        WHERE relative_path LIKE :prefix ESCAPE '!' 
        GROUP BY prefix
    """)
    
    result = await session.execute(sql, {"prefix": prefix_filter})
    rows = result.fetchall()
    
    # 第二层：对小量候选进行精确匹配
    matched = []
    for row in rows:
        dir_path, count = row[0], row[1]
        if search_match(q, dir_path):
            matched.append({"path": dir_path, "image_count": count})
            if len(matched) >= limit:
                break
    
    return {"dirs": matched}
```

#### 实现步骤

1. 修改 `search_dirs` API 使用 SQL LIKE 预过滤
2. 评估是否需要为简繁/拼音搜索添加预计算列

---

### 3. 批量删除效率低

**文件**: `app/routers/images.py`  
**位置**: 第 135-163 行

#### 问题描述

删除图片时，逐条处理文件删除和数据库删除，效率不高。

```python
# 当前实现
for img in images:
    delete_image_files(img.relative_path, PHOTOS_DIR, CACHE_DIR)
    await session.delete(img)
    deleted += 1
await session.commit()
```

#### 修复方案

```python
# 优化后：先收集路径，批量删除文件，最后删除数据库记录
@app.post("/delete-images")
async def delete_images(
    body: DeleteImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    if not task_state.start_task("delete-images"):
        return {"deleted": 0, "error": "有任务正在进行中"}
    
    try:
        if not body.ids:
            return {"deleted": 0}
        
        # 1. 批量获取所有图片记录
        all_images = []
        for i in range(0, len(body.ids), IN_CLAUSE_BATCH_SIZE):
            batch_ids = body.ids[i : i + IN_CLAUSE_BATCH_SIZE]
            stmt = select(Image).where(Image.id.in_(batch_ids))
            result = await session.execute(stmt)
            all_images.extend(result.scalars().all())
        
        # 2. 收集所有文件路径
        photo_paths = []
        cache_paths = []
        for img in all_images:
            photo_paths.append(PHOTOS_DIR / img.relative_path)
            cache_name = cache_filename(img.relative_path)
            cache_paths.append(CACHE_DIR / cache_name)
        
        # 3. 批量删除文件（使用线程池）
        def _delete_files(paths: list[Path]):
            for p in paths:
                if p.exists():
                    p.unlink(missing_ok=True)
        
        await asyncio.gather(
            asyncio.to_thread(_delete_files, photo_paths),
            asyncio.to_thread(_delete_files, cache_paths),
        )
        
        # 4. 批量删除数据库记录
        for img in all_images:
            await session.delete(img)
        await session.commit()
        
        if len(all_images) > 0:
            invalidate_folder_tree_cache()
        
        task_state.end_task({"deleted": len(all_images)})
        return {"deleted": len(all_images)}
```

#### 实现步骤

1. 创建优化后的 `delete_images` 函数
2. 使用 `asyncio.gather` 并行删除原图和缓存
3. 确保事务一致性

---

### 4. 缓存失效粒度过粗

**文件**: `app/utils/folder_tree.py`  
**位置**: 第 232-242 行

#### 问题描述

任何文件夹操作都会清空所有缓存，导致频繁操作时缓存反复失效。

```python
def invalidate_folder_tree_cache() -> None:
    global _folder_tree_cache, _subfolder_cache
    _folder_tree_cache = None
    _subfolder_cache = {}  # 清空所有子文件夹缓存
```

#### 修复方案

**细粒度缓存失效**

```python
# 修改 invalidate_folder_tree_cache

def invalidate_folder_tree_cache(affected_path: str | None = None) -> None:
    """affected_path 为受影响的路径，仅清除相关缓存"""
    global _folder_tree_cache, _subfolder_cache
    
    if affected_path is None:
        # 全量失效
        _folder_tree_cache = None
        _subfolder_cache = {}
    else:
        # 细粒度失效：清除受影响的子缓存
        _folder_tree_cache = None  # folder_tree 仍需重建
        
        # 清除以 affected_path 为前缀的缓存
        to_remove = [k for k in _subfolder_cache if k.startswith(affected_path)]
        for k in to_remove:
            del _subfolder_cache[k]
    
    # 清除 path_count 缓存
    try:
        from app.utils.path_count_cache import invalidate_path_count_cache
        invalidate_path_count_cache(affected_path)
    except ImportError:
        pass
```

#### 实现步骤

1. 修改 `invalidate_folder_tree_cache` 支持路径参数
2. 更新所有调用处，传入受影响的路径
3. 修改 `path_count_cache.py` 的 `invalidate_path_count_cache` 支持路径参数

---

### 5. LIKE 查询无法利用索引

**文件**: `app/utils/query_builder.py`、`app/utils/folder_tree.py`

#### 问题描述

一些查询使用前导通配符，无法利用数据库索引：

```python
# 无法利用索引
stmt = stmt.where(~Image.relative_path.like("%/%"))
```

#### 修复方案

使用 `SUBSTRING_INDEX` 提取路径前缀再匹配：

```python
# 优化：提取第一级路径后匹配，可以利用索引
stmt = stmt.where(
    func.substring_index(Image.relative_path, '/', 1) != ''
)
```

对于需要前导通配符的场景，考虑：

1. 添加反向索引列
2. 使用全文索引
3. 预先提取并存储路径组件

---

## 轻微问题

### 6. 图片信息 API 多次数据库查询

**文件**: `app/routers/images.py`  
**位置**: 第 248-278 行

#### 问题描述

获取图片信息执行了 2 次独立查询：

```python
# 查询 1：获取图片
result = await session.execute(select(Image).where(Image.id == image_id))
img = result.scalar_one_or_none()

# 查询 2：获取标签
tag_result = await session.execute(
    select(Tag.name)
    .join(ImageTag, ImageTag.tag_id == Tag.id)
    .where(ImageTag.image_id == image_id)
)
tags = [r[0] for r in tag_result.fetchall()]
```

#### 修复方案

使用 JOIN 一次查询：

```python
# 优化后：单次查询获取图片和标签
from sqlalchemy import func

stmt = (
    select(Image, func.group_concat(Tag.name).label('tags'))
    .outerjoin(ImageTag, ImageTag.image_id == Image.id)
    .outerjoin(Tag, Tag.id == ImageTag.tag_id)
    .where(Image.id == image_id)
    .group_by(Image.id)
)
result = await session.execute(stmt)
row = result.one_or_none()

if row:
    img = row[0]
    tags = (row[1] or '').split(',') if row[1] else []
```

---

### 7. 前端缺少请求防抖

**文件**: `app/templates/` 中的 JS 代码

#### 问题描述

搜索、筛选等操作没有防抖，可能发送大量请求。

#### 修复方案

添加防抖函数：

```javascript
// app/static/js/utils.js 新增

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 使用示例
const debouncedSearch = debounce(function(q) {
    htmx.ajax('GET', '/api/search-dirs', {
        target: '#search-results',
        swap: 'innerHTML'
    });
}, 300);
```

---

### 8. 扫描时内存占用高

**文件**: `app/services/scanner.py`  
**位置**: 第 306 行

#### 问题描述

`seen_in_run` 集合在百万级图库中会占用较多内存。

#### 修复方案

1. 使用数据库唯一索引约束替代内存集合
2. 使用流式处理，减少内存中维护的数据

---

### 9. 静态资源缓存

**文件**: `app/main.py` 或部署配置

#### 问题描述

CSS/JS 没有设置长期缓存策略。

#### 修复方案

配置静态文件中间件：

```python
# app/main.py
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse

# 配置长期缓存
app.mount("/static", StaticFiles(directory="app/static"), name="static")
```

在部署层面（Nginx）配置：

```nginx
location /static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

---

## 待办事项清单

### 高优先级

- [ ] **实现上传快速哈希去重** (`app/utils/hash_utils.py`, `app/routers/images.py`)
  - [ ] 创建 `compute_quick_hash` 函数
  - [ ] 修改 `_compute_existing_hashes` 使用快速哈希
  - [ ] 仅对冲突文件计算完整 MD5

- [ ] **优化搜索 API** (`app/routers/folders.py`)
  - [ ] 使用 SQL LIKE 预过滤
  - [ ] 减少返回的数据量

- [ ] **批量删除优化** (`app/routers/images.py`)
  - [ ] 收集路径后批量删除
  - [ ] 使用 `asyncio.gather` 并行删除

### 中优先级

- [ ] **缓存失效优化** (`app/utils/folder_tree.py`, `app/utils/path_count_cache.py`)
  - [ ] 支持细粒度缓存失效
  - [ ] 更新调用处传入受影响路径

- [ ] **LIKE 查询优化** (`app/utils/query_builder.py`)
  - [ ] 使用 `SUBSTRING_INDEX` 替代前导通配符

- [ ] **图片信息 API 优化** (`app/routers/images.py`)
  - [ ] 使用 JOIN 合并查询

### 低优先级

- [ ] **添加前端防抖** (`app/static/js/`)
- [ ] **扫描内存优化** (`app/services/scanner.py`)
- [ ] **静态资源缓存配置**

---

## 相关文件索引

| 问题 | 涉及文件 |
|------|----------|
| 上传哈希 | `app/routers/images.py`, `app/utils/hash_utils.py` |
| 搜索优化 | `app/routers/folders.py`, `app/utils/search.py` |
| 批量删除 | `app/routers/images.py`, `app/utils/images.py` |
| 缓存失效 | `app/utils/folder_tree.py`, `app/utils/path_count_cache.py` |
| LIKE 查询 | `app/utils/query_builder.py` |
| 图片信息 | `app/routers/images.py` |
| 前端防抖 | `app/static/js/utils.js` |
| 扫描优化 | `app/services/scanner.py` |

---

## 测试建议

优化后应进行以下测试：

1. **上传性能测试**：万级图库下上传新图片的准备时间
2. **搜索响应测试**：搜索 API 的响应时间
3. **批量删除测试**：删除 100+ 张图片的耗时
4. **缓存命中率测试**：频繁文件夹操作后的缓存状态
5. **内存占用测试**：扫描百万级图片时的内存使用
