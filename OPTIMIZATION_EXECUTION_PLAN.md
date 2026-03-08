# FastPic 优化实施计划

> 生成日期: 2026-03-08
> 目的: 记录可执行的优化项及具体改动文件

---

## 优化优先级

| 优先级 | 优化项 | 预期收益 | 改动文件数 |
|--------|--------|----------|------------|
| P0 | 搜索 API 优化 | 高 | 1 |
| P1 | 标签预加载优化 | 中 | 2 |
| P2 | 前端加载优化 | 中 | 2 |

---

## P0: 搜索 API 优化

### 问题
`app/routers/folders.py:771-788` 的 `search_dirs` API:
- 当前获取全量 folder_counts 后在 Python 层遍历匹配
- 百万级图库时性能差

### 实施方案

**改动文件:**
- `app/routers/folders.py` - 修改 `search_dirs` API
- `app/utils/search.py` - 保持现有逻辑,作为第二层过滤

**修改内容:**

```python
# app/routers/folders.py 中的 search_dirs 函数

# 第一层: SQL LIKE 预过滤
escaped = escape_like(q)
prefix_filter = f"{escaped}%"

# 使用 SUBSTRING_INDEX 提取第一级路径后匹配
sql = text("""
    SELECT SUBSTRING_INDEX(relative_path, '/', 1) AS prefix, COUNT(*) AS cnt 
    FROM images 
    WHERE relative_path LIKE :prefix ESCAPE '!' 
    GROUP BY prefix
    HAVING cnt > 0
    LIMIT :limit
""")
result = await session.execute(sql, {"prefix": prefix_filter, "limit": limit * 5})
rows = result.fetchall()

# 第二层: 精确匹配 (简繁/拼音)
matched = []
for row in rows:
    dir_path, count = row[0], row[1]
    if search_match(q, dir_path):  # 复用现有函数
        matched.append({"path": dir_path, "image_count": count})
        if len(matched) >= limit:
            break

return {"dirs": matched}
```

**关联改动:**
- 无需修改其他文件

**测试要求:**
- [ ] 新增/修改测试: `tests/test_search_dirs.py`
- [ ] 测试用例:
  - [ ] 普通搜索返回正确结果
  - [ ] 简繁转换搜索 (如搜索"图片"匹配"圖片")
  - [ ] 拼音搜索 (如搜索"tupian"匹配"图片")
  - [ ] 搜索结果数量限制
  - [ ] 空搜索返回空列表
- [ ] 运行测试: `uv run pytest tests/test_search_dirs.py -v`

---

## P1: 标签预加载优化

### 问题
每次获取图片列表后需要单独查询标签

### 实施方案

**改动文件:**
- `app/models.py` - 定义 Image-Tag 关系
- `app/main.py` - 使用预加载

**修改内容:**

```python
# app/models.py 添加关系定义

class Image(SQLModel, table=True):
    # ... 现有字段 ...
    tags: list["Tag"] = Relationship(
        "Tag",
        back_populates="images",
        link_model=ImageTag,
    )


class Tag(SQLModel, table=True):
    # ... 现有字段 ...
    images: list["Image"] = Relationship(
        "Image",
        back_populates="tags",
        link_model=ImageTag,
    )
```

```python
# app/main.py 使用

# 在查询图片时预加载标签
stmt = (
    select(Image)
    .options(selectinload(Image.tags))
    .where(...)
)
```

**测试要求:**
- [ ] 新增/修改测试: `tests/test_tag_preload.py`
- [ ] 测试用例:
  - [ ] Image.tags 关系正确加载
  - [ ] Tag.images 关系正确加载
  - [ ] selectinload 预加载返回正确数据
  - [ ] 批量查询标签与预加载查询结果一致
- [ ] 运行测试: `uv run pytest tests/test_tag_preload.py -v`

---

## P2: 前端加载优化

### 问题
gallery 页面需要加载多个片段,用户感知加载时间长

### 实施方案

**改动文件:**
- `app/templates/gallery.html` - 添加骨架屏
- `app/static/js/gallery.js` - 优化加载逻辑 (如不存在需创建)

**修改内容:**

```html
<!-- app/templates/gallery.html 添加骨架屏 -->

<!-- 图片骨架 -->
<div id="gallery-grid" class="grid grid-cols-4 gap-4">
    {% for i in range(cols * 6) %}
    <div class="aspect-square bg-gray-200 animate-pulse rounded">
        <div class="w-full h-full bg-gray-300 rounded"></div>
    </div>
    {% endfor %}
</div>

<!-- 子文件夹骨架 -->
<div id="subfolders-grid">
    {% if defer_subfolders and not subfolders %}
    <div class="grid grid-cols-4 gap-4">
        {% for i in range(4) %}
        <div class="aspect-square bg-gray-200 animate-pulse rounded">
            <div class="w-full h-full bg-gray-300 rounded"></div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
```

```javascript
// app/static/js/gallery.js

// 页面加载时同时发起多个请求
document.addEventListener('DOMContentLoaded', () => {
    // 并行加载图片和子文件夹
    Promise.all([
        htmx.ajax('GET', '/gallery?path={{ path }}', { target: '#gallery-grid', swap: 'innerHTML' }),
        htmx.ajax('GET', '/api/gallery-subfolders?path={{ path }}', { target: '#subfolders', swap: 'innerHTML' })
    ]);
});
```

**测试要求:**
- [ ] 手动测试: 打开 gallery 页面,验证骨架屏显示正常
- [ ] 验证骨架屏在数据加载后正确替换
- [ ] 检查不同 cols 参数下骨架屏列数正确

---

## 实施检查清单

### P0: 搜索 API 优化
- [ ] 修改 `app/routers/folders.py` 的 `search_dirs` 函数
- [ ] 新增测试: `tests/test_search_dirs.py`
- [ ] 运行测试: `uv run pytest tests/test_search_dirs.py -v`
- [ ] 运行代码检查: `uv run ruff check app/routers/folders.py`

### P1: 标签预加载优化
- [ ] 修改 `app/models.py` 添加 Image-Tag 关系定义
- [ ] 修改 `app/main.py` 使用 selectinload 预加载
- [ ] 新增测试: `tests/test_tag_preload.py`
- [ ] 运行测试: `uv run pytest tests/test_tag_preload.py -v`
- [ ] 运行代码检查: `uv run ruff check app/models.py app/main.py`

### P2: 前端加载优化
- [ ] 修改 `app/templates/gallery.html` 添加骨架屏
- [ ] 创建/修改 `app/static/js/gallery.js`
- [ ] 手动测试: 打开 gallery 页面验证骨架屏显示
- [ ] 运行代码检查: `uv run ruff check app/templates/`

### 全局检查
- [ ] 运行全部测试: `uv run pytest`
- [ ] 运行代码检查: `uv run ruff check .`

---

## 改动文件索引

| 文件路径 | 改动内容 | 关联测试文件 |
|----------|----------|--------------|
| `app/routers/folders.py` | 搜索 API SQL 预过滤 | `tests/test_search_dirs.py` (新增) |
| `app/models.py` | 添加 Image-Tag 关系定义 | - |
| `app/main.py` | 使用 selectinload 预加载 | `tests/test_tag_preload.py` (新增) |
| `app/templates/gallery.html` | 添加骨架屏 | 手动测试 |
| `app/static/js/gallery.js` | 优化并行加载逻辑 | 手动测试 |

---

## 预期效果

| 优化项 | 预期改善 |
|--------|----------|
| 搜索 API | 响应时间从 O(n) 降到 O(预过滤结果数) |
| 标签预加载 | 减少一次 DB 往返 |
| 前端骨架屏 | 用户感知加载更快 |
