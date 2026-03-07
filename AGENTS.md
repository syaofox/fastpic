# FastPic 开发指南

面向 Agentic Coding 助手的开发指南。

## 项目概览

- **技术栈**: FastAPI + HTMX + SQLModel + MariaDB + Jinja2
- **Python**: 3.13+ | **环境管理**: uv | **部署**: Docker

## 开发命令

```bash
# 安装依赖
uv sync && npm install

# 启动开发服务
./dev-run.sh start -d    # 后台运行
./dev-run.sh stop        # 停止

# 手动启动 (需先启动 MariaDB)
docker compose -f docker-compose.dev.yml up -d
export MYSQL_HOST=127.0.0.1
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 测试

```bash
# 确保 MariaDB 已启动 (测试需要数据库)
docker compose -f docker-compose.dev.yml up -d

# 运行所有测试
uv run pytest

# 运行单个测试
uv run pytest tests/test_db.py::test_db_connection -v

# 运行指定测试文件
uv run pytest tests/test_db.py -v

# 运行包含关键字的测试
uv run pytest -k "test_image" -v

# 查看测试输出
uv run pytest -v -s
```

## 代码检查

```bash
uv run ruff check .          # 检查
uv run ruff check . --fix    # 自动修复
uv run ruff format .         # 格式化
```

- 行长度: 120 | 规则: E, W, F, I, UP | 测试目录已排除

## 代码风格

### Python

- **导入顺序**: 标准库 → 第三方 → 本地模块
- **类型注解**: 使用 Python 3.13+ 原生注解 (`int | None`)
- **异步**: 路由用 `async def`，内部计算用同步函数
- **路径处理**: 使用 `normalize_path(path, allow_empty=True/False)` 规范化输入
- **SQL LIKE 查询**: 必须用 `escape_like(value)` 转义，配合 `LIKE_ESCAPE = "!"`
- **错误处理**: `raise HTTPException(status_code=4xx, detail="简短中文")`
- **命名**: 私有函数/变量加 `_` 前缀，常量全大写
- **异常**: 使用具体异常类，捕获时指定具体类型

### 前端

- **页面模板**: 继承 `base.html`，用 `{% block content %}`
- **HTMX**: `hx-get` + `hx-target` + `hx-swap="innerHTML"`
- **安全**: `innerHTML` 写入用户数据必须用 `escapeHtml()` 转义
- **CSS**: 使用 Tailwind 类

## 核心工具函数

### 路径工具 (app/utils/path_utils.py)

```python
from app.utils.path_utils import normalize_path, escape_like, LIKE_ESCAPE, resolve_and_validate_relative_path

# 规范化用户输入路径
path = normalize_path(user_input, allow_empty=False)

# SQL LIKE 转义 (MariaDB 使用 ! 转义)
escaped = escape_like(user_search_term)
results = session.exec(select(Image).where(Image.path.like(f"%{escaped}%", escape=LIKE_ESCAPE)))
```

### 数据库模型 (app/models.py)

```python
from app.models import Image, Folder, Tag

# 使用 SQLModel，自动创建外键关系
```

### 工具模块

| 模块 | 用途 |
|------|------|
| `app/utils/images.py` | 图片处理、缩略图、格式转换 |
| `app/utils/cache_utils.py` | 缓存管理 |
| `app/utils/search.py` | 搜索功能 |
| `app/utils/query_builder.py` | SQL 查询构建 |
| `app/services/scanner.py` | 图片扫描服务 |
| `app/services/watcher.py` | 文件监控服务 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_HOST` | - | 数据库地址(必填) |
| `ACCESS_PASSWORD` | - | 访问密码(空则无需登录) |
| `MAX_UPLOAD_FILE_SIZE` | 100M | 单文件上限 |
| `TZ` | Asia/Shanghai | 时区 |
| `PHOTOS_DIR` | photos/ | 图片目录 |

## 常用路径

- 图片存储: `photos/`
- 缩略图缓存: `cache/{hash[:2]}/{hash[2:4]}/{hash[4:]}.webp`
- 静态文件: `app/static/`
- 样式文件: `src/input.css`

## 测试规范

- 测试文件放在 `tests/` 目录
- 使用 `pytest` + `pytest-asyncio`
- 测试需要 MariaDB，确保数据库已启动
- 使用 `conftest.py` 中的 fixture 设置测试环境变量
- 数据库测试: 测试前确保数据准备充分，测试后清理

## 注意事项

- 开发用 `docker-compose.dev.yml` 只启动 MariaDB
- ruff 排除测试目录检查
- 图片上传/处理使用同步函数避免阻塞事件循环
- 生产环境使用 Docker 部署

## 快速参考

```python
# 创建图片记录
from app.utils.image_records import create_image_record
record = await create_image_record(file_path, folder_id)

# 批量处理图片
from app.utils.image_batch import ImageBatch
batch = ImageBatch(paths)
results = await batch.process()

# 搜索图片
from app.utils.search import search_images
results = await search_images(query, folder_id=None, tags=None)
```
