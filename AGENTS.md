# FastPic 开发指南

本文件面向 Agentic Coding 助手，提供项目的开发、测试和代码风格指南。

## 项目概述

- **技术栈**: FastAPI + HTMX + SQLModel + MariaDB + Jinja2
- **Python 版本**: 3.13+
- **环境管理**: uv（虚拟环境）
- **部署**: Docker + docker-compose

## 开发环境

### 环境准备

```bash
# 安装依赖（自动创建 .venv）
uv sync

# 安装前端依赖（Tailwind CSS）
npm install
```

### 前端开发（Tailwind CSS）

```bash
# 开发模式：监听文件变化自动编译
./dev-run.sh watch:css

# 或手动编译一次
./dev-run.sh build:css
```

### 启动开发服务

```bash
# 方式1：使用脚本（推荐）
./dev-run.sh start          # 前台运行
./dev-run.sh start -d       # 后台运行
./dev-run.sh stop           # 停止服务
./dev-run.sh restart        # 重启
./dev-run.sh status         # 查看状态

# 方式2：手动启动
# 1. 启动 MariaDB
docker compose -f docker-compose.dev.yml up -d

# 2. 设置环境变量并运行
export MYSQL_HOST=127.0.0.1
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 部署

```bash
# 构建并启动
docker compose build
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

## 测试

### 运行测试

```bash
# 需先启动 MariaDB
docker compose -f docker-compose.dev.yml up -d

# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_db.py

# 运行单个测试函数
uv run pytest tests/test_db.py::test_db_connection -v
```

### 测试文件位置

- `tests/` 目录下
- 使用 pytest + pytest-asyncio
- 配置文件: `pyproject.toml` 中 `[tool.pytest.ini_options]`

### 代码检查

```bash
# 运行 ruff 检查
uv run ruff check .

# 自动修复可自动修复的问题
uv run ruff check . --fix
```

- 使用 ruff 进行代码检查
- 配置文件: `pyproject.toml` 中 `[tool.ruff]`

## 代码风格指南

### 后端 (Python)

#### 导入组织

```python
# 标准库 → 第三方库 → 本地模块
import asyncio
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from sqlmodel import select

from app.config import (...)
from app.models import (...)
from app.routers import auth, tags, images
from app.utils.path_utils import normalize_path
```

#### 类型注解

- 使用 Python 3.13+ 原生类型注解 (`int | None` 而非 `Optional[int]`)
- 函数参数和返回值尽量添加类型注解
- 复杂泛型使用 `typing` 模块

#### 异步/同步

- **路由与对外 API**: 使用 `async def`
- **Session**: 统一使用 `Depends(get_async_session)`
- **内部纯计算/IO**: 使用同步 `def`，在路由里用 `await asyncio.to_thread(...)` 调用

#### 路径规范

- 路径参数统一使用 `utils.path_utils.normalize_path(path, allow_empty=True/False)`
- 非法路径返回 `None`；需空字符串时用 `normalize_path(...) or ""`

#### 错误处理

- 抛出 HTTP 异常: `raise HTTPException(status_code=4xx/5xx, detail="简短中文描述")`
- 页面响应: `templates.TemplateResponse(...)`
- 重定向: `RedirectResponse(url=..., status_code=302)`

#### SQL LIKE 转义

- 用户输入做 `ilike`/`like` 前必须用 `utils.path_utils.escape_like(value)`
- 配合 `LIKE_ESCAPE`（MariaDB 使用 `"!"`）
- 示例: `Image.filename.ilike(f"%{escaped}%", escape=LIKE_ESCAPE)`

#### 命名规范

- 仅模块内部使用的函数: 以单下划线前缀 `_`
- 可跨模块复用的函数: 放在 `utils/` 或 `models.py`

#### 常用工具函数

- 简繁转换/拼音: `utils.search` 模块 (`to_simplified`, `to_traditional`, `to_pinyin_lower`, `search_match`)
- 自然排序: `models.py` 的 `natural_sort_key(s)`
- 文件夹树: `utils.folder_tree` 模块
- 图片查询: `utils.query_builder` 模块

### 前端 (Jinja2 + HTMX + Tailwind)

#### 页面结构

- 页面一律继承 `templates/base.html`
- 使用 `{% block content %}`
- 公共片段放 `templates/partials/`
- 可复用宏放 `templates/macros/`

#### 样式规范

- 使用 Tailwind CSS 编译后的样式 (`app/static/assets/styles.css`)
- 主色: `blue-500/600`，中性色: `slate-*`
- 圆角: `rounded-lg`
- 输入/按钮: `focus:ring-2 focus:ring-blue-500 focus:border-blue-500`

#### HTMX 约定

- 列表刷新: `hx-get` + `hx-target` + `hx-swap="innerHTML"`
- 全局状态: `hx-include` 指向顶栏隐藏 input
- 参数一致: 链接/按钮触发替换时带齐 `path, search, mode, sort_by, sort_order, page, cols, filter_*`
- 路径编码: 使用 `|urlencode_path` 过滤器

#### 安全

- `innerHTML` 写入用户可控数据必须使用 `escapeHtml()` 或 `escapeAttr()` 转义
- 防止 XSS 攻击

## 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_HOST` | - | 数据库地址 |
| `ACCESS_PASSWORD` | - | 访问密码（设为空则无需登录） |
| `MAX_UPLOAD_FILE_SIZE` | `100M` | 单文件上传上限 |
| `MAX_UPLOAD_TOTAL_SIZE` | `500M` | 总上传上限 |
| `TZ` | `Asia/Shanghai` | 时区 |
| `SKIP_FULL_SCAN_ON_STARTUP` | - | 设为 `1` 启动时跳过全量扫描 |

## 常用路径

- 图片根目录: `photos/`
- 缩略图缓存: `cache/`（三层结构: `cache/{hash[:2]}/{hash[2:4]}/{hash[4:]}.webp`）
- 静态资源: `app/static/`
- CSS 源文件: `src/input.css`
- 数据库: MariaDB（生产 `data/`，本地 `data-dev/`）

## 注意事项

- Python 3.13+ 必需
- 开发时使用 `docker-compose.dev.yml` 只启动 MariaDB，应用在宿主机运行便于调试
- 测试需 MariaDB 运行，conftest 会自动设置 `MYSQL_HOST=127.0.0.1`
