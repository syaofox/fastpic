# FastPic 开发指南

面向 Agentic Coding 助手的开发指南。

## 项目概览

- **技术栈**: FastAPI + HTMX + Tailwind CSS + SQLModel + MariaDB + Jinja2
- **Python**: 3.13+ | **环境管理**: uv | **部署**: Docker
- **测试框架**: pytest + pytest-asyncio

## 开发命令

```bash
# 安装依赖（Python + 前端）
uv sync && npm install

# 启动开发服务
./dev-run.sh start        # 前台运行
./dev-run.sh start -d     # 后台运行
./dev-run.sh stop         # 停止

# 手动启动 (需先启动 MariaDB)
docker compose -f docker-compose.dev.yml up -d
export MYSQL_HOST=127.0.0.1
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# CSS 开发（监听变化自动编译）
./dev-run.sh watch:css
./dev-run.sh build:css    # 一次性编译
```

## 测试

```bash
# 确保 MariaDB 已启动（测试需要数据库）
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

# 触发手动扫描
curl -X POST http://localhost:8000/scan
```

## 代码检查

```bash
# Ruff 检查
uv run ruff check .          # 检查
uv run ruff check . --fix     # 自动修复
uv run ruff format .          # 格式化

# Pyright 类型检查（可选）
uv run pyright
```

- 行长度: 120 | Ruff 规则: E, W, F, I, UP | 测试目录已排除
- 测试时只检查 py 文件: `uv run ruff check . --select E,W,F,I,UP --exclude tests/`

## 代码风格

### Python

- **导入顺序**: 标准库 → 第三方 → 本地模块
- **类型注解**: 使用 Python 3.13+ 原生注解 (`int | None`)
- **异步**: 路由用 `async def`，内部计算用同步函数（避免阻塞事件循环）
- **路径处理**: 使用 `normalize_path(path, allow_empty=True/False)` 规范化输入
- **SQL LIKE 查询**: 必须用 `escape_like(value)` 转义，配合 `LIKE_ESCAPE = "!"`
- **错误处理**: `raise HTTPException(status_code=4xx, detail="简短中文")`
- **API 响应**: 使用 `ApiResponse` 统一响应格式，返回 `ApiResponse.success/error/partial`
- **WebSocket**: 使用 `message_broadcaster` 广播消息到客户端
- **任务进度**: 使用 `task_state.async_update_progress()` 同步状态并推送 WebSocket
- **命名**: 私有函数/变量加 `_` 前缀，常量全大写
- **异常**: 使用具体异常类，捕获时指定具体类型
- **行长度**: 最大 120 字符

### 前端

- **页面模板**: 继承 `base.html`，用 `{% block content %}`
- **HTMX**: `hx-get` + `hx-target` + `hx-swap="innerHTML"`
- **安全**: `innerHTML` 写入用户数据必须用 `escapeHtml()` 转义
- **CSS**: 使用 Tailwind 类
- **JS 模块**: 使用 ES6 模块，通过 `main.js` 导出到 window
- **操作服务**: 使用 `operationService` 封装 API 调用，自动处理响应和 Toast

## 项目结构

```
fastpic/
├── app/                          # 应用主目录
│   ├── main.py                   # FastAPI 应用入口
│   ├── config.py                 # 配置（环境变量解析）
│   ├── models.py                 # SQLModel 数据模型
│   ├── schemas.py                # Pydantic 请求/响应模型（包含 ApiResponse）
│   ├── app_common.py             # 公共依赖、依赖注入
│   ├── routers/                  # 路由模块
│   │   ├── auth.py              # 登录认证
│   │   ├── folders.py           # 文件夹 API
│   │   ├── images.py            # 图片/视频 API
│   │   ├── settings.py           # 设置页面
│   │   ├── tags.py               # 标签管理
│   │   └── websocket.py          # WebSocket 实时通信
│   ├── services/                 # 业务服务
│   │   ├── scanner.py           # 异步扫描与缩略图生成
│   │   ├── scan_state.py        # 扫描状态管理
│   │   ├── watcher.py           # 文件监控（热重载）
│   │   ├── message_broadcaster.py # WebSocket 消息广播
│   │   └── task_service.py      # 任务执行服务
│   ├── utils/                    # 工具函数
│   ├── templates/                # Jinja2 模板
│   └── static/                   # 静态资源
│       └── js/
│           ├── state/            # 状态管理（Signals）
│           │   ├── signals.js    # 信号实现
│           │   └── stores/       # 状态仓库
│           │       ├── taskStore.js
│           │       ├── galleryStore.js
│           │       └── selectionStore.js
│           ├── services/         # API 服务
│           │   ├── websocket.js  # WebSocket 管理
│           │   ├── api.js        # 统一 API 调用
│           │   └── operations.js # 操作服务
│           ├── components/       # UI 组件
│           │   ├── Toast.js      # Toast 通知
│           │   └── Progress.js   # 进度指示器
│           └── main.js           # 入口文件
├── src/input.css                 # Tailwind CSS 源
├── photos/                       # 图片根目录
├── cache/                        # 缩略图缓存（三层）
├── tests/                        # 测试文件
└── docker-compose.dev.yml        # 开发环境 MariaDB
```

## 核心工具函数

```python
from app.utils.path_utils import normalize_path, escape_like, LIKE_ESCAPE

# 规范化用户输入路径
path = normalize_path(user_input, allow_empty=False)

# SQL LIKE 转义（MariaDB 使用 ! 转义）
escaped = escape_like(user_search_term)
results = session.exec(select(Image).where(Image.path.like(f"%{escaped}%", escape=LIKE_ESCAPE)))

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

## 统一 API 响应格式

所有 API 端点返回统一格式 `ApiResponse`：

```python
from app.schemas import ApiResponse, ResponseStatus

# 成功响应
return ApiResponse.success({"deleted": 5}, "删除成功")
# {"status": "success", "message": "删除成功", "data": {"deleted": 5}, "affected": [], "errors": []}

# 错误响应
return ApiResponse.error("操作失败", ["错误信息"])
# {"status": "error", "message": "操作失败", "data": null, "affected": [], "errors": ["错误信息"]}

# 部分成功
return ApiResponse.partial("部分成功", data={"moved": 3}, errors=["部分错误"])
```

## 任务服务

使用 `task_service` 执行费时任务，自动广播进度：

```python
from app.services.task_service import task_service

# 注册任务处理器
@task_service.register("my-task")
async def handler(context, params):
    await context.broadcast_progress(5, 10, "处理中")
    return {"result": "ok"}

# 执行任务
result = await task_service.execute("my-task", "任务标题", params, 10)
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_HOST` | - | 数据库地址（必填） |
| `ACCESS_PASSWORD` | - | 访问密码（空则无需登录） |
| `MAX_UPLOAD_FILE_SIZE` | 100M | 单文件上限 |
| `MAX_UPLOAD_TOTAL_SIZE` | 5000M | 总上传上限 |
| `TZ` | Asia/Shanghai | 时区 |
| `PHOTOS_DIR` | photos/ | 图片目录 |

## 功能特性

- **图片/视频支持**: 支持常见图片格式和 mp4/webm/mov/mkv/ts 视频
- **视频缩略图**: 需系统安装 ffmpeg
- **无限滚动**: 滚动到底部自动加载下一页
- **大图预览**: 点击缩略图打开模态框，支持左右切换、ESC/遮罩关闭
- **文件夹树**: 左侧导航按目录筛选
- **实时搜索**: 搜索框 300ms 防抖，支持简繁转换和拼音
- **WebSocket 实时通信**: 任务进度实时推送，替代旧 SSE 方案
- **统一 API 响应**: 所有 API 返回 `ApiResponse` 格式

## 前端状态管理

使用 Signals 模式管理前端状态：

```javascript
import { galleryStore } from './state/stores/galleryStore.js';
import { selectionStore } from './state/stores/selectionStore.js';
import { operationService } from './services/operations.js';

// 使用 galleryStore
galleryStore.setPath('/photos');
galleryStore.invalidateCache();

// 使用 selectionStore
selectionStore.toggleImage(id);
selectionStore.clearSelection();

// 使用 operationService（推荐）
operationService.deleteImages([1, 2, 3]);
operationService.moveImages(ids, targetPath);
operationService.batchDelete(imageIds, folderPaths);
```

## 测试规范

- 测试文件放在 `tests/` 目录
- 使用 `pytest` + `pytest-asyncio`
- 测试需要 MariaDB，确保数据库已启动
- 使用 `conftest.py` 中的 fixture 设置测试环境变量
- 数据库测试: 测试前确保数据准备充分，测试后清理

## 生产部署

```bash
# Docker 构建
docker compose build
docker compose up -d
docker compose logs -f
```

## 注意事项

- ruff 排除测试目录检查
- 图片上传/处理使用同步函数避免阻塞事件循环
- 生产环境使用 Docker 部署
- 开发用 `docker-compose.dev.yml` 只启动 MariaDB
