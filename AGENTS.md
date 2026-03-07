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
# 运行所有测试
uv run pytest

# 运行单个测试
uv run pytest tests/test_db.py::test_db_connection -v
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

- **导入**: 标准库 → 第三方 → 本地模块
- **类型**: 使用 Python 3.13+ 原生注解 (`int | None`)
- **异步**: 路由用 `async def`，内部计算用同步函数
- **路径**: `normalize_path(path, allow_empty=True/False)`
- **SQL LIKE**: 必须用 `escape_like(value)` 转义，配合 `LIKE_ESCAPE = "!"`
- **错误**: `raise HTTPException(status_code=4xx, detail="简短中文")`
- **命名**: 私有函数加 `_` 前缀

### 前端

- **页面**: 继承 `base.html`，用 `{% block content %}`
- **HTMX**: `hx-get` + `hx-target` + `hx-swap="innerHTML"`
- **安全**: `innerHTML` 写入用户数据必须 `escapeHtml()`

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_HOST` | - | 数据库地址 |
| `ACCESS_PASSWORD` | - | 访问密码(空则无需登录) |
| `MAX_UPLOAD_FILE_SIZE` | 100M | 单文件上限 |
| `TZ` | Asia/Shanghai | 时区 |

## 常用路径

- 图片: `photos/` | 缓存: `cache/{hash[:2]}/{hash[2:4]}/{hash[4:]}.webp`
- 静态: `app/static/` | CSS: `src/input.css`

## 注意事项

- 开发用 `docker-compose.dev.yml` 只启动 MariaDB
- ruff 排除测试目录检查
