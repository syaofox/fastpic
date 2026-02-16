# ============================================================
# FastPic Docker Image — 多阶段构建，最小化镜像
# ============================================================

# ---------- 阶段 1: 构建依赖 ----------
FROM python:3.12-slim AS builder

# 安装 uv（极速 Python 包管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 使用与运行时相同的路径，确保 venv 内 shebang 路径一致
WORKDIR /app

# 先拷贝依赖清单，利用 Docker 缓存层
COPY pyproject.toml uv.lock ./

# 安装依赖到 /app/.venv（不安装项目本身）
RUN uv sync --frozen --no-install-project --no-dev

# 拷贝应用源码
COPY . .
# 确保 static 目录存在（favicon.ico 等静态资源）
RUN mkdir -p /app/static

# ---------- 阶段 2: 运行时镜像 ----------
FROM python:3.12-slim AS runtime

# 时区支持 + ffmpeg（视频缩略图生成）
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户（默认 UID/GID 1000，可通过构建参数覆盖）
ARG PUID=1000
ARG PGID=1000
RUN groupadd -g ${PGID} fastpic && useradd -u ${PUID} -g fastpic -d /app -s /sbin/nologin fastpic

WORKDIR /app

# 从构建阶段拷贝虚拟环境和源码
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/*.py /app/
COPY --from=builder /app/pyproject.toml /app/
COPY --from=builder /app/templates /app/templates
COPY --from=builder /app/static /app/static

# 创建数据目录并设置权限
RUN mkdir -p /app/photos /app/cache /app/data \
    && chown -R fastpic:fastpic /app

# 设置环境变量
# 默认时区 Asia/Shanghai，可通过 docker-compose 覆盖
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 切换到非 root 用户
USER fastpic

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
