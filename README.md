# FastPic - 局域网图片查看器

基于 FastAPI + HTMX + Tailwind CSS + SQLModel 的局域网图片查看器，支持异步扫描、缩略图生成、无限滚动、实时搜索、大图预览和文件夹树导航。

## 快速开始

```bash
# 安装依赖（uv 会自动创建虚拟环境）
uv sync

# 安装前端依赖（Tailwind CSS）
npm install

# 启动开发服务
./dev-run.sh start
```

然后访问 http://localhost:8000

## 开发

### 前端开发

```bash
# 开发模式：监听 CSS 文件变化自动编译
./dev-run.sh watch:css

# 或手动编译一次
./dev-run.sh build:css
```

### 后端开发

```bash
# 启动开发服务（前台）
./dev-run.sh start

# 后台运行
./dev-run.sh start -d

# 查看日志
tail -f /tmp/fastpic.log
```

### 手动触发扫描

```bash
curl -X POST http://localhost:8000/scan
```

## 生产部署

### Docker 构建

```bash
# 构建镜像
docker compose build

# 启动容器
docker compose up -d

# 查看日志
docker compose logs -f
```

首次启动时会自动扫描 `photos/` 目录并生成缩略图。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_HOST` | - | 数据库地址（生产环境需要） |
| `ACCESS_PASSWORD` | - | 访问密码 |
| `MAX_UPLOAD_FILE_SIZE` | `100M` | 单文件上传上限 |
| `MAX_UPLOAD_TOTAL_SIZE` | `5000M` | 总上传上限 |
| `TZ` | `Asia/Shanghai` | 时区 |

## 功能

- **异步扫描**：启动时自动扫描 `photos/` 目录，生成 300px 宽缩略图到 `cache/`
- **视频支持**：支持 mp4、webm、mov、mkv、ts 格式，可在线播放（需系统安装 ffmpeg 以生成视频缩略图）
- **文件夹树**：左侧导航按目录筛选
- **实时搜索**：搜索框输入 300ms 防抖后刷新网格
- **无限滚动**：滚动到底部自动加载下一页
- **大图预览**：点击缩略图打开模态框，支持左右切换、ESC 关闭、点击遮罩关闭

## 项目结构

```
fastpic/
├── app/                          # 应用主目录
│   ├── __init__.py
│   ├── main.py                   # FastAPI 应用入口、路由挂载
│   ├── config.py                 # 配置（路径、环境变量解析）
│   ├── models.py                 # SQLModel 数据模型
│   ├── schemas.py                # Pydantic 请求/响应模型
│   ├── app_common.py             # 公共依赖、依赖注入
│   ├── routers/                  # 路由模块
│   │   ├── __init__.py
│   │   ├── auth.py               # 登录认证
│   │   ├── folders.py            # 文件夹相关 API
│   │   ├── images.py             # 图片/视频 API
│   │   ├── settings.py           # 设置页面
│   │   └── tags.py               # 标签管理
│   ├── services/                 # 业务服务
│   │   ├── scanner.py            # 异步扫描与缩略图生成
│   │   ├── scan_state.py         # 扫描状态管理
│   │   └── watcher.py            # 文件监控（热重载）
│   ├── utils/                    # 工具函数
│   │   ├── path_utils.py         # 路径规范化与安全检查
│   │   ├── hash_utils.py         # 文件哈希计算
│   │   ├── image_records.py      # 数据库记录操作
│   │   ├── images.py             # 图片处理
│   │   ├── image_batch.py        # 批量图片处理
│   │   ├── query_builder.py      # SQL 查询构建
│   │   ├── folder_tree.py        # 文件夹树结构
│   │   ├── search.py             # 搜索（简繁转换、拼音）
│   │   ├── tags.py               # 标签处理
│   │   ├── format.py             # 格式化（文件大小、日期）
│   │   └── stats.py              # 统计信息
│   ├── templates/                # Jinja2 模板
│   │   ├── base.html             # 基础模板
│   │   ├── index.html            # 首页（登录后重定向）
│   │   ├── gallery.html          # 图片画廊
│   │   ├── login.html            # 登录页
│   │   ├── settings.html         # 设置页
│   │   ├── macros/               # 可复用宏
│   │   └── partials/             # 公共片段
│   └── static/                   # 静态资源
│       ├── assets/
│       │   └── styles.css        # 编译后的 Tailwind CSS
│       ├── js/
│       │   ├── htmx.min.js       # HTMX 库
│       │   ├── mpegts.min.js     # 视频播放库
│       │   ├── utils.js          # 公共 JS 工具
│       │   ├── gallery.js        # 画廊交互逻辑
│       │   └── folder-browser.js # 文件夹浏览逻辑
│       └── favicon.ico
├── src/
│   └── input.css                 # Tailwind CSS 源文件
├── photos/                       # 图片根目录（挂载外部存储）
├── cache/                        # 缩略图缓存（三层结构）
├── data/                         # MariaDB 数据目录
├── tests/                        # 测试文件
│   ├── conftest.py               # pytest 配置
│   ├── test_db.py                # 数据库测试
│   └── test_filter_media_type.py # 媒体类型筛选测试
├── dev-run.sh                    # 开发脚本
├── Dockerfile                     # Docker 构建文件
├── docker-compose.yml            # 生产环境编排
├── docker-compose.dev.yml        # 开发环境编排
├── tailwind.config.js            # Tailwind 配置
├── package.json                  # npm 依赖
├── pyproject.toml                # Python 项目配置
└── uv.lock                       # uv 锁文件
```
