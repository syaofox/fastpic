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
├── app/
│   ├── main.py          # FastAPI 应用、路由
│   ├── models.py        # 数据模型
│   ├── config.py        # 配置
│   ├── templates/       # Jinja2 模板
│   ├── static/          # 静态资源（CSS/JS/图片）
│   └── utils/           # 工具函数
├── src/
│   └── input.css        # Tailwind CSS 源文件
├── photos/              # 图片根目录
├── cache/               # 缩略图缓存
└── dev-run.sh          # 开发脚本
```
