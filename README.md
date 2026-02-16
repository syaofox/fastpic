# FastPic - 局域网图片查看器

基于 FastAPI + HTMX + SQLModel 的局域网图片查看器，支持异步扫描、缩略图生成、无限滚动、实时搜索、大图预览和文件夹树导航。

## 快速开始

```bash
# 安装依赖（uv 会自动创建虚拟环境）
uv sync

# 将图片放入 photos/ 目录
# 启动服务
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

然后访问 http://localhost:8000

## 功能

- **异步扫描**：启动时自动扫描 `photos/` 目录，生成 300px 宽缩略图到 `cache/`
- **视频支持**：支持 mp4、webm、mov、mkv 格式，可在线播放（需系统安装 ffmpeg 以生成视频缩略图）
- **文件夹树**：左侧导航按目录筛选
- **实时搜索**：搜索框输入 300ms 防抖后刷新网格
- **无限滚动**：滚动到底部自动加载下一页
- **大图预览**：点击缩略图打开模态框，支持左右切换、ESC 关闭、点击遮罩关闭

## 手动触发扫描

```bash
curl -X POST http://localhost:8000/scan
```

## 项目结构

```
fastpic/
├── main.py          # FastAPI 应用、路由
├── models.py        # Image 模型
├── scanner.py       # 异步扫描与缩略图生成
├── templates/       # Jinja2 模板
├── photos/          # 图片根目录
└── cache/           # 缩略图缓存
```
