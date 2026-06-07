import os
import re
import sys
from urllib.parse import quote

from sqlalchemy import BigInteger, Column, String, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import Field, SQLModel, create_engine

# 数据库配置：仅支持 MariaDB，需设置 MYSQL_HOST
# 检查推迟到 init_db()，便于测试时 mock 或通过 conftest 设置 MYSQL_HOST
_MYSQL_HOST = os.environ.get("MYSQL_HOST", "").strip()
_MYSQL_USER = os.environ.get("MYSQL_USER", "fastpic")
_MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "fastpic")
_MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "fastpic")

_db_user = quote(_MYSQL_USER, safe="")
_db_pass = quote(_MYSQL_PASSWORD, safe="")
if _MYSQL_HOST:
    DATABASE_URL = f"mysql+pymysql://{_db_user}:{_db_pass}@{_MYSQL_HOST}/{_MYSQL_DATABASE}"
    ASYNC_DATABASE_URL = f"mysql+aiomysql://{_db_user}:{_db_pass}@{_MYSQL_HOST}/{_MYSQL_DATABASE}"
else:
    # 占位 URL，仅用于导入；实际连接前 init_db() 会检查 MYSQL_HOST
    DATABASE_URL = "mysql+pymysql://fake:fake@127.0.0.1:3306/fake"
    ASYNC_DATABASE_URL = "mysql+aiomysql://fake:fake@127.0.0.1:3306/fake"

# 连接池：可通过环境变量调优，生产环境建议 pool_size=20, max_overflow=40
_db_pool_size = int(os.environ.get("DB_POOL_SIZE", "20"))
_db_max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "40"))
_db_pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))  # 1 小时，避免 MariaDB wait_timeout 关闭后复用

# 自然排序：数字按数值排（1,2,10,100），非数字按字典序。用于生成可比较的 sort key
_NATURAL_PAD = 10

# natural 列长度：含大量数字的文件名（如 UUID、时间戳）经 natural_sort_key 补零后会显著变长
_FILENAME_NATURAL_LEN = 512
_RELATIVE_PATH_NATURAL_LEN = 2048


def natural_sort_key(s: str) -> str:
    """将字符串转为自然排序键：数字段左补零，使 1<2<10<100"""
    return re.sub(r"\d+", lambda m: m.group(0).zfill(_NATURAL_PAD), s or "")


class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    relative_path: str = Field(unique=True, index=True)
    modified_at: float = Field(index=True)
    file_size: int = Field(default=0, sa_column=Column(BigInteger, default=0, index=True))
    width: int = 0
    height: int = 0
    filename_natural: str | None = Field(
        default=None,
        sa_column=Column(String(_FILENAME_NATURAL_LEN), index=True),
    )
    relative_path_natural: str | None = Field(
        default=None,
        sa_column=Column(String(_RELATIVE_PATH_NATURAL_LEN), index=False),
    )
    media_type: str = Field(default="image", index=True)  # "image" | "video"
    md5_hash: str | None = Field(
        default=None,
        sa_column=Column(String(32), index=True),
    )


class Tag(SQLModel, table=True):
    __tablename__ = "tags"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)


class ImageTag(SQLModel, table=True):
    __tablename__ = "image_tags"

    image_id: int = Field(foreign_key="images.id", primary_key=True)
    tag_id: int = Field(foreign_key="tags.id", primary_key=True)


class PathCountCache(SQLModel, table=True):
    """path 下图片数量持久化缓存，减轻百万级 COUNT 查询"""

    __tablename__ = "path_count_cache"

    path: str = Field(primary_key=True)
    mode: str = Field(primary_key=True)
    total: int = 0
    updated_at: float = 0


class FolderThumbnail(SQLModel, table=True):
    """用户指定的文件夹缩略图，优先于自动选取。同一文件夹最多 4 张。"""

    __tablename__ = "folder_thumbnails"

    folder_path: str = Field(primary_key=True)
    image_relative_path: str = Field(primary_key=True)
    display_order: int = 0


class Task(SQLModel, table=True):
    """持久化任务记录，支持多任务并发追踪和历史查看"""

    __tablename__ = "tasks"

    id: str = Field(primary_key=True)
    task_type: str = Field(index=True)
    title: str = ""
    status: str = Field(default="pending", index=True)  # pending | running | completed | failed | cancelled
    progress_percent: float = 0.0
    current_operation: str = ""
    total_items: int = 0
    completed_items: int = 0
    error_message: str | None = None
    result_summary: str | None = None
    created_at: float = 0
    started_at: float | None = None
    finished_at: float | None = None


sync_engine = create_engine(DATABASE_URL, echo=False)
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=_db_pool_size,
    max_overflow=_db_max_overflow,
    pool_recycle=_db_pool_recycle,
)
async_session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def _run_natural_sort_index_migration() -> None:
    """为 relative_path_natural 添加自然排序索引"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        try:
            conn.execute(text("CREATE INDEX ix_images_relative_path_natural ON images(relative_path_natural(512))"))
        except Exception:
            pass
        conn.commit()


def _run_performance_indexes_migration() -> None:
    """添加性能优化索引"""
    from sqlalchemy import text

    indexes = [
        ("ix_images_path_media", "CREATE INDEX ix_images_path_media ON images(relative_path(255), media_type)"),
        ("ix_image_tags_tag_id", "CREATE INDEX ix_image_tags_tag_id ON image_tags(tag_id)"),
        (
            "ix_folder_thumbnails_folder",
            "CREATE INDEX ix_folder_thumbnails_folder ON folder_thumbnails(folder_path(255), display_order)",
        ),
        ("ix_images_mod_size", "CREATE INDEX ix_images_mod_size ON images(modified_at, file_size)"),
        ("ix_images_filename_media", "CREATE INDEX ix_images_filename_media ON images(filename(255), media_type)"),
    ]

    with sync_engine.connect() as conn:
        for index_name, create_sql in indexes:
            r = conn.execute(
                text(
                    f"SELECT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS "
                    f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ('images', 'image_tags', 'folder_thumbnails') "
                    f"AND INDEX_NAME = '{index_name}'"
                )
            )
            if r.fetchone() is None:
                try:
                    conn.execute(text(create_sql))
                except Exception:
                    pass
        conn.commit()


def _run_fulltext_migration() -> None:
    """为 images.filename 添加 FULLTEXT 索引，支持百万级搜索"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(
            text(
                "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' "
                "AND INDEX_TYPE = 'FULLTEXT' AND COLUMN_NAME = 'filename'"
            )
        )
        if r.fetchone() is None:
            try:
                conn.execute(text("CREATE FULLTEXT INDEX ft_images_filename ON images(filename)"))
                conn.commit()
            except Exception:
                conn.rollback()


def _run_md5_hash_migration() -> None:
    """添加 md5_hash 列及索引"""
    with sync_engine.connect() as conn:
        r = conn.execute(
            text(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' AND COLUMN_NAME = 'md5_hash'"
            )
        )
        if r.fetchone() is None:
            conn.execute(text("ALTER TABLE images ADD COLUMN md5_hash VARCHAR(32) DEFAULT NULL"))
            conn.execute(text("CREATE INDEX idx_images_md5_hash ON images(md5_hash)"))
            conn.commit()


def init_db() -> None:
    """创建数据库表（仅支持全新部署）"""
    if not _MYSQL_HOST:
        print("错误: 请设置 MYSQL_HOST 环境变量连接 MariaDB", file=sys.stderr)
        sys.exit(1)
    SQLModel.metadata.create_all(sync_engine)
    _run_natural_sort_index_migration()
    _run_performance_indexes_migration()
    _run_fulltext_migration()
    _run_md5_hash_migration()


async def get_async_session():
    async with async_session_factory() as session:
        yield session
