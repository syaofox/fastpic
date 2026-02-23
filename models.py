import os
import re
import sys

from sqlmodel import Field, SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# 数据库配置：仅支持 MariaDB，需设置 MYSQL_HOST
_MYSQL_HOST = os.environ.get("MYSQL_HOST", "").strip()
_MYSQL_USER = os.environ.get("MYSQL_USER", "fastpic")
_MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "fastpic")
_MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "fastpic")

if not _MYSQL_HOST:
    print("错误: 请设置 MYSQL_HOST 环境变量连接 MariaDB", file=sys.stderr)
    sys.exit(1)

_db_user = _MYSQL_USER
_db_pass = _MYSQL_PASSWORD.replace("%", "%%")
DATABASE_URL = f"mysql+pymysql://{_db_user}:{_db_pass}@{_MYSQL_HOST}/{_MYSQL_DATABASE}"
ASYNC_DATABASE_URL = f"mysql+aiomysql://{_db_user}:{_db_pass}@{_MYSQL_HOST}/{_MYSQL_DATABASE}"

# 连接池：可通过环境变量调优，生产环境建议 pool_size=20, max_overflow=40
_db_pool_size = int(os.environ.get("DB_POOL_SIZE", "10"))
_db_max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "20"))

# 自然排序：数字按数值排（1,2,10,100），非数字按字典序。用于生成可比较的 sort key
_NATURAL_PAD = 10


def natural_sort_key(s: str) -> str:
    """将字符串转为自然排序键：数字段左补零，使 1<2<10<100"""
    return re.sub(r"\d+", lambda m: m.group(0).zfill(_NATURAL_PAD), s or "")


class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    relative_path: str = Field(unique=True, index=True)
    modified_at: float = Field(index=True)
    file_size: int = Field(default=0, index=True)
    width: int = 0
    height: int = 0
    filename_natural: str | None = Field(default=None, index=True)
    relative_path_natural: str | None = Field(default=None, index=True)
    media_type: str = Field(default="image", index=True)  # "image" | "video"


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


sync_engine = create_engine(DATABASE_URL, echo=False)
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=_db_pool_size,
    max_overflow=_db_max_overflow,
)
async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


def _run_natural_sort_migration() -> None:
    """为已有表添加 filename_natural / relative_path_natural 列并回填"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' AND COLUMN_NAME = 'filename_natural'"
        ))
        if r.fetchone() is None:
            conn.execute(text("ALTER TABLE images ADD COLUMN filename_natural VARCHAR(255)"))
            conn.execute(text("ALTER TABLE images ADD COLUMN relative_path_natural VARCHAR(512)"))
            conn.commit()
        r = conn.execute(
            text("SELECT id, filename, relative_path FROM images WHERE filename_natural IS NULL")
        )
        rows = r.fetchall()
        if rows:
            for row in rows:
                kid, fn, rp = row
                nfn = natural_sort_key(fn or "")
                nrp = natural_sort_key(rp or "")
                conn.execute(
                    text("UPDATE images SET filename_natural = :nfn, relative_path_natural = :nrp WHERE id = :kid"),
                    {"nfn": nfn, "nrp": nrp, "kid": kid},
                )
            conn.commit()
        try:
            conn.execute(text("CREATE INDEX ix_images_filename_natural ON images(filename_natural)"))
        except Exception:
            pass
        try:
            conn.execute(text("CREATE INDEX ix_images_relative_path_natural ON images(relative_path_natural)"))
        except Exception:
            pass
        conn.commit()


def _run_media_type_migration() -> None:
    """为已有表添加 media_type 列并回填为 image"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' AND COLUMN_NAME = 'media_type'"
        ))
        if r.fetchone() is None:
            conn.execute(text("ALTER TABLE images ADD COLUMN media_type VARCHAR(32) DEFAULT 'image'"))
            conn.execute(text("UPDATE images SET media_type = 'image' WHERE media_type IS NULL"))
            conn.commit()
        try:
            conn.execute(text("CREATE INDEX ix_images_media_type ON images(media_type)"))
        except Exception:
            pass
        conn.commit()


def _run_fulltext_migration() -> None:
    """为 images.filename 添加 FULLTEXT 索引，支持百万级搜索"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text(
            "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'images' "
            "AND INDEX_TYPE = 'FULLTEXT' AND COLUMN_NAME = 'filename'"
        ))
        if r.fetchone() is None:
            try:
                conn.execute(text("CREATE FULLTEXT INDEX ft_images_filename ON images(filename)"))
                conn.commit()
            except Exception:
                conn.rollback()


def _run_path_count_cache_migration() -> None:
    """创建 path_count_cache 表（若不存在）"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'path_count_cache'"
        ))
        if r.fetchone() is None:
            conn.execute(text(
                "CREATE TABLE path_count_cache ("
                "path VARCHAR(512) NOT NULL, "
                "mode VARCHAR(32) NOT NULL, "
                "total INT NOT NULL DEFAULT 0, "
                "updated_at DOUBLE NOT NULL DEFAULT 0, "
                "PRIMARY KEY (path, mode))"
            ))
            conn.commit()


def _run_tags_migration() -> None:
    """创建 tags 和 image_tags 表（若不存在）"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tags'"
        ))
        if r.fetchone() is None:
            conn.execute(text(
                "CREATE TABLE tags (id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255) NOT NULL)"
            ))
            conn.execute(text("CREATE UNIQUE INDEX ix_tags_name ON tags (name)"))
            conn.commit()
        r = conn.execute(text(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'image_tags'"
        ))
        if r.fetchone() is None:
            conn.execute(text(
                "CREATE TABLE image_tags (image_id BIGINT NOT NULL, tag_id BIGINT NOT NULL, "
                "PRIMARY KEY (image_id, tag_id), "
                "FOREIGN KEY(image_id) REFERENCES images (id), FOREIGN KEY(tag_id) REFERENCES tags (id))"
            ))
            conn.commit()


def init_db() -> None:
    """创建数据库表"""
    SQLModel.metadata.create_all(sync_engine)
    _run_natural_sort_migration()
    _run_media_type_migration()
    _run_tags_migration()
    _run_fulltext_migration()
    _run_path_count_cache_migration()


async def get_async_session():
    async with async_session_factory() as session:
        yield session
