import os
import re

from sqlalchemy import event
from sqlmodel import Field, SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# 数据库存放在 data/ 目录下，支持通过环境变量覆盖
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("DATA_DIR", "data"))
os.makedirs(_DATA_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{_DATA_DIR}/fastpic.db"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{_DATA_DIR}/fastpic.db"


def _set_sqlite_pragma(dbapi_conn, connection_record):
    """启用 WAL 模式与 busy_timeout，提升并发读写性能"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


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


# 同步引擎用于 create_all
sync_engine = create_engine(DATABASE_URL, echo=False)
# 异步引擎：多连接池支持并行查询（count/subfolders/images 同时执行）
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)
event.listens_for(sync_engine, "connect")(_set_sqlite_pragma)
event.listens_for(async_engine.sync_engine, "connect")(_set_sqlite_pragma)
async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


def _run_natural_sort_migration() -> None:
    """为已有表添加 filename_natural / relative_path_natural 列并回填"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text("PRAGMA table_info(images)"))
        cols = {row[1] for row in r.fetchall()}
        if "filename_natural" not in cols:
            conn.execute(text("ALTER TABLE images ADD COLUMN filename_natural TEXT"))
            conn.execute(text("ALTER TABLE images ADD COLUMN relative_path_natural TEXT"))
            conn.commit()
        # 回填：对空值用 natural_sort_key 生成并更新
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
        # 创建索引以加速排序
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_images_filename_natural ON images(filename_natural)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_images_relative_path_natural ON images(relative_path_natural)"))
            conn.commit()
        except Exception:
            pass


def _run_media_type_migration() -> None:
    """为已有表添加 media_type 列并回填为 image"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text("PRAGMA table_info(images)"))
        cols = {row[1] for row in r.fetchall()}
        if "media_type" not in cols:
            conn.execute(text("ALTER TABLE images ADD COLUMN media_type TEXT DEFAULT 'image'"))
            conn.execute(text("UPDATE images SET media_type = 'image' WHERE media_type IS NULL"))
            conn.commit()
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_images_media_type ON images(media_type)"))
            conn.commit()
        except Exception:
            pass


def _run_tags_migration() -> None:
    """创建 tags 和 image_tags 表（若不存在）"""
    from sqlalchemy import text

    with sync_engine.connect() as conn:
        r = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'"))
        if r.fetchone() is None:
            conn.execute(text(
                "CREATE TABLE tags (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, name VARCHAR NOT NULL)"
            ))
            conn.execute(text("CREATE UNIQUE INDEX ix_tags_name ON tags (name)"))
            conn.commit()
        r = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='image_tags'"))
        if r.fetchone() is None:
            conn.execute(text(
                "CREATE TABLE image_tags (image_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, "
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


async def get_async_session():
    async with async_session_factory() as session:
        yield session
