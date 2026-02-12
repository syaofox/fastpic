import os

from sqlmodel import Field, SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# 数据库存放在 data/ 目录下，支持通过环境变量覆盖
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("DATA_DIR", "data"))
os.makedirs(_DATA_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{_DATA_DIR}/fastpic.db"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{_DATA_DIR}/fastpic.db"


class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    relative_path: str = Field(unique=True, index=True)
    modified_at: float = Field(index=True)
    file_size: int = Field(default=0, index=True)
    width: int = 0
    height: int = 0


# 同步引擎用于 create_all
sync_engine = create_engine(DATABASE_URL, echo=False)
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


def init_db() -> None:
    """创建数据库表"""
    SQLModel.metadata.create_all(sync_engine)


async def get_async_session():
    async with async_session_factory() as session:
        yield session
