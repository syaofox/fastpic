from sqlmodel import Field, SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

DATABASE_URL = "sqlite:///./fastpic.db"
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./fastpic.db"


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
