import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import dotenv

dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

engine = None
async_session = None

def connect():
    global engine, async_session
    print("Attempting to connect to the database...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    print(f'engine: {engine}')
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    print("Database connection successful.")

async def disconnect():
    if engine:
        await engine.dispose()

async def get_db_session():
    if async_session is None:
        raise IOError("Database not connected")
    async with async_session() as session:
        yield session

def get_session_factory():
    return async_session
