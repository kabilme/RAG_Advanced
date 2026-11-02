"""Database engine, session factory, and initialization."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.core.logging import logger
from app.database.models import Base

db_path = Path(settings.SQLITE_DB_PATH)
db_path.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initializes all database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Initialized SQLite relational database at %s", db_path)


def get_db():
    """FastAPI dependency yielding database sessions."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
