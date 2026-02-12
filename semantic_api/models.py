from typing import Annotated, Optional

from fastapi import Depends
from helper import DATABASES_URL
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=create_engine(DATABASES_URL)
)

Base = DeclarativeBase()


# Database thread to open and close sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


class ProductDB(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    category: Mapped[str] = mapped_column(nullable=False)


class Product(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    category: str
    score: Optional[float] = None  # Similarity score


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
