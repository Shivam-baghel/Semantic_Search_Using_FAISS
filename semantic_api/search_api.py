from typing import List, Optional

import faiss
import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI

# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, Mapped, mapped_column
from helper import logger
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from models import Product, ProductDB, SearchRequest, db_dependency

app = FastAPI()

# Embedding setup
embedding_model = OllamaEmbeddings(model="embeddinggemma:latest")

# vector store will be initialize lazily
vector_store: Optional[FAISS] = None


def preprocess_text(text: str) -> str:
    """Preprocess text by lowering case and stripping whitespace."""
    return text.lower().strip()


def retrieve_data_from_db(db: db_dependency) -> List[ProductDB]:
    """Retrieve all products from the database.
    :param db: database session
    :type db: Session
    :return: list of ProductDB instances
    :rtype: List[ProductDB]"""

    products = db.query(ProductDB).all()
    logger.info(f"Retrieved {len(products)} products from the database.")
    return products


def products_to_df_via_pydantic(products: List[ProductDB]) -> pl.DataFrame:
    """Convert list of ProductDB to Polars DataFrame via Pydantic models.
    :param products: list of ProductDB instances
    :type products: List[ProductDB]
    :return: Polars DataFrame containing product data
    :rtype: pl.DataFrame"""
    pydantic_products = [Product.model_validate(prod).model_dump() for prod in products]
    df = pl.DataFrame(pydantic_products)
    logger.info("Converted products to Polars DataFrame.")
    return df


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using the embedding model.
    :param texts: list of text strings
    :type texts: List[str]
    :return: numpy array of embeddings
    :rtype: np.ndarray"""

    embeddings = embedding_model.embed_documents(texts)
    logger.info("Generated embeddings for texts.")
    return np.array(embeddings).astype("float32")


def initialize_vector_store(products_df: pl.DataFrame):
    """Initialize the FAISS vector store with product embeddings.
    :param products_df: Polars DataFrame containing product data
    :type products_df: pl.DataFrame
    :return: FAISS vector store initialized with product embeddings
    :rtype: FAISS"""

    # products_df = products_df.with_columns(pl.concat_str(["product_name", "category", "description"], separator=", ").alias("all"))

    df = products_df.with_columns(
        pl.col("description").apply(preprocess_text).alias("processed_description")
    )

    descriptions = df["processed_description"].to_list()
    embeddings = generate_embeddings(descriptions)

    documents = [
        Document(
            page_content=desc,
            metadata={"id": str(i), **df[i].to_dict()},
        )
        for i, desc in enumerate(descriptions)
    ]

    # create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # creating docstore
    docstore = InMemoryDocstore()
    index_to_docstore = {}
    for i, document in enumerate(documents):
        docstore.add({str(i): document})
        index_to_docstore[i] = str(i)

    # create FAISS vector store
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore=index_to_docstore,
    )
    logger.info("Initialized FAISS vector store with product embeddings.")

    return vector_store


def semantic_search(query: str, db: db_dependency, top_k: int) -> List[Product]:
    """
    Perform semantic search on products based on the query.

    :param query: search query string
    :type query: str
    :param top_k: number of top similar products to retrieve
    :type top_k: int
    :return: list of top_k similar products
    :rtype: List[Product]
    """
    global vector_store

    if vector_store is None or vector_store.index.ntotal == 0:
        products = retrieve_data_from_db(db)
        products_df = products_to_df_via_pydantic(products)

        vector_store_instance = initialize_vector_store(products_df)
        logger.info("Vector store initialized inside semantic_search.")

    query_embedding = generate_embeddings([query])[0]
    search_results = vector_store_instance.similarity_search_with_score_by_vector(
        query_embedding, k=top_k
    )

    results = []
    for doc, score in search_results:
        metadata = doc.metadata
        product = Product(
            id=int(metadata["id"]),
            name=metadata["name"],
            description=metadata["description"],
            category=metadata["category"],
            score=score,
        )
        results.append(product)
    logger.info(f"Retrieved {len(results)} similar products for the query.")
    return results


@app.get("/products/")
def read_products(
    request: Optional[SearchRequest], db: db_dependency = db_dependency
) -> List[Product]:
    """API endpoint to perform semantic search on products.
    :param request: SearchRequest object containing query and top_k
    :type request: SearchRequest
    :param db: database session
    :type db: Session"""
    results = semantic_search(request.query, db, request.top_k)
    return results


if __name__ == "__main__":
    uvicorn.run(
        "search_api:app", host="0.0.0.0", reload=True, port=8081, log_config=None
    )
