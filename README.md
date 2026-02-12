This is Semantic search api.
It retrieves the data from database. extracts the description from retrieved data converts to a polars/pandas dataframe.
the list of description is converted to embeddings and stored in FAISS vector store.
The query is extracted from the request and converted to embeddings, to use these embeddings to search in vector store.
vector store provides with matching data. the data transformed and returened back to user.
