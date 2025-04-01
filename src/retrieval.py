class Retrieval:

    def __init__(self):
        pass

    def retrieval_one_query(self, db, query: str, top_k: int = 5) -> list:
        """
        Perform a similarity search in the database using the provided query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to return.

        Returns:
            list: A list of the top-k chunks retrieved from the database.
            list: A list of the chunk IDs corresponding to the retrieved chunks.
        """
        docs = db.similarity_search(query=query, k=top_k)
        chunks, chunks_id = [], []
        for x in docs:
            chunks.append(x.page_content.strip())
            chunks_id.append(x.metadata['chunk_id'])
        return chunks, chunks_id
    
    def retrieval_multiple_queries(self, db, queries: list, top_k: int = 5) -> list:
        """
        Perform a similarity search in the database using multiple queries.

        Args:
            queries (list): A list of query strings to search for.
            top_k (int): The number of top results to return for each query.

        Returns:
            list
        """
        mult_chunks, mult_chunks_id = [], []
        for query in queries:
            chunks, chunks_id = self.retrieval_one_query(db=db, query=query, top_k=top_k)
            mult_chunks.append(chunks); mult_chunks_id.append(chunks_id)
        return mult_chunks, mult_chunks_id