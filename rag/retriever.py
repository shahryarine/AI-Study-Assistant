from typing import List, Dict, Any, Optional
from rag.embedder import Embedder
from rag.vector_store import VectorStore

class Retriever:
    """
    Coordinates semantic search by converting user queries into vector representations
    and retrieving the most relevant chunks from the VectorStore.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant knowledge chunks for a given query safely.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k threshold must be strictly positive.")
            
        if self.vector_store.count() == 0:
            return []

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Safely unpack ChromaDB results to prevent IndexErrors on empty returns
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        retrieved_chunks: List[Dict[str, Any]] = []

        for document, metadata, distance in zip(documents, metadatas, distances):
            retrieved_chunks.append({
                "text": document,
                "metadata": metadata,
                "distance": distance,
            })

        return retrieved_chunks