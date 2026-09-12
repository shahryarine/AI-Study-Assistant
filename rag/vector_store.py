import chromadb


DEFAULT_PERSIST_DIRECTORY = "data/chroma_db"
DEFAULT_COLLECTION_NAME = "study_documents"


class VectorStore:
    """
    Persistent ChromaDB vector store for study document chunks.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
        )

    def add_documents(
        self,
        documents: list,
        embeddings: list[list[float]],
    ) -> None:
        """
        Add documents and their embeddings to ChromaDB.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        if not documents:
            return

        current_count = self.collection.count()

        ids = [
            f"chunk_{current_count + i}"
            for i in range(len(documents))
        ]

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """
        Search for the most relevant document chunks.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()