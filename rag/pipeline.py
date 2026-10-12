import logging
from typing import Optional

from ingestion.loader import MarkdownLoader
from ingestion.cleaner import clean_pages
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.utils import track_time

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Coordinates the end-to-end data ingestion workflow.
    """
    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def process_file(self, file_path: str, batch_size: int = 100) -> int:
        logger.info(f"Starting ingestion pipeline for document: {file_path}")
        
        with track_time("Loading Markdown/TXT File", logger):
            loader = MarkdownLoader(file_path)
            raw_pages = loader.load()
        
        with track_time(f"Cleaning {len(raw_pages)} pages", logger):
            cleaned_pages = clean_pages(raw_pages)

        with track_time("Initializing Chunker generator", logger):
            chunk_generator = self.chunker.chunk_pages(cleaned_pages)
        
        total_indexed = 0
        current_batch = []

        def _process_batch(batch, batch_number):
            if not batch:
                return
            with track_time(f"Embedding and Indexing Batch {batch_number} ({len(batch)} chunks)", logger):
                texts_to_embed = [chunk["content"] for chunk in batch]
                embeddings = self.embedder.embed_texts(texts_to_embed)
                self.vector_store.add_documents(documents=batch, embeddings=embeddings)

        batch_counter = 1
        for chunk in chunk_generator:
            current_batch.append(chunk)
            if len(current_batch) >= batch_size:
                _process_batch(current_batch, batch_counter)
                total_indexed += len(current_batch)
                current_batch.clear()
                batch_counter += 1

        if current_batch:
            _process_batch(current_batch, batch_counter)
            total_indexed += len(current_batch)

        return total_indexed