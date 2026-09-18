import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore

# Configure logging for the indexing pipeline
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PAGE_MARKER_PATTERN = re.compile(
    r"---\s*\[PDF Page (\d+)\s*\|\s*Book Page (\d+)\]\s*---"
)

class Indexer:
    """
    Coordinates the ingestion pipeline: TXT -> Pages -> Chunks -> Embeddings -> VectorStore.
    Designed to process large documents in memory-efficient batches.
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

    def load_text_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a marked TXT file into a list of page dictionaries.
        Captures any preamble text occurring before the first page marker.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"The input file '{path.name}' is empty.")

        matches = list(PAGE_MARKER_PATTERN.finditer(text))
        pages: List[Dict[str, Any]] = []

        if not matches:
            logger.warning(f"No page markers found in '{path.name}'. Treating entire file as one page.")
            return [{
                "content": text.strip(),
                "metadata": {"filename": path.name, "pdf_page": 0, "book_page": 0}
            }]

        # Capture preamble text before the first marker (if any exists)
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                pages.append({
                    "content": preamble,
                    "metadata": {"filename": path.name, "pdf_page": 0, "book_page": 0}
                })

        for index, match in enumerate(matches):
            pdf_page = int(match.group(1))
            book_page = int(match.group(2))
            start_pos = match.end()
            
            end_pos = matches[index + 1].start() if (index + 1 < len(matches)) else len(text)
            content = text[start_pos:end_pos].strip()

            if content:
                pages.append({
                    "content": content,
                    "metadata": {
                        "filename": path.name,
                        "pdf_page": pdf_page,
                        "book_page": book_page,
                    },
                })

        return pages

    def index_text_file(self, file_path: str, batch_size: int = 100) -> int:
        """
        Processes a document in batches to maintain a low memory footprint.
        Yields chunks from the Chunker, embeds them, and upserts to ChromaDB.
        """
        logger.info(f"Starting ingestion pipeline for: {file_path}")
        
        pages = self.load_text_pages(file_path)
        logger.info(f"Extracted {len(pages)} distinct pages.")

        # TextChunker.chunk_pages() is a generator (from Phase 1 optimization)
        chunk_generator = self.chunker.chunk_pages(pages)
        
        total_indexed = 0
        current_batch_chunks = []

        def _process_batch(batch: List[Dict[str, Any]]):
            if not batch:
                return
            texts_to_embed = [c["content"] for c in batch]
            embeddings = self.embedder.embed_texts(texts_to_embed)
            
            # Assuming VectorStore.add_documents expects the chunk dicts and their embeddings
            self.vector_store.add_documents(documents=batch, embeddings=embeddings)
            logger.info(f"Successfully indexed batch of {len(batch)} chunks.")

        for chunk in chunk_generator:
            current_batch_chunks.append(chunk)
            
            if len(current_batch_chunks) >= batch_size:
                _process_batch(current_batch_chunks)
                total_indexed += len(current_batch_chunks)
                current_batch_chunks.clear()

        # Process any remaining chunks in the final partial batch
        if current_batch_chunks:
            _process_batch(current_batch_chunks)
            total_indexed += len(current_batch_chunks)

        logger.info(f"Ingestion complete. Total chunks indexed: {total_indexed}")
        return total_indexed