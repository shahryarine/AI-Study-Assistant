import re
from pathlib import Path

from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore


PAGE_MARKER_PATTERN = re.compile(
    r"---\s*\[PDF Page (\d+) \| Book Page (\d+)\]\s*---"
)


class Indexer:
    """
    Index a text study document into ChromaDB.

    Pipeline:
    TXT -> pages -> chunks -> embeddings -> ChromaDB
    """

    def __init__(
        self,
        chunker: TextChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def load_text_pages(self, file_path: str) -> list[dict]:
        """
        Parse a TXT file containing PDF/Book page markers.

        Returns:
            List of page dictionaries compatible with TextChunker.chunk_pages().
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError("The input file is empty.")

        matches = list(PAGE_MARKER_PATTERN.finditer(text))

        if not matches:
            raise ValueError(
                "No page markers found in the input file."
            )

        pages = []

        for index, match in enumerate(matches):
            pdf_page = int(match.group(1))
            book_page = int(match.group(2))

            start = match.end()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(text)

            content = text[start:end].strip()

            if not content:
                continue

            pages.append(
                {
                    "content": content,
                    "metadata": {
                        "filename": path.name,
                        "pdf_page": pdf_page,
                        "book_page": book_page,
                    },
                }
            )

        return pages

    def index_text_file(self, file_path: str) -> int:
        """
        Parse, chunk, embed, and store a TXT study document.

        Returns:
            Number of indexed chunks.
        """

        pages = self.load_text_pages(file_path)

        chunks = self.chunker.chunk_pages(pages)

        if not chunks:
            return 0

        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.embedder.embed_texts(texts)

        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
        )

        return len(chunks)