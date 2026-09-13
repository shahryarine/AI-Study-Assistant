import hashlib
from typing import List, Dict, Any, Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Splits cleaned texts into smaller overlapping chunks for vector embedding.
    Utilizes a recursive character splitting strategy prioritizing natural language boundaries.
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",   # Paragraph boundary
        "\n",     # Line boundary
        "؟ ",     # Persian/Arabic question mark
        "? ",     # English question mark
        "! ",     # Exclamation mark
        "؛ ",     # Persian/Arabic semicolon
        "; ",     # English semicolon
        "، ",     # Persian/Arabic comma
        ", ",     # English comma
        ". ",     # Sentence boundary
        " ",      # Word boundary
        ""        # Character level fallback
    ]

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        """
        Initialize the TextChunker with specific size and overlap constraints.
        """
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Invalid chunk configuration: overlap must be non-negative and strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.DEFAULT_SEPARATORS,
            keep_separator=True,
            length_function=len,
        )

    def _generate_chunk_id(self, text: str, page_index: int, chunk_index: int) -> str:
        """
        Generate a deterministic, unique hash for a chunk to prevent duplicate vector DB insertions.
        """
        unique_string = f"{text}_{page_index}_{chunk_index}"
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:16]

    def split_text(self, text: str) -> List[str]:
        """
        Split a single text string into multiple chunk strings.
        """
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text.strip())
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        Generate chunk dictionaries from a list of page data dictionaries.
        Yields chunks lazily to optimize memory usage for large documents.
        """
        for page_index, page_data in enumerate(pages):
            content = page_data.get("content", "")
            if not content or not content.strip():
                continue

            # Ensure we do not mutate the original dictionary reference
            metadata = dict(page_data.get("metadata", {}))
            page_chunks = self.split_text(content)

            for chunk_index, chunk_text in enumerate(page_chunks):
                yield {
                    "chunk_id": self._generate_chunk_id(chunk_text, page_index, chunk_index),
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_size": len(chunk_text),
                        "chunk_index": chunk_index
                    }
                }

def create_chunks(pages: List[Dict[str, Any]], chunk_size: int = 700, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Helper function to initialize the chunker and process pages eagerly into a list.
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return list(chunker.chunk_pages(pages))

if __name__ == "__main__":
    # Test execution
    dummy_pages = [{
        "content": "Machine learning is fascinating. It allows systems to learn from data.\n\nHowever, data preprocessing is essential.",
        "metadata": {"filename": "test_doc.pdf", "page": 1, "total_pages": 1}
    }]
    
    chunks = create_chunks(dummy_pages, chunk_size=50, chunk_overlap=10)
    for c in chunks:
        print(f"ID: {c['chunk_id']} | Size: {c['metadata']['chunk_size']} | Content: {c['content']}")