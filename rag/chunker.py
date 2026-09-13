import hashlib
from typing import List, Dict, Any, Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Splits text into chunks using Regex separators to intelligently handle
    abbreviations and sentence boundaries.
    """
    
    DEFAULT_SEPARATORS = [
        r"\n\n", 
        r"\n",
        r"(?<=[.?!؟؛])\s+",  # Matches spaces after punctuation marks
        r"\s+",              # Fallback to general whitespace
        r""
    ]

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Invalid chunk configuration.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.DEFAULT_SEPARATORS,
            is_separator_regex=True,  # Activated Regex parsing for smarter splits
            keep_separator=False,     # Clean up separators from the final chunk text
            length_function=len,
        )

    def _generate_chunk_id(self, text: str, source_identifier: str) -> str:
        """
        Generates a robust, order-independent hash based purely on the chunk's
        content and source origin.
        """
        unique_string = f"{source_identifier}_{text}"
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:16]

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text.strip())
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for page_data in pages:
            content = page_data.get("content", "")
            if not content or not content.strip():
                continue

            metadata = dict(page_data.get("metadata", {}))
            source_id = metadata.get("filename", "unknown_source")
            page_chunks = self.split_text(content)

            for chunk_index, chunk_text in enumerate(page_chunks):
                yield {
                    "chunk_id": self._generate_chunk_id(chunk_text, source_id),
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_size": len(chunk_text),
                        "chunk_index": chunk_index
                    }
                }