from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Split cleaned texts into smaller overlapping chunks.
    Prioritizes paragraphs, then sentences, words, and characters.
    """
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Invalid chunk size or overlap configuration.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",   # Paragraph boundary
                "\n",     # Line boundary
                "؟ ",     # Persian/Arabic question mark
                "? ",     # English question mark
                "! ",     # Exclamation mark
                "؛ ",     # Persian/Arabic semicolon
                "; ",     # English semicolon
                "، ",     # Persian/Arabic comma
                ", ",     # English comma
                ". ",     # English sentence boundary
                " ",      # Word boundary
                ""        # Character level fallback
            ],
            keep_separator=True,
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text.strip())
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        chunk_counter = 0

        for page_data in pages:
            content = page_data.get("content", "")
            if not content or not content.strip():
                continue

            metadata = page_data.get("metadata", {})
            page_chunks = self.split_text(content)

            for page_chunk_index, chunk_text in enumerate(page_chunks):
                chunk_counter += 1
                chunk = {
                    "chunk_id": f"chunk_{chunk_counter:06d}",
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_size": len(chunk_text),
                        "chunk_index": page_chunk_index
                    }
                }
                all_chunks.append(chunk)

        return all_chunks

def create_chunks(pages: List[Dict[str, Any]], chunk_size: int = 700, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages)

if __name__ == "__main__":
    # Test execution
    dummy_pages = [{
        "content": "Machine learning is fascinating. It allows systems to learn from data.\n\nHowever, data preprocessing is essential.",
        "metadata": {"filename": "test_doc.pdf", "page": 1, "total_pages": 1}
    }]
    
    chunks = create_chunks(dummy_pages, chunk_size=50, chunk_overlap=10)
    for c in chunks:
        print(f"ID: {c['chunk_id']} | Size: {c['metadata']['chunk_size']} | Content: {c['content']}")