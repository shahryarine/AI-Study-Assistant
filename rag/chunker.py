from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Split cleaned page texts into smaller overlapping chunks.

    Strategy:
    - Paragraphs are prioritized first.
    - Then sentences.
    - Then words.
    - Finally characters if necessary.

    Designed for Persian and English texts.
    """

    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 100
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size:
                Maximum number of characters in each chunk.
                Default: 700

            chunk_overlap:
                Number of overlapping characters between
                neighboring chunks.
                Default: 100
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Order is important:
        # paragraph -> sentence -> word -> character
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",   # Paragraph
                "\n",     # Line
                "؟ ",     # Persian question
                "? ",     # English question
                "! ",     # Exclamation
                "؛ ",     # Persian semicolon
                "; ",     # English semicolon
                "، ",     # Persian comma
                ", ",     # English comma
                ". ",     # English sentence
                " ",      # Word
                ""        # Character
            ],
            keep_separator=True,
            length_function=len,
        )

    # ---------------------------------------------------------
    # Split a single text
    # ---------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.

        Args:
            text:
                Cleaned text.

        Returns:
            List of chunk strings.
        """

        if not text or not text.strip():
            return []

        text = text.strip()

        chunks = self.splitter.split_text(text)

        return [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

    # ---------------------------------------------------------
    # Chunk pages
    # ---------------------------------------------------------

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split cleaned pages into chunks while preserving metadata.

        Input example:

        {
            "content": "...",
            "metadata": {
                "filename": "book.pdf",
                "page": 10,
                "total_pages": 100
            }
        }

        Output example:

        {
            "chunk_id": "chunk_000001",
            "content": "...",
            "metadata": {
                "filename": "book.pdf",
                "page": 10,
                "total_pages": 100,
                "chunk_size": 650,
                "chunk_index": 0
            }
        }
        """

        all_chunks = []

        chunk_counter = 0

        for page_data in pages:

            # -------------------------------------------------
            # Read page content
            # -------------------------------------------------

            content = page_data.get("content", "")

            if not content or not content.strip():
                continue

            # -------------------------------------------------
            # Copy original metadata
            # -------------------------------------------------

            metadata = page_data.get(
                "metadata",
                {}
            ).copy()

            # -------------------------------------------------
            # Split page
            # -------------------------------------------------

            page_chunks = self.split_text(content)

            # -------------------------------------------------
            # Create chunk objects
            # -------------------------------------------------

            for page_chunk_index, chunk_text in enumerate(
                page_chunks
            ):

                chunk_counter += 1

                chunk = {
                    "chunk_id": (
                        f"chunk_{chunk_counter:06d}"
                    ),

                    "content": chunk_text,

                    "metadata": {
                        **metadata,

                        "chunk_size": len(
                            chunk_text
                        ),

                        "chunk_index": (
                            page_chunk_index
                        )
                    }
                }

                all_chunks.append(chunk)

        return all_chunks


# ============================================================
# Convenience function
# ============================================================

def create_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Convenience function for creating chunks.

    Args:
        pages:
            Cleaned pages.

        chunk_size:
            Maximum chunk size.

        chunk_overlap:
            Overlap between neighboring chunks.

    Returns:
        List of chunk dictionaries.
    """

    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return chunker.chunk_pages(pages)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    sample_pages = [
        {
            "content": """
            یادگیری ماشین یکی از شاخه‌های مهم هوش مصنوعی است.
            این حوزه به سیستم‌ها کمک می‌کند تا از داده‌ها
            الگوهای مختلف را یاد بگیرند و بر اساس آنها تصمیم‌گیری کنند.

            در یادگیری ماشین، داده‌های آموزشی نقش بسیار مهمی دارند.
            کیفیت داده‌ها می‌تواند مستقیماً بر عملکرد مدل تأثیر بگذارد.
            بنابراین، پیش‌پردازش و پاک‌سازی داده‌ها اهمیت زیادی دارد.

            Machine learning is a branch of artificial intelligence.
            It allows computer systems to learn patterns from data
            and make predictions based on those patterns.
            """,

            "metadata": {
                "filename": "persian_lecture.pdf",
                "page": 5,
                "total_pages": 20
            }
        }
    ]

    chunks = create_chunks(
        sample_pages,
        chunk_size=200,
        chunk_overlap=50
    )

    print("=" * 70)
    print(f"Created {len(chunks)} chunks")
    print("=" * 70)

    for chunk in chunks:

        print(f"\nID: {chunk['chunk_id']}")

        print(
            f"Page: {chunk['metadata']['page']}"
        )

        print(
            f"Size: {chunk['metadata']['chunk_size']} characters"
        )

        print(
            f"Content:\n{chunk['content']}"
        )

        print("-" * 70)