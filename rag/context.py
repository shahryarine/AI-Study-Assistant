class ContextBuilder:
    """
    Builds a structured context string from retrieved chunks.
    """

    def __init__(self, max_chunks: int = 5):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero.")

        self.max_chunks = max_chunks

    def build(self, retrieved_chunks: list[dict]) -> str:
        """
        Convert retrieved chunks into a formatted context string.
        """

        if not retrieved_chunks:
            return ""

        selected_chunks = retrieved_chunks[:self.max_chunks]

        context_parts = []

        for index, chunk in enumerate(selected_chunks, start=1):
            text = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})

            if not text:
                continue

            filename = metadata.get("filename", "Unknown")
            pdf_page = metadata.get("pdf_page", "Unknown")
            book_page = metadata.get("book_page", "Unknown")
            chunk_index = metadata.get("chunk_index", "Unknown")

            context_parts.append(
                f"[Source {index}]\n"
                f"File: {filename}\n"
                f"PDF Page: {pdf_page}\n"
                f"Book Page: {book_page}\n"
                f"Chunk: {chunk_index}\n\n"
                f"{text}"
            )

        return "\n\n".join(context_parts)