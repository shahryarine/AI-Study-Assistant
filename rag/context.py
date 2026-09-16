from typing import List, Dict, Any, Optional


class ContextBuilder:
    """
    Builds a structured context string from retrieved chunks.

    Retrieved chunks are expected to contain:
    - text/content
    - metadata
    - distance (optional)

    Lower cosine distance means a more relevant result.
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_characters: int = 4000,
        max_distance: Optional[float] = None,
    ):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero.")

        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero.")

        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        self.max_chunks = max_chunks
        self.max_characters = max_characters
        self.max_distance = max_distance

    def _get_distance(self, chunk: Dict[str, Any]) -> Optional[float]:
        """
        Extract distance from the chunk or its metadata.
        """
        metadata = chunk.get("metadata") or {}

        distance = chunk.get("distance")
        if distance is None:
            distance = metadata.get("distance")

        if distance is None:
            return None

        try:
            return float(distance)
        except (TypeError, ValueError):
            return None

    def _passes_threshold(self, chunk: Dict[str, Any]) -> bool:
        """
        Keep chunks whose cosine distance is within the configured threshold.

        Lower distance means higher relevance.
        Chunks without a distance are kept.
        """
        if self.max_distance is None:
            return True

        distance = self._get_distance(chunk)

        if distance is None:
            return True

        return distance <= self.max_distance

    def _sort_by_distance(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Sort chunks from most relevant to least relevant.

        Chunks without a valid distance are placed last.
        """
        return sorted(
            chunks,
            key=lambda chunk: (
                self._get_distance(chunk) is None,
                self._get_distance(chunk)
                if self._get_distance(chunk) is not None
                else float("inf"),
            ),
        )

    def build(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a structured context string while enforcing:
        - relevance threshold
        - relevance ordering
        - maximum number of chunks
        - maximum character budget
        - safe truncation
        """
        if not retrieved_chunks:
            return ""

        # 1. Filter weak results.
        filtered_chunks = [
            chunk
            for chunk in retrieved_chunks
            if self._passes_threshold(chunk)
        ]

        if not filtered_chunks:
            return ""

        # 2. Sort by cosine distance: lower distance = better result.
        sorted_chunks = self._sort_by_distance(filtered_chunks)

        # 3. Keep only the top N relevant chunks.
        selected_chunks = sorted_chunks[:self.max_chunks]

        context_parts: List[str] = []
        current_char_count = 0

        for index, chunk in enumerate(selected_chunks, start=1):
            text = (
                chunk.get("text")
                or chunk.get("content")
                or ""
            ).strip()

            if not text:
                continue

            metadata = chunk.get("metadata") or {}

            filename = metadata.get("filename")
            if filename is None:
                filename = chunk.get("filename")

            page = metadata.get("page")
            if page is None:
                page = chunk.get("page")

            distance = self._get_distance(chunk)

            meta_lines = [f"[Source {index}]"]

            if filename is not None:
                meta_lines.append(f"File: {filename}")

            if page is not None:
                meta_lines.append(f"Page: {page}")

            if distance is not None:
                meta_lines.append(f"Distance: {distance:.4f}")

            header = "\n".join(meta_lines)

            separator = "\n\n---\n\n" if context_parts else ""
            chunk_block = f"{header}\n\n{text}"

            projected_size = (
                current_char_count
                + len(separator)
                + len(chunk_block)
            )

            # 4. If the complete chunk exceeds the budget,
            #    add a truncated version if enough space remains.
            if projected_size > self.max_characters:
                remaining = (
                    self.max_characters
                    - current_char_count
                    - len(separator)
                    - len(header)
                    - 2
                )

                truncation_marker = " ... [TRUNCATED]"

                if remaining > len(truncation_marker):
                    allowed_text_len = remaining - len(truncation_marker)

                    truncated_text = (
                        text[:allowed_text_len]
                        + truncation_marker
                    )

                    truncated_block = (
                        f"{header}\n\n{truncated_text}"
                    )

                    context_parts.append(
                        separator + truncated_block
                    )

                break

            context_parts.append(separator + chunk_block)
            current_char_count = projected_size

        return "".join(context_parts)