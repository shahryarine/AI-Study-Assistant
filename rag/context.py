from typing import List, Dict, Any, Optional

class ContextBuilder:
    """
    Builds a structured context string and strictly enforces token/character budgets.
    """

    def __init__(self, max_chunks: int = 5, max_characters: int = 4000):
        if max_chunks <= 0 or max_characters <= 0:
            raise ValueError("Constraints must be greater than zero.")

        self.max_chunks = max_chunks
        self.max_characters = max_characters

    def build(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return ""

        context_parts: List[str] = []
        current_char_count = 0
        selected_chunks = retrieved_chunks[:self.max_chunks]

        for index, chunk in enumerate(selected_chunks, start=1):
            text = (chunk.get("text") or chunk.get("content") or "").strip()
            if not text:
                continue

            metadata = chunk.get("metadata", {})
            meta_lines: List[str] = [f"[Source {index}]"]
            
            # Map available metadata
            for key, label in [("filename", "File"), ("page", "Page"), ("score", "Relevance")]:
                val = metadata.get(key) or chunk.get(key)
                if val is not None:
                    # Format float scores to 4 decimal places
                    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                    meta_lines.append(f"{label}: {val_str}")

            header = "\n".join(meta_lines)
            chunk_block = f"{header}\n\n{text}"
            separator = "\n\n---\n\n" if context_parts else ""
            
            projected_size = current_char_count + len(chunk_block) + len(separator)

            # If the block exceeds the budget, dynamically truncate it
            if projected_size > self.max_characters:
                allowed_text_len = self.max_characters - current_char_count - len(header) - len(separator) - 10
                if allowed_text_len > 50:
                    truncated_text = text[:allowed_text_len] + " ... [TRUNCATED]"
                    chunk_block = f"{header}\n\n{truncated_text}"
                    context_parts.append(separator + chunk_block)
                break # Reached budget limit, stop processing further chunks

            context_parts.append(separator + chunk_block)
            current_char_count = projected_size

        return "".join(context_parts)