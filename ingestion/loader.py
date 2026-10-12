import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Regex pattern to match custom pagination markers in Markdown/TXT files
PAGE_MARKER_PATTERN = re.compile(
    r"---\s*\[PDF Page (\d+)\s*\|\s*Book Page (\d+)\]\s*---"
)

class MarkdownLoader:
    """
    Loads and parses marked Markdown or TXT files.
    Extracts content and assigns proper pagination metadata for the RAG system.
    Integrates robust file validation and error handling.
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def validate_file(self) -> None:
        """Validates the existence and format of the input file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"The provided path is not a valid file: {self.file_path}")
        if self.file_path.suffix.lower() not in [".txt", ".md"]:
            raise ValueError(f"Expected a .txt or .md file, but received: {self.file_path.suffix}")

    def load(self) -> List[Dict[str, Any]]:
        """
        Reads the file, extracts pages based on regex markers, and captures preamble.
        
        Returns:
            List of dictionaries containing 'content' and 'metadata'.
        """
        self.validate_file()
        
        try:
            text = self.file_path.read_text(encoding="utf-8")
        except Exception as error:
            logger.error(f"Critical error reading file '{self.file_path.name}': {error}")
            raise RuntimeError(f"Failed to read file: {error}") from error

        if not text.strip():
            raise ValueError(f"The input file '{self.file_path.name}' is empty.")

        matches = list(PAGE_MARKER_PATTERN.finditer(text))
        pages: List[Dict[str, Any]] = []

        # Handle documents without structured markers
        if not matches:
            logger.warning(f"No page markers found in '{self.file_path.name}'. Treating entire file as one block.")
            return [{
                "content": text.strip(),
                "metadata": {"filename": self.file_path.name, "pdf_page": 0, "book_page": 0}
            }]

        # Capture preamble text appearing before the first marker
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                pages.append({
                    "content": preamble,
                    "metadata": {"filename": self.file_path.name, "pdf_page": 0, "book_page": 0}
                })

        # Process each marked segment as a distinct logical page
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
                        "filename": self.file_path.name,
                        "pdf_page": pdf_page,
                        "book_page": book_page,
                    },
                })

        logger.info(f"Successfully loaded {len(pages)} logical pages from {self.file_path.name}")
        return pages