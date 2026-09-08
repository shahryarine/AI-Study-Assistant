import logging
from pathlib import Path
from typing import List, Dict, Any
import pymupdf

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PDFLoader:
    """
    Load a PDF file page by page and extract its text.
    Each page is returned as a dictionary containing content and metadata.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def validate_file(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"The provided path is not a valid file: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, but received: {self.file_path.suffix}")

    def load(self) -> List[Dict[str, Any]]:
        self.validate_file()
        pages = []

        try:
            document = pymupdf.open(self.file_path)
            total_pages = len(document)

            for page_number, page in enumerate(document, start=1):
                try:
                    text = page.get_text("text").strip()

                    if not text:
                        logger.warning(f"Page {page_number} in '{self.file_path.name}' contains no extractable text.")
                        continue

                    page_data = {
                        "content": text,
                        "metadata": {
                            "filename": self.file_path.name,
                            "page": page_number,
                            "total_pages": total_pages,
                        },
                    }
                    pages.append(page_data)

                except Exception as page_error:
                    logger.error(f"Could not extract text from page {page_number}: {page_error}")

            document.close()

        except Exception as error:
            logger.error(f"Critical error loading PDF '{self.file_path.name}': {error}")
            raise RuntimeError(f"Failed to process PDF: {error}") from error

        return pages

if __name__ == "__main__":
    # Manual execution test with dummy path
    sample_pdf_path = "data/sample/test_document.pdf"
    
    try:
        loader = PDFLoader(sample_pdf_path)
        extracted_pages = loader.load()
        logger.info(f"Successfully loaded {len(extracted_pages)} pages.")
    except Exception as e:
        logger.error(f"Test failed: {e}")