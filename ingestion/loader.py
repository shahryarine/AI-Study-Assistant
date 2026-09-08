from pathlib import Path
from typing import List, Dict, Any
import pymupdf 
from .cleaner import clean_pages

class PDFLoader:
    """
    Load a PDF file page by page and extract its text.

    Each page is returned as a dictionary containing:
    - content: extracted text
    - metadata:
        - filename
        - page
        - total_pages
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def validate_file(self) -> None:
        """
        Validate that the input path exists and points to a PDF file.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"The provided path is not a file: {self.file_path}"
            )

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, but received: {self.file_path.suffix}"
            )

    def load(self) -> List[Dict[str, Any]]:
        """
        Extract text from the PDF page by page.

        Returns:
            A list of dictionaries, one dictionary per page.
        """

        self.validate_file()

        pages = []

        try:
            document = pymupdf.open(self.file_path)

            total_pages = len(document)

            for page_number, page in enumerate(document, start=1):

                try:
                    text = page.get_text("text")

                    # Remove leading/trailing whitespace
                    text = text.strip()

                    if not text:
                        print(
                            f"Warning: Page {page_number} "
                            f"contains no extractable text."
                        )

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
                    print(
                        f"Warning: Could not extract page "
                        f"{page_number}: {page_error}"
                    )

            document.close()

        except Exception as error:
            raise RuntimeError(
                f"Could not load PDF file '{self.file_path.name}': {error}"
            ) from error

        return pages


if __name__ == "__main__":
    """
    Simple manual test.

    Replace the path below with the path to your PDF.
    """

    pdf_path = "data/sample/روش تحقيق و مشاوره آماري-2.pdf"

    try:
        loader = PDFLoader(pdf_path)
        pages = loader.load()
        cleaned_pages = clean_pages(pages)
        print("\nFirst page data:")
        print(pages[0])

        print("\nSecond page data:")
        print(pages[1])

        print(f"Successfully loaded {len(pages)} pages.")

        for page in pages[:2]:
            print("\n" + "=" * 50)
            print(f"Page: {page['metadata']['page']}")
            print("=" * 50)
            print(page["content"][:1000])

    except Exception as error:
        print(f"Error: {error}")