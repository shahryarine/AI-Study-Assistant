import re
import unicodedata
from typing import List, Dict, Any


class TextCleaner:
    """
    Clean and normalize extracted PDF text.

    The cleaner focuses on:
    - whitespace normalization
    - newline normalization
    - removal of invalid/control characters
    - Persian character normalization
    - basic punctuation normalization
    - preservation of scientific content
    """

    def clean(self, text: str) -> str:
        """
        Apply all cleaning and normalization steps to a text.

        Args:
            text: Raw text extracted from a PDF.

        Returns:
            Cleaned and normalized text.
        """

        if not text:
            return ""

        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.normalize_persian_characters(text)
        text = self.normalize_spaces(text)
        text = self.fix_line_breaks(text)
        text = self.normalize_blank_lines(text)
        text = text.strip()

        return text

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters using NFC.

        NFC combines compatible Unicode sequences into
        their standard composed representation.
        """

        return unicodedata.normalize("NFC", text)

    def remove_control_characters(self, text: str) -> str:
        """
        Remove unwanted control characters while preserving
        useful whitespace such as spaces and newlines.
        """

        cleaned_chars = []

        for char in text:
            category = unicodedata.category(char)

            if category.startswith("C"):
                if char in ("\n", "\t", "\r"):
                    cleaned_chars.append(char)
                else:
                    continue
            else:
                cleaned_chars.append(char)

        return "".join(cleaned_chars)

    def normalize_persian_characters(self, text: str) -> str:
        """
        Normalize common Arabic/Persian character variants.

        Arabic:
            ي -> ی
            ى -> ی
            ك -> ک

        Persian:
            ة -> ه

        Zero-width characters are handled separately.
        """

        replacements = {
            "ي": "ی",
            "ى": "ی",
            "ك": "ک",
            "ة": "ه",
            "ۀ": "ه",
        }

        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)

        # Normalize zero-width non-joiner.
        # Persian half-space is U+200C.
        text = text.replace("\u200c", "\u200c")

        return text

    def normalize_spaces(self, text: str) -> str:
        """
        Normalize spaces and tabs without destroying newlines.
        """

        # Replace tabs with a normal space.
        text = text.replace("\t", " ")

        # Normalize non-breaking spaces.
        text = text.replace("\u00a0", " ")

        # Collapse multiple spaces.
        text = re.sub(r"[ ]{2,}", " ", text)

        # Remove spaces at the beginning/end of lines.
        text = re.sub(r"[ ]+\n", "\n", text)
        text = re.sub(r"\n[ ]+", "\n", text)

        return text

    def fix_line_breaks(self, text: str) -> str:
        """
        Repair common line breaks introduced by PDF extraction.

        Examples:

            "multi-\nvariate" -> "multivariate"

            "statistical\nanalysis" -> "statistical analysis"

        Existing paragraph boundaries are preserved as much as possible.
        """

        # Fix words broken by a hyphen at the end of a line.
        # Example:
        # multi-
        # variate
        #
        # This is mainly useful for English PDF extraction.
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

        # Replace a single newline between two pieces of text
        # with a space.
        #
        # Double newlines are preserved as paragraph boundaries.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        return text

    def normalize_blank_lines(self, text: str) -> str:
        """
        Prevent excessive empty lines.

        Maximum: two consecutive newline characters.
        """

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text


def clean_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean the content of each page while preserving metadata.

    Args:
        pages:
            List of page dictionaries produced by PDFLoader.

    Returns:
        List of cleaned page dictionaries.
    """

    cleaner = TextCleaner()
    cleaned_pages = []

    for page_data in pages:
        cleaned_content = cleaner.clean(page_data.get("content", ""))

        cleaned_page = {
            "content": cleaned_content,
            "metadata": page_data.get("metadata", {}).copy(),
        }

        cleaned_pages.append(cleaned_page)

    return cleaned_pages


if __name__ == "__main__":
    # Simple manual test
    sample_text = """
    سلام    دنیا

    این   یک متن آزمایشی است.
    این متن دارای
    شکست خطی نامناسب است.

    كلمه عربي
    يک کلمه با ي و ك
    """

    cleaner = TextCleaner()

    print("========== ORIGINAL ==========")
    print(sample_text)

    cleaned_text = cleaner.clean(sample_text)

    print("\n========== CLEANED ==========")
    print(cleaned_text)