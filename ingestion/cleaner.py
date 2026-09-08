import re
import unicodedata
from typing import List, Dict, Any

class TextCleaner:
    """
    Clean and normalize extracted text.
    Handles whitespace, newline normalizations, and Arabic/Persian character variants.
    """
    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.normalize_persian_characters(text)
        text = self.normalize_spaces(text)
        text = self.fix_line_breaks(text)
        text = self.normalize_blank_lines(text)
        
        return text.strip()

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def remove_control_characters(self, text: str) -> str:
        cleaned_chars = [
            char for char in text 
            if not unicodedata.category(char).startswith("C") or char in ("\n", "\t", "\r")
        ]
        return "".join(cleaned_chars)

    def normalize_persian_characters(self, text: str) -> str:
        # Crucial for text embedding quality: normalizing Arabic/Persian letters
        replacements = {
            "ي": "ی",
            "ى": "ی",
            "ك": "ک",
            "ة": "ه",
            "ۀ": "ه",
        }
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
            
        # ZWNJ (Zero-width non-joiner) preservation for Persian half-spaces
        text = text.replace("\u200c", "\u200c")
        return text

    def normalize_spaces(self, text: str) -> str:
        text = text.replace("\t", " ").replace("\u00a0", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"[ ]+\n", "\n", text)
        text = re.sub(r"\n[ ]+", "\n", text)
        return text

    def fix_line_breaks(self, text: str) -> str:
        # Fix words broken by a hyphen (English standard)
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        # Fix single newlines inside paragraphs
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return text

    def normalize_blank_lines(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)

def clean_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaner = TextCleaner()
    cleaned_pages = []

    for page_data in pages:
        cleaned_page = {
            "content": cleaner.clean(page_data.get("content", "")),
            "metadata": page_data.get("metadata", {}).copy(),
        }
        cleaned_pages.append(cleaned_page)

    return cleaned_pages

if __name__ == "__main__":
    # Test execution
    sample_text = "This is a   test text.\nIt contains\nwrong line breaks and Arabic letters like ي and ك."
    cleaner = TextCleaner()
    print(f"Original:\n{sample_text}\n")
    print(f"Cleaned:\n{cleaner.clean(sample_text)}")