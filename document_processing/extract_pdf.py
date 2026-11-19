"""Print text from the first two pages of the insurance PDF for inspection."""

from pathlib import Path

import pymupdf


pdf_path = Path(__file__).resolve().parent.parent / "Insurance_Document.pdf"

with pymupdf.open(pdf_path) as document:
    for page_index in range(min(2, len(document))):
        page = document[page_index]
        print(f"\n--- PDF page {page_index + 1} ---")
        print(page.get_text("text"))
