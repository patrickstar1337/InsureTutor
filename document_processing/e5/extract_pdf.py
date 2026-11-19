"""Save the insurance PDF's text as one JSON record per page."""

import json
from pathlib import Path

import pymupdf


document_dir = Path(__file__).resolve().parent
pdf_path = document_dir.parent / "Insurance_Document.pdf"
output_path = document_dir / "extracted_pages.jsonl"

with pymupdf.open(pdf_path) as document, output_path.open("w", encoding="utf-8") as output:
    for pdf_page, page in enumerate(document, start=1):
        record = {
            "source": pdf_path.name,
            "pdf_page": pdf_page,
            "text": page.get_text("text"),
        }
        output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(document)} pages to {output_path}")
