"""Split extracted PDF pages into small passages for later embedding."""

import json
from pathlib import Path


CHUNK_SIZE = 1000
OVERLAP_SIZE = 150
MIN_LAST_CHUNK_SIZE = 300
document_dir = Path(__file__).resolve().parent
input_path = document_dir / "extracted_pages.jsonl"
output_path = document_dir / "chunks.jsonl"


def chunk_page(text: str) -> list[str]:
    # PDF text can contain invisible control characters such as bell (\x07).
    lines = [
        "".join(char if char.isprintable() else " " for char in line).strip()
        for line in text.splitlines()
    ]
    lines = [line for line in lines if line]
    chunks = []
    start = 0

    while start < len(lines):
        end = start
        length = 0
        while end < len(lines):
            added_length = len(lines[end]) + (1 if end > start else 0)
            if end > start and length + added_length > CHUNK_SIZE:
                break
            length += added_length
            end += 1

        # Move a few lines into the final chunk instead of leaving a tiny tail.
        if end < len(lines):
            remaining_length = len("\n".join(lines[end:]))
            while remaining_length < MIN_LAST_CHUNK_SIZE and end > start + 1:
                end -= 1
                remaining_length = len("\n".join(lines[end:]))

        chunks.append("\n".join(lines[start:end]))
        if end == len(lines):
            break

        # Repeat a few complete lines, while always advancing on the next pass.
        next_start = end
        overlap_length = 0
        while next_start > start + 1:
            previous_line = lines[next_start - 1]
            added_length = len(previous_line) + (1 if overlap_length else 0)
            if overlap_length + added_length > OVERLAP_SIZE:
                break
            next_start -= 1
            overlap_length += added_length
        start = next_start

    return chunks


chunk_count = 0
with input_path.open(encoding="utf-8") as pages, output_path.open(
    "w", encoding="utf-8"
) as output:
    for page_line in pages:
        page = json.loads(page_line)
        for chunk_number, chunk_text in enumerate(chunk_page(page["text"]), start=1):
            chunk = {
                "chunk_id": f"{Path(page['source']).stem}:p{page['pdf_page']}:c{chunk_number}",
                "source": page["source"],
                "pdf_page": page["pdf_page"],
                "text": chunk_text,
            }
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += 1

print(f"Saved {chunk_count} chunks to {output_path}")
