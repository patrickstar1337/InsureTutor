"""Split extracted PDF pages into small passages for later embedding."""

import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 50
MIN_LAST_CHUNK_TOKENS = 100
document_dir = Path(__file__).resolve().parent
input_path = document_dir / "extracted_pages.jsonl"
output_path = document_dir / "chunks.jsonl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def token_count(lines: list[str]) -> int:
    passage = "passage: " + "\n".join(lines)
    return len(
        tokenizer.encode(
            passage, add_special_tokens=True, truncation=False, verbose=False
        )
    )


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
        end = start + 1
        if token_count(lines[start:end]) > CHUNK_TOKENS:
            raise ValueError(f"One extracted line exceeds {CHUNK_TOKENS} tokens.")
        while end < len(lines) and token_count(lines[start : end + 1]) <= CHUNK_TOKENS:
            end += 1

        # Move a few lines into the final chunk instead of leaving a tiny tail.
        if end < len(lines):
            while end > start + 1 and token_count(lines[end:]) < MIN_LAST_CHUNK_TOKENS:
                end -= 1

        chunks.append("\n".join(lines[start:end]))
        if end == len(lines):
            break

        # Repeat up to 50 tokens of complete lines, while still advancing.
        next_start = end
        while (
            next_start > start + 1
            and token_count(lines[next_start - 1 : end]) <= OVERLAP_TOKENS
        ):
            next_start -= 1
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
