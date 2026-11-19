"""Make page-cited, token-sized chunks from the LlamaParse Markdown export.

Run from the repository root with:
    .venv/bin/python document_processing/chunk_markdown.py

This is a separate experiment from the existing E5/PyMuPDF pipeline. It only
creates chunks; it does not download the Harrier model weights or embed text.
"""

import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from transformers import AutoTokenizer


MODEL_NAME = "microsoft/harrier-oss-v1-270m"
MODEL_REVISION = "31de22b673913c7d658c0f03f792d77c2dcf8ebd"
MAX_CHUNK_TOKENS = 800
MAX_BLOCK_TOKENS = 650
MIN_SECTION_CHUNK_TOKENS = 450
OVERLAP_TOKENS = 100
SOURCE_PDF = "Insurance_Document.pdf"
document_dir = Path(__file__).resolve().parent
input_path = document_dir / "Insurance_Document.md"
output_path = document_dir / "harrier_chunks.jsonl"

# The cover has no printed page number. This verified line starts PDF page 2.
FIRST_CONTENT_PAGE_START = "您可創造更豐厚的財富"
FOOTER = re.compile(r"^\s*(\d{1,2})\s*$")
TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
DECORATIVE_CAPTION = re.compile(r"^(?:photo|logo|icon|code):\s", re.IGNORECASE)


class TableText(HTMLParser):
    """Extract cells as readable rows, repeating labels from rowspan cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int]]] = []
        self.row: list[tuple[str, int]] | None = None
        self.cell: list[str] | None = None
        self.rowspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"th", "td"}:
            self.cell = []
            self.rowspan = int(dict(attrs).get("rowspan") or 1)
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.cell is not None:
            value = re.sub(r"\s+", " ", "".join(self.cell)).strip()
            if self.row is not None:
                self.row.append((value, self.rowspan))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def lines(self) -> list[str]:
        active: list[tuple[str, int]] = []
        result: list[str] = []
        for row in self.rows:
            labels = [value for value, _ in active]
            active = [(value, left - 1) for value, left in active if left > 1]
            labels.extend(value for value, _ in row if value)
            active.extend((value, span - 1) for value, span in row if value and span > 1)
            if labels:
                result.append(" | ".join(labels))
        return result


def pdf_pages(markdown: str) -> list[tuple[int, str]]:
    """Use brochure footers and the unnumbered covers to recover PDF pages."""
    lines = markdown.splitlines()
    footers = [
        (index, int(match.group(1)))
        for index, line in enumerate(lines)
        if (match := FOOTER.fullmatch(line))
    ]
    if [number for _, number in footers] != list(range(1, 19)):
        raise ValueError("Expected printed brochure footers 1 through 18 in order.")

    starts = [
        index for index, line in enumerate(lines)
        if line.startswith(FIRST_CONTENT_PAGE_START)
    ]
    if len(starts) != 1 or not 0 < starts[0] < footers[0][0]:
        raise ValueError("Cannot locate the PDF page 1 / page 2 cover break.")

    slices = [lines[: starts[0]], lines[starts[0] : footers[0][0]]]
    slices.extend(
        lines[footers[number - 2][0] + 1 : footers[number - 1][0]]
        for number in range(2, 19)
    )
    slices.append(lines[footers[-1][0] + 1 :])
    if len(slices) != 20 or any(not "".join(page).strip() for page in slices):
        raise ValueError("Markdown did not resolve to 20 nonempty PDF pages.")
    return [(number, "\n".join(page)) for number, page in enumerate(slices, 1)]


def plain_markdown(page_text: str) -> str:
    """Turn HTML tables into rows while retaining normal Markdown structure."""
    if len(re.findall(r"<table\b", page_text, re.IGNORECASE)) != len(
        TABLE.findall(page_text)
    ):
        raise ValueError("An HTML table crosses a page or has no closing tag.")

    def replace_table(match: re.Match[str]) -> str:
        parser = TableText()
        parser.feed(match.group())
        return "\n\n" + "\n\n".join(parser.lines()) + "\n\n"

    text = TABLE.sub(replace_table, page_text)
    text = re.sub(r"</?sup\b[^>]*>", "", text, flags=re.IGNORECASE)
    return html.unescape(text)


def page_blocks(page_number: int, page_text: str) -> list[dict]:
    blocks: list[dict] = []
    paragraph: list[str] = []

    def finish() -> None:
        if paragraph:
            blocks.append(
                {"text": "\n".join(paragraph).strip(), "pdf_pages": [page_number],
                 "section_start": False}
            )
            paragraph.clear()

    for raw_line in plain_markdown(page_text).splitlines():
        line = raw_line.strip()
        if not line or line == "----":
            finish()
            continue
        if DECORATIVE_CAPTION.match(line):
            finish()
            continue
        heading = HEADING.match(line)
        if heading:
            finish()
            blocks.append(
                {"text": heading.group(2).replace("**", "").strip(),
                 "pdf_pages": [page_number], "section_start": len(heading.group(1)) == 2}
            )
        elif line.startswith("- "):
            finish()
            paragraph.append(line)
        else:
            paragraph.append(line)
    finish()
    return [block for block in blocks if block["text"]]


def split_large_block(text: str, count_tokens) -> list[str]:
    """Cut an unusually long paragraph or table row without losing its text."""
    label = text.split(" | ", 1)[0] if " | " in text else ""
    # A long table row may span chunks. Repeat its first cell so each piece
    # still identifies the benefit or charge being described.
    repeat_label = label if label and count_tokens(label) < 80 else ""
    budget = MAX_BLOCK_TOKENS - count_tokens(repeat_label) - 3 if repeat_label else MAX_BLOCK_TOKENS
    pieces: list[str] = []
    remaining = text
    while count_tokens(remaining) > budget:
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(remaining[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        cut = low
        if cut < 1:
            raise ValueError("A single character exceeds the token budget.")
        # Prefer a nearby sentence/word boundary over cutting in mid-word.
        candidates = [match.end() for match in re.finditer(r"[。！？；;.!?]\s*|\s+", remaining[:cut])
                      if match.end() >= cut * 0.7]
        if candidates:
            cut = candidates[-1]
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [pieces[0]] + [f"{repeat_label} | {part}" for part in pieces[1:]] if repeat_label else pieces


def tail_of(text: str, count_tokens) -> str:
    """Carry a complete short block or final sentence, not a sentence fragment."""
    if count_tokens(text) <= OVERLAP_TOKENS:
        return text
    if " | " in text:
        return ""  # A detached table-cell suffix loses its row label.
    starts = [match.end() for match in re.finditer(r"[。！？.!?；;]\s*", text)]
    for start in starts:
        suffix = text[start:].strip()
        if suffix and count_tokens(suffix) <= OVERLAP_TOKENS:
            return suffix
    return ""


def make_chunks(blocks: list[dict], count_tokens) -> list[dict]:
    pieces: list[dict] = []
    for block in blocks:
        for index, text in enumerate(split_large_block(block["text"], count_tokens)):
            pieces.append({**block, "text": text,
                           "section_start": block["section_start"] and index == 0})

    chunks: list[dict] = []
    current: list[dict] = []

    def emit() -> None:
        if current:
            chunks.append(
                {"text": "\n\n".join(piece["text"] for piece in current),
                 "pdf_pages": sorted({page for piece in current for page in piece["pdf_pages"]})}
            )

    for piece in pieces:
        if (piece["section_start"] and current
                and count_tokens("\n\n".join(part["text"] for part in current))
                >= MIN_SECTION_CHUNK_TOKENS):
            emit()
            current = []
        candidate = current + [piece]
        pages = {page for part in candidate for page in part["pdf_pages"]}
        if current and (count_tokens("\n\n".join(part["text"] for part in candidate))
                        > MAX_CHUNK_TOKENS or len(pages) > 2):
            emit()
            last = current[-1]
            overlap = tail_of(last["text"], count_tokens)
            current = ([{**last, "text": overlap, "section_start": False}]
                       if overlap and len(set(last["pdf_pages"] + piece["pdf_pages"])) <= 2
                       else [])
            if count_tokens("\n\n".join(part["text"] for part in current + [piece])) > MAX_CHUNK_TOKENS:
                current = []
        current.append(piece)
    emit()
    return chunks


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

    pages = pdf_pages(input_path.read_text(encoding="utf-8"))
    blocks = [block for number, text in pages for block in page_blocks(number, text)]
    chunks = make_chunks(blocks, count_tokens)
    if not chunks:
        raise ValueError("No chunks were generated.")
    for number, chunk in enumerate(chunks, 1):
        chunk["chunk_id"] = f"Insurance_Document:harrier:c{number:03d}"
        chunk["source"] = SOURCE_PDF
        if count_tokens(chunk["text"]) > MAX_CHUNK_TOKENS:
            raise ValueError(f"Chunk {number} exceeds {MAX_CHUNK_TOKENS} tokens.")
        if not chunk["pdf_pages"]:
            raise ValueError(f"Chunk {number} has no PDF page citation.")

    with output_path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} Harrier chunks from {len(pages)} PDF pages to {output_path}")
    print(f"Largest chunk: {max(count_tokens(chunk['text']) for chunk in chunks)} tokens")


if __name__ == "__main__":
    main()
