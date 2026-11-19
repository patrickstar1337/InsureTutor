# InsureTutor

## Document processing

The source PDF and its LlamaParse Markdown export live in `document_processing/`.
The two retrieval experiments keep their scripts and generated files separate:

- `document_processing/e5/`: PyMuPDF extraction and multilingual E5 small.
- `document_processing/harrier/`: Markdown chunking and Harrier OSS v1 270M.

Use the shared virtual environment at the repository root. Run each pipeline
from the repository root in this order:

```bash
.venv/bin/python document_processing/e5/extract_pdf.py
.venv/bin/python document_processing/e5/chunk_pages.py
.venv/bin/python document_processing/e5/embed_chunks.py
.venv/bin/python document_processing/e5/retrieve.py
```

```bash
.venv/bin/python document_processing/harrier/chunk_markdown.py
.venv/bin/python document_processing/harrier/embed_harrier.py
.venv/bin/python document_processing/harrier/retrieve_harrier.py
```

The retrieval scripts accept questions interactively. The Harrier script also
accepts a quoted question as a command line argument.
