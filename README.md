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

## RAG answers

Install the LLM client in the shared virtual environment, then copy the
configuration template:

```bash
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with the evaluator's OpenAI-compatible chat endpoint, API key, and
model name. The settings are `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`;
`.env` is ignored by Git. The Harrier embeddings still run locally, so the
provider does not need an embeddings API.

Ask questions interactively from the repository root, or pass one question on
the command line:

```bash
.venv/bin/python rag_answer.py
.venv/bin/python rag_answer.py "What is the guaranteed interest rate?"
```

In interactive mode, type a question and press Enter. An empty line exits.
Each question triggers a separate LLM request and is answered independently;
the script does not retain conversation history yet.

The script retrieves the five closest Harrier chunks and sends their **text**
and PDF page numbers with the question to the chat model. It prints the answer
and the retrieved sources. These sources are for inspection; this first version
does not yet verify that each answer claim is supported by a citation. The
grounding instruction asks the model to say when the excerpts are insufficient,
but that behavior still needs to be tested against unsupported questions.
