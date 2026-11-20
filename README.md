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

## Web chat demo

Install the backend and frontend dependencies once:

```bash
.venv/bin/python -m pip install -r requirements.txt
cd frontend && npm ci
```

Make sure the repository-root `.env` contains your `LLM_BASE_URL`,
`LLM_API_KEY`, and `LLM_MODEL`. Start the API from the repository root in one
terminal:

```bash
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the React page:

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`). Vite
proxies `/api` to FastAPI, so the browser never receives the LLM API key.
FastAPI loads the Harrier model and saved index once at startup.

The page calls `POST /api/chat` with the current `question` and up to two prior
completed exchanges in `history`, for example:

```json
{
  "question": "Is that guaranteed for life?",
  "history": [
    {"role": "user", "content": "What is the guaranteed interest rate?"},
    {"role": "assistant", "content": "The brochure states 2.5%."}
  ]
}
```

The backend asks the LLM to turn a follow-up into a standalone search question,
retrieves Harrier passages with that question, then asks the LLM to answer the
original question using the passages. Earlier answers provide context only; the
brochure passages are the evidence. The first question uses one LLM call, while
later questions normally use two, plus any retries after errors. The history
window is limited to two exchanges and resets when the page reloads. The CLI
above still answers each question independently.

The response has `answer` and `sources`; each source has the PDF filename, PDF
page numbers, and chunk ID. The page labels these as retrieved passages because
claim-level citations have not been verified yet. A failed request leaves the
question in the text box for retry.
