"""HTTP API for the InsureTutor chat page.

Run from the repository root:
    .venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from document_processing.harrier.retrieve_harrier import (
    load_index,
    load_model,
    retrieve_results,
)
from rag_answer import generate_answer, load_llm_settings


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    pdf_pages: list[int]
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_llm_settings()
    embeddings, chunk_ids, chunks_by_id, manifest = load_index()
    model = load_model(manifest, embeddings.shape[1])
    with OpenAI(
        api_key=settings["LLM_API_KEY"],
        base_url=settings["LLM_BASE_URL"],
        timeout=30.0,
        max_retries=1,
    ) as client:
        app.state.rag = (model, embeddings, chunk_ids, chunks_by_id, client, settings["LLM_MODEL"])
        yield
    del app.state.rag


app = FastAPI(title="InsureTutor", lifespan=lifespan)


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    model, embeddings, chunk_ids, chunks_by_id, client, model_name = request.app.state.rag
    try:
        passages = retrieve_results(question, model, embeddings, chunk_ids, chunks_by_id)
        answer = generate_answer(question, passages, client, model_name)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail="The answer service is unavailable. Please try again shortly.",
        ) from error

    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                source=passage["source"],
                pdf_pages=passage["pdf_pages"],
                chunk_id=passage["chunk_id"],
            )
            for passage in passages
        ],
    )
