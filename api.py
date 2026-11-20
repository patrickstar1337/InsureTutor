"""HTTP API for the InsureTutor chat page.

Run from the repository root:
    .venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from document_processing.harrier.retrieve_harrier import (
    load_index,
    load_model,
    retrieve_results,
)
from rag_answer import generate_answer, load_llm_settings, rewrite_question


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=4)


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
    if len(body.history) % 2 or any(
        turn.role != ("user" if index % 2 == 0 else "assistant")
        for index, turn in enumerate(body.history)
    ):
        raise HTTPException(status_code=422, detail="History must contain completed user/assistant pairs.")

    model, embeddings, chunk_ids, chunks_by_id, client, model_name = request.app.state.rag
    history = [turn.model_dump() for turn in body.history]
    try:
        search_question = rewrite_question(question, history, client, model_name)
        passages = retrieve_results(search_question, model, embeddings, chunk_ids, chunks_by_id)
        answer = generate_answer(question, passages, client, model_name, history)
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


# Vite serves the page during local development. The Docker image includes the
# production build, which FastAPI serves alongside /api/chat on one port.
frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
