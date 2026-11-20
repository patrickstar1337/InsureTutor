"""Inspect Harrier's five closest insurance passages for a question.

Run interactively:
    .venv/bin/python document_processing/harrier/retrieve_harrier.py

Or ask one question:
    .venv/bin/python document_processing/harrier/retrieve_harrier.py "What is the guaranteed rate?"
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


TOP_K = 5
PROMPT_NAME = "web_search_query"
document_dir = Path(__file__).resolve().parent
chunks_path = document_dir / "harrier_chunks.jsonl"
embeddings_path = document_dir / "harrier_embeddings.npy"
manifest_path = document_dir / "harrier_embedding_manifest.json"


def load_index():
    chunk_bytes = chunks_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(chunk_bytes).hexdigest() != manifest["chunks_sha256"]:
        raise ValueError("Chunks changed since indexing. Run embed_harrier.py again.")

    chunks = [json.loads(line) for line in chunk_bytes.splitlines() if line.strip()]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunk_ids = manifest["chunk_ids"]
    embeddings = np.load(embeddings_path, allow_pickle=False)
    expected_shape = (len(chunk_ids), manifest["embedding_dim"])
    if (
        not chunks
        or len(chunks_by_id) != len(chunks)
        or len(set(chunk_ids)) != len(chunk_ids)
        or set(chunk_ids) != set(chunks_by_id)
        or not manifest["normalized"]
        or embeddings.shape != expected_shape
        or embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
        or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)
    ):
        raise ValueError("Harrier index rows do not match the chunks and manifest.")

    return embeddings, chunk_ids, chunks_by_id, manifest


def load_model(manifest, embedding_dim):
    model = SentenceTransformer(
        manifest["model_id"],
        revision=manifest["model_revision"],
        model_kwargs={"dtype": torch.float32},
    )
    if model.get_embedding_dimension() != embedding_dim:
        raise ValueError("Harrier model dimension differs from the saved index.")
    return model


def retrieve_results(question, model, embeddings, chunk_ids, chunks_by_id):
    prompt = model.prompts.get(PROMPT_NAME)
    if prompt is None:
        raise ValueError(f"Harrier does not provide the {PROMPT_NAME!r} prompt.")
    token_count = len(
        model.tokenizer.encode(
            prompt + question, add_special_tokens=True, truncation=False
        )
    )
    if token_count > model.max_seq_length:
        raise ValueError(f"Question exceeds Harrier's {model.max_seq_length}-token limit.")

    query_vector = np.asarray(
        model.encode(question, prompt_name=PROMPT_NAME, normalize_embeddings=True),
        dtype=np.float32,
    )
    if (
        query_vector.shape != (embeddings.shape[1],)
        or not np.isfinite(query_vector).all()
        or not np.isclose(np.linalg.norm(query_vector), 1.0, atol=1e-3)
    ):
        raise ValueError("Harrier returned an invalid question embedding.")

    # Both sides are unit vectors, so their dot product is cosine similarity.
    scores = embeddings @ query_vector
    best_rows = np.argsort(scores)[::-1][:TOP_K]
    return [
        {**chunks_by_id[chunk_ids[row]], "score": float(scores[row])}
        for row in best_rows
    ]


def show_results(question, model, embeddings, chunk_ids, chunks_by_id):
    try:
        results = retrieve_results(question, model, embeddings, chunk_ids, chunks_by_id)
    except ValueError as error:
        print(error)
        return

    for rank, chunk in enumerate(results, start=1):
        pages = ", ".join(str(page) for page in chunk["pdf_pages"])
        print(
            f"\n{rank}. score={chunk['score']:.3f} | {chunk['chunk_id']} | "
            f"{chunk['source']} PDF pages {pages}"
        )
        print(chunk["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Ask one question and exit")
    args = parser.parse_args()

    embeddings, chunk_ids, chunks_by_id, manifest = load_index()
    model = load_model(manifest, embeddings.shape[1])

    if args.question is not None:
        question = args.question.strip()
        if question:
            show_results(question, model, embeddings, chunk_ids, chunks_by_id)
        return

    print(f"Loaded {len(chunk_ids)} chunks. Ask a question; press Enter to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        show_results(question, model, embeddings, chunk_ids, chunks_by_id)


if __name__ == "__main__":
    main()
