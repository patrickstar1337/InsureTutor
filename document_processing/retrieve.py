"""Inspect the most relevant insurance passages for a question."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TOP_K = 5
document_dir = Path(__file__).resolve().parent
chunks_path = document_dir / "chunks.jsonl"
embeddings_path = document_dir / "embeddings.npy"
manifest_path = document_dir / "embedding_manifest.json"


def load_index():
    chunk_bytes = chunks_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(chunk_bytes).hexdigest() != manifest["chunks_sha256"]:
        raise ValueError("Chunks changed since indexing. Run embed_chunks.py again.")

    chunks = [json.loads(line) for line in chunk_bytes.splitlines() if line.strip()]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunk_ids = manifest["chunk_ids"]
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if (
        len(chunks_by_id) != len(chunks)
        or len(set(chunk_ids)) != len(chunk_ids)
        or set(chunk_ids) != set(chunks_by_id)
        or embeddings.ndim != 2
        or embeddings.shape != (len(chunk_ids), 384)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Index rows do not match the saved chunks and manifest.")

    return embeddings, chunk_ids, chunks_by_id, manifest


def main():
    embeddings, chunk_ids, chunks_by_id, manifest = load_index()
    model = SentenceTransformer(
        manifest["model_id"], revision=manifest["model_revision"]
    )
    print(f"Loaded {len(chunk_ids)} chunks. Ask a question; press Enter to quit.")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        query = f"query: {question}"
        token_count = len(
            model.tokenizer.encode(query, add_special_tokens=True, truncation=False)
        )
        if token_count > model.max_seq_length:
            print(f"Question exceeds the model's {model.max_seq_length}-token limit.")
            continue

        query_vector = model.encode(query, normalize_embeddings=True)
        # The saved passage vectors are normalized too, so this is cosine similarity.
        scores = embeddings @ query_vector
        best_rows = np.argsort(scores)[::-1][:TOP_K]
        for rank, row in enumerate(best_rows, start=1):
            chunk = chunks_by_id[chunk_ids[row]]
            print(
                f"\n{rank}. score={scores[row]:.3f} | {chunk['chunk_id']} | "
                f"{chunk['source']} PDF page {chunk['pdf_page']}"
            )
            print(chunk["text"])


if __name__ == "__main__":
    main()
