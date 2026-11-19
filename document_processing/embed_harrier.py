"""Embed the Markdown chunks with Harrier and save their row mapping.

Run from the repository root with:
    .venv/bin/python document_processing/embed_harrier.py

Document chunks are embedded as plain text. Harrier's query instruction is
only needed later, when embedding a user's search question.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


MODEL_NAME = "microsoft/harrier-oss-v1-270m"
MODEL_REVISION = "31de22b673913c7d658c0f03f792d77c2dcf8ebd"
EMBEDDING_DIM = 640
document_dir = Path(__file__).resolve().parent
chunks_path = document_dir / "harrier_chunks.jsonl"
embeddings_path = document_dir / "harrier_embeddings.npy"
manifest_path = document_dir / "harrier_embedding_manifest.json"


def main() -> None:
    chunk_bytes = chunks_path.read_bytes()
    chunks = [json.loads(line) for line in chunk_bytes.splitlines() if line.strip()]
    if not chunks:
        raise ValueError("No Markdown chunks found. Run chunk_markdown.py first.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique.")
    if any(not isinstance(chunk["text"], str) or not chunk["text"].strip()
           for chunk in chunks):
        raise ValueError("Every chunk must contain nonempty text.")

    # Harrier defaults to bfloat16 here; float32 gives stable CPU inference.
    model = SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION, model_kwargs={"dtype": torch.float32}
    )
    if model.get_embedding_dimension() != EMBEDDING_DIM:
        raise ValueError("Harrier's embedding dimension is not the expected 640.")

    texts = [chunk["text"] for chunk in chunks]
    oversized = [
        chunk_id
        for chunk_id, text in zip(chunk_ids, texts)
        if len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False))
        > model.max_seq_length
    ]
    if oversized:
        raise ValueError(f"Chunks exceed Harrier's token limit: {oversized}")

    embeddings = np.asarray(
        model.encode(texts, batch_size=2, normalize_embeddings=True,
                     show_progress_bar=True),
        dtype=np.float32,
    )
    norms = np.linalg.norm(embeddings, axis=1)
    if (embeddings.shape != (len(chunks), EMBEDDING_DIM)
            or not np.isfinite(embeddings).all()
            or not np.allclose(norms, 1.0, atol=1e-3)):
        raise ValueError(f"Invalid Harrier embeddings: {embeddings.shape}")

    manifest = {
        "model_id": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "embedding_dim": EMBEDDING_DIM,
        "normalized": True,
        "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "chunk_ids": chunk_ids,
    }
    np.save(embeddings_path, embeddings)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {embeddings.shape} Harrier embeddings to {embeddings_path}")
    print(f"Saved row mapping to {manifest_path}")


if __name__ == "__main__":
    main()
