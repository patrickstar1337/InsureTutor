"""Embed every PDF chunk and save an index for later retrieval."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
document_dir = Path(__file__).resolve().parent
chunks_path = document_dir / "chunks.jsonl"
embeddings_path = document_dir / "embeddings.npy"
manifest_path = document_dir / "embedding_manifest.json"

with chunks_path.open(encoding="utf-8") as chunk_file:
    chunks = [json.loads(line) for line in chunk_file if line.strip()]
if not chunks:
    raise ValueError("No chunks found. Run chunk_pages.py first.")

chunk_ids = [chunk["chunk_id"] for chunk in chunks]
if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("Chunk IDs must be unique.")

model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
passages = [f"passage: {chunk['text']}" for chunk in chunks]
oversized = [
    chunk_id
    for chunk_id, passage in zip(chunk_ids, passages)
    if len(model.tokenizer.encode(passage, add_special_tokens=True, truncation=False))
    > model.max_seq_length
]
if oversized:
    raise ValueError(f"Chunks exceed the model token limit: {oversized}")

embeddings = np.asarray(
    model.encode(passages, batch_size=8, normalize_embeddings=True),
    dtype=np.float32,
)
if embeddings.shape != (len(chunks), 384) or not np.isfinite(embeddings).all():
    raise ValueError(f"Unexpected embedding output: {embeddings.shape}")

manifest = {
    "model_id": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
    "chunk_ids": chunk_ids,
}
np.save(embeddings_path, embeddings)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Saved {embeddings.shape} embeddings to {embeddings_path}")
print(f"Saved row mapping to {manifest_path}")
