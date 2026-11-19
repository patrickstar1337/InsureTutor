"""Check E5 token limits and embed two sample insurance chunks."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"
chunks_path = Path(__file__).resolve().with_name("chunks.jsonl")

with chunks_path.open(encoding="utf-8") as chunk_file:
    chunks = [json.loads(line) for line in chunk_file]
if not chunks:
    raise ValueError("No chunks found. Run chunk_pages.py first.")

model = SentenceTransformer(MODEL_NAME)
passages = [f"passage: {chunk['text']}" for chunk in chunks]
token_counts = [
    len(model.tokenizer.encode(passage, add_special_tokens=True, truncation=False))
    for passage in passages
]

print(f"Model: {MODEL_NAME}")
print(f"Chunks checked: {len(chunks)}")
print(f"Token limit: {model.max_seq_length}")
print(f"Token counts: {min(token_counts)} to {max(token_counts)}")

oversized = [
    (chunk["chunk_id"], count)
    for chunk, count in zip(chunks, token_counts)
    if count > model.max_seq_length
]
safe_indices = [
    index for index, count in enumerate(token_counts) if count <= model.max_seq_length
]
if len(safe_indices) < 2:
    raise ValueError("Need at least two chunks within the token limit to test embeddings.")

sample_indices = [safe_indices[0], safe_indices[len(safe_indices) // 2]]
sample_passages = [passages[index] for index in sample_indices]
embeddings = model.encode(sample_passages, normalize_embeddings=True)

if embeddings.shape != (2, 384) or not np.isfinite(embeddings).all():
    raise ValueError(f"Unexpected embedding output: {embeddings.shape}")

print(f"Sample chunks: {[chunks[index]['chunk_id'] for index in sample_indices]}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Vector lengths: {[round(float(np.linalg.norm(row)), 4) for row in embeddings]}")

if oversized:
    for chunk_id, count in oversized:
        print(f"Too long: {chunk_id} ({count} tokens)")
    raise SystemExit("Some chunks would be truncated; adjust chunking first.")
