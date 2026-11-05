from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.db.connection import ROOT
from src.rag.retriever import EMBEDDING_DIM, embed_texts


def generate(input_path: Path, output_path: Path, batch_size: int = 8) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("No chunks found in the input snapshot.")
    ids = [chunk["id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("The snapshot contains duplicate chunk IDs.")
    for chunk in chunks:
        if not chunk["embed_text"].strip():
            raise ValueError(f"Empty embed_text for {chunk['id']}")

    model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    ollama_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    if partial.exists():
        with partial.open(encoding="utf-8") as existing:
            for line in existing:
                row = json.loads(line)
                if completed >= len(chunks):
                    raise ValueError("Partial export contains more records than the snapshot.")
                chunk = chunks[completed]
                expected_hash = hashlib.sha256(chunk["embed_text"].encode()).hexdigest()[:16]
                vector = np.asarray(row["embedding"], dtype=np.float32)
                if (row["id"] != chunk["id"] or row["content_hash"] != expected_hash
                        or row["embedding_model"] != model or row["ollama_model"] != ollama_model
                        or vector.shape != (EMBEDDING_DIM,)
                        or not np.isfinite(vector).all() or not np.any(vector)):
                    raise ValueError("Partial export does not match the current snapshot/model.")
                completed += 1

    print(f"Embedding {len(chunks)} chunks with {ollama_model}; resuming at {completed}.", flush=True)
    with partial.open("a", encoding="utf-8") as output:
        for start in range(completed, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = embed_texts([chunk["embed_text"] for chunk in batch])
            for chunk, vector in zip(batch, vectors):
                output.write(json.dumps({
                    "id": chunk["id"],
                    "embedding": vector.tolist(),
                    "content_hash": hashlib.sha256(chunk["embed_text"].encode()).hexdigest()[:16],
                    "embedding_model": model,
                    "ollama_model": ollama_model,
                }, allow_nan=False) + "\n")
            output.flush()
            print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)}", flush=True)
    partial.replace(output_path)
    print(f"Saved local vectors: {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local JSON list containing id and embed_text")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/embeddings_qwen3-0.6b.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    generate(args.input, args.out, args.batch_size)


if __name__ == "__main__":
    main()
