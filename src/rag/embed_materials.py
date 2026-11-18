import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

from src.db.connection import ROOT
from src.rag.embed_chunks import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/materials-v2")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    prior = json.loads((ROOT / "eval/results/candidate-pool-2026-11-17/manifest.json").read_text(encoding="utf-8"))
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    with urlopen(base + "/api/tags", timeout=15) as response:
        live = next(m for m in json.load(response)["models"] if m["name"] == model)
    if model != prior["model"]["ollama_model"] or live["digest"] != prior["model"]["digest"]:
        raise ValueError("Embedding model weights differ from the baseline")
    input_path = args.input / "chunks.json"
    options = {"num_ctx": 8192, "num_thread": 12}
    identity = {"model": prior["model"], "chunks_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(), "options": options, "batch_size": 8}
    path = args.input / "embedding_identity.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Embedding snapshot or model changed during generation")
    path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    generate(input_path, args.input / "embeddings.jsonl", batch_size=8, options=options)


if __name__ == "__main__":
    main()
