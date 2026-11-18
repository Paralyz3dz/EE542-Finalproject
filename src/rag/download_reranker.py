import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.db.connection import ROOT
from src.rag.reranker import DEFAULT_MODEL, DEFAULT_REVISION, model_path


FILES = ["config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    model = os.getenv("RERANKER_MODEL", DEFAULT_MODEL)
    revision = settings.get("reranker_revision", DEFAULT_REVISION)
    target = model_path(model, revision)
    print(f"Downloading {model} at {revision} into {target}", flush=True)
    snapshot_download(repo_id=model, revision=revision, local_dir=target, allow_patterns=FILES, token=False)
    hashes = {}
    for name in FILES:
        with (target / name).open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    identity = {"model": model, "revision": revision, "files_sha256": hashes}
    (target / "model_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(identity, indent=2), flush=True)


if __name__ == "__main__":
    main()
