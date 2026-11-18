import hashlib
import json
from pathlib import Path


POLICY = Path(__file__).resolve().parents[2] / "config/retrieval_exclusions.json"


def load_policy(path=POLICY):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("version") != 1:
        raise ValueError("Unsupported exclusion policy version")
    seen = set()
    for entry in policy["exclusions"]:
        if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in seen:
            raise ValueError("Invalid or duplicate exclusion ID")
        seen.add(entry["id"])
        for key, width in (("content_hash", 16), ("text_sha256", 64)):
            value = entry.get(key, "")
            if len(value) != width or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid exclusion fingerprint")
        if entry["reason"] not in {"index", "reference_list", "reference_directory"}:
            raise ValueError("Unknown exclusion reason")
    if seen & set(policy.get("protected_ids", [])):
        raise ValueError("A protected mixed-content chunk cannot be excluded")
    return policy


def filter_chunks(chunks, policy=None):
    policy = load_policy() if policy is None else policy
    entries = {entry["id"]: entry for entry in policy["exclusions"]}
    kept, excluded, stale = [], [], []
    for chunk in chunks:
        entry = entries.get(chunk["id"])
        if entry is not None:
            content = chunk.get("content_hash") or hashlib.sha256(chunk["embed_text"].encode()).hexdigest()[:16]
            text = hashlib.sha256(chunk["text"].encode()).hexdigest()
            if entry["content_hash"] == content and entry["text_sha256"] == text:
                excluded.append(chunk["id"])
                continue
            stale.append(chunk["id"])
        kept.append(chunk)
    return kept, excluded, stale
