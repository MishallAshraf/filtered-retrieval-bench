"""Embed the FiQA corpus and evaluation queries with a local 384-dimension model."""
import json
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
FIQA = ROOT / "data" / "fiqa"
OUT = ROOT / "data" / "embeddings"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_qrels(split):
    qrels = {}
    with open(FIQA / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, doc_id, _ = line.split()
            qrels.setdefault(query_id, []).append(doc_id)
    return qrels


def embed(model, texts, batch_size=64):
    start = time.time()
    vectors = []
    for i, vector in enumerate(model.embed(texts, batch_size=batch_size)):
        vectors.append(vector)
        if (i + 1) % 5000 == 0:
            print(f"{i + 1}/{len(texts)} in {time.time() - start:.0f}s", flush=True)
    return np.asarray(vectors, dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(MODEL, cache_dir=str(ROOT / "data" / "model_cache"))

    corpus = [d for d in read_jsonl(FIQA / "corpus.jsonl") if d["text"].strip()]
    if (OUT / "corpus.npy").exists():
        print("corpus embeddings already saved, skipping", flush=True)
    else:
        np.save(OUT / "corpus.npy", embed(model, [d["text"] for d in corpus]))
        (OUT / "corpus_ids.json").write_text(json.dumps([d["_id"] for d in corpus]))

    queries = {q["_id"]: q["text"] for q in read_jsonl(FIQA / "queries.jsonl")}
    for split in ("dev", "test"):
        if (OUT / f"queries_{split}.npy").exists():
            print(f"{split} query embeddings already saved, skipping", flush=True)
            continue
        ids = sorted(read_qrels(split))
        np.save(OUT / f"queries_{split}.npy", embed(model, [queries[i] for i in ids]))
        (OUT / f"queries_{split}_ids.json").write_text(json.dumps(ids))

    (OUT / "manifest.json").write_text(json.dumps(
        {"model": MODEL, "dimensions": 384, "dataset": "fiqa", "documents": len(corpus)}, indent=2))
    print(f"done: {len(corpus)} documents", flush=True)


if __name__ == "__main__":
    main()
