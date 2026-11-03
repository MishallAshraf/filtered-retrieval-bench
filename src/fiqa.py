"""Load FiQA documents, questions and labels, lined up with the saved embeddings."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIQA = ROOT / "data" / "fiqa"
EMB = ROOT / "data" / "embeddings"
SPLITS = ("test", "dev")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def documents():
    """Return (vectors, texts, doc_index) with index i the same row in each."""
    ids = json.loads((EMB / "corpus_ids.json").read_text())
    text_by_id = {d["_id"]: d["text"] for d in _read_jsonl(FIQA / "corpus.jsonl")}
    return np.load(EMB / "corpus.npy"), [text_by_id[i] for i in ids], {doc: i for i, doc in enumerate(ids)}


def labels(split, doc_index):
    """question id -> document indexes judged relevant. Labels pointing at the empty documents
    dropped before embedding are removed, and a question left with none is dropped."""
    found = {}
    with open(FIQA / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            question, doc, score = line.split()
            if int(score) > 0 and doc in doc_index:
                found.setdefault(question, []).append(doc_index[doc])
    return found


def questions():
    """Return rows of (qid, split, text, vector, question id). The test split comes first, so
    test questions are qid 0 to 647."""
    text_by_id = {q["_id"]: q["text"] for q in _read_jsonl(FIQA / "queries.jsonl")}
    rows = []
    for split in SPLITS:
        ids = json.loads((EMB / f"queries_{split}_ids.json").read_text())
        vectors = np.load(EMB / f"queries_{split}.npy")
        for question, vector in zip(ids, vectors):
            rows.append((len(rows), split, text_by_id[question], vector, question))
    return rows
