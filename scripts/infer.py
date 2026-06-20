"""Semantic search demo: embed a query and return the top-k passages.

    python scripts/infer.py --query "where to buy single-origin coffee beans"

Uses the fine-tuned model by default and the project corpus, so you can see the
trained embeddings actually retrieving relevant content.
"""

from __future__ import annotations

import argparse
import os

from sentence_transformers import SentenceTransformer, util

import ir_common as ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--model", default="models/geo-minilm")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()

    if not os.path.isdir(a.model):
        raise SystemExit(f"[infer] model dir '{a.model}' not found — train it on "
                         f"Colab and unzip it here, or pass --model <hub id>.")

    corpus = ir.load_corpus(a.corpus)
    cids, passages = list(corpus), list(corpus.values())

    model = SentenceTransformer(a.model)
    cemb = model.encode(passages, convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=False)
    qemb = model.encode(a.query, convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=False)
    hits = util.semantic_search(qemb, cemb, top_k=a.top_k)[0]

    print(f'\nQuery: "{a.query}"\nModel: {a.model}\n')
    for rank, h in enumerate(hits, 1):
        cid = cids[h["corpus_id"]]
        print(f"{rank}. [{h['score']:.3f}] ({cid}) {corpus[cid]}")


if __name__ == "__main__":
    main()
