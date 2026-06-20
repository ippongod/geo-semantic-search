"""Evaluate base vs fine-tuned retrieval quality with IR metrics.

Runs the SAME InformationRetrievalEvaluator (same seed=42 held-out split, same
name -> same metric keys) on the base model and the fine-tuned model, then
records Recall@1/5/10, MRR@10, nDCG@10 and Accuracy@1 for both plus the delta.

Results are merged into results.json under ``models[tag]`` so the MiniLM and BGE
runs accumulate into a single file. Honest by design: a negative delta is
written out, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os

from sentence_transformers import SentenceTransformer, util

import ir_common as ir


def topk_examples(model, queries, corpus, rel, n=3, k=3):
    """A few qualitative query -> top-k retrievals for the README."""
    cids = list(corpus)
    cemb = model.encode([corpus[c] for c in cids], convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=False)
    out = []
    for qid in list(queries)[:n]:
        qemb = model.encode(queries[qid], convert_to_tensor=True,
                            normalize_embeddings=True, show_progress_bar=False)
        hits = util.semantic_search(qemb, cemb, top_k=k)[0]
        out.append({
            "query": queries[qid],
            "gold_doc_ids": sorted(rel[qid]),
            "top": [{"doc_id": cids[h["corpus_id"]],
                     "score": round(float(h["score"]), 3),
                     "hit": cids[h["corpus_id"]] in rel[qid],
                     "passage": corpus[cids[h["corpus_id"]]][:140]} for h in hits],
        })
    return out


def evaluate(path, evaluator, queries, corpus, rel):
    model = SentenceTransformer(path)          # local dir OR hub id
    raw = {k: float(v) for k, v in evaluator(model).items()}
    return ir.summarize_metrics(raw), raw, topk_examples(model, queries, corpus, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tuned", default="models/geo-minilm")
    ap.add_argument("--tag", default="minilm", help="model family key in results.json")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--eval_frac", type=float, default=ir.EVAL_FRAC)
    a = ap.parse_args()

    pairs = ir.dedup_pairs(ir.load_pairs(a.pairs))
    corpus = ir.load_corpus(a.corpus)
    _, eq, c, rel = ir.split_queries(pairs, corpus, a.eval_frac, ir.SEED)   # SAME split
    evaluator = ir.build_ir_evaluator(eq, c, rel)                           # SAME name

    print(f"[eval] tag={a.tag}  held-out queries={len(eq)}  corpus={len(c)}")
    base_sum, base_raw, base_ex = evaluate(a.base, evaluator, eq, c, rel)
    tuned_sum, tuned_raw, tuned_ex = evaluate(a.tuned, evaluator, eq, c, rel)
    delta = {k: round(tuned_sum[k] - base_sum[k], 4) for k in base_sum}

    entry = {
        "base_model": a.base, "tuned_model": a.tuned,
        "base": base_sum, "tuned": tuned_sum, "delta": delta,
        "base_full": base_raw, "tuned_full": tuned_raw,
        "examples": tuned_ex, "examples_base": base_ex,
    }

    # Merge into results.json (keep the other model family's entry intact).
    results = {"seed": ir.SEED, "eval_queries": len(eq), "corpus_size": len(c),
               "models": {}}
    if os.path.exists(a.out):
        try:
            with open(a.out, encoding="utf-8-sig") as fh:
                results = json.load(fh)
            results.setdefault("models", {})
        except Exception:
            pass
    results["seed"] = ir.SEED
    results["eval_queries"] = len(eq)
    results["corpus_size"] = len(c)
    results["models"][a.tag] = entry

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(json.dumps({"tag": a.tag, "base": base_sum, "tuned": tuned_sum,
                      "delta": delta}, indent=2))
    neg = [k for k, v in delta.items() if v < 0]
    if neg:
        print(f"[eval] NOTE: fine-tuned did NOT beat base on: {neg} "
              f"(reported honestly).")


if __name__ == "__main__":
    main()
