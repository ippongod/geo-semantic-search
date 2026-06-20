"""Single source of truth for the GEO semantic-search project.

Everything that must stay consistent across dataset building, training and
evaluation lives here so the three scripts can never drift apart:

  * the data schema (positive ``(query, passage)`` pairs + a ``doc_id`` corpus),
  * text normalization and deduplication,
  * the deterministic *query-level* train/eval split (seed=42), and
  * the IR evaluator / training-dataset builders + metric helpers.

The module top level imports **stdlib only** so ``make_dataset.py`` can reuse
the schema/dedup helpers locally without pulling in ``sentence-transformers``
or ``datasets``. The heavy libraries are imported lazily inside the few
functions that actually need them (these run on Colab).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import unicodedata

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
SEED = 42
EVAL_FRAC = 0.12          # fraction of *unique queries* held out for IR eval
EVAL_NAME = "ir"          # evaluator name -> metric key prefix ("ir_cosine_*")

MIN_PASSAGE_CHARS = 60    # a passage must be a real 2-4 sentence description
MAX_PASSAGE_CHARS = 1200
MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 160


# ---------------------------------------------------------------------------
# Normalization / ids
# ---------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Lowercase, NFC-normalize, strip and collapse whitespace.

    Used as the canonical key for dedup and for the split, so the *same*
    normalization is applied everywhere (build, train, eval) -> no skew.
    """
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace(" ", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def doc_id_for(passage: str) -> str:
    """Stable content-addressed id for a passage.

    Identical (normalized) passages collapse to the same ``doc_id``, which
    makes corpus dedup automatic and keeps pair -> corpus references valid.
    """
    h = hashlib.sha1(normalize_text(passage).encode("utf-8")).hexdigest()
    return "d" + h[:12]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def atomic_write_jsonl(path: str, records) -> None:
    """Write JSONL atomically (tmp file + os.replace) so an interrupted run
    never leaves a half-written dataset."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_pairs(path: str) -> list:
    """Load positive pairs: ``[{"query","passage","doc_id"}, ...]``."""
    return _read_jsonl(path)


def load_corpus(path: str) -> dict:
    """Load corpus ``{"doc_id","passage"}`` rows into a ``{doc_id: passage}`` dict."""
    corpus = {}
    for row in _read_jsonl(path):
        corpus[str(row["doc_id"])] = str(row["passage"])
    return corpus


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_pairs(pairs: list) -> None:
    if not pairs:
        raise ValueError("no pairs loaded")
    for i, p in enumerate(pairs):
        for k in ("query", "passage", "doc_id"):
            if not isinstance(p.get(k), str) or not p[k].strip():
                raise ValueError(f"pair {i}: missing/blank '{k}': {p!r}")


def validate_corpus(corpus: dict) -> None:
    if not corpus:
        raise ValueError("empty corpus")
    for cid, passage in corpus.items():
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError(f"corpus {cid}: blank passage")


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
def dedup_pairs(pairs: list) -> list:
    """Drop duplicate ``(normalized query, doc_id)`` pairs, preserving order."""
    seen = set()
    out = []
    for p in pairs:
        key = (normalize_text(p["query"]), str(p["doc_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def dedup_corpus(corpus: dict) -> dict:
    """Collapse passages with identical normalized text to a single doc_id."""
    out = {}
    seen = {}
    for cid, passage in corpus.items():
        norm = normalize_text(passage)
        if norm in seen:
            continue
        seen[norm] = cid
        out[cid] = passage
    return out


# ---------------------------------------------------------------------------
# Deterministic query-level split (seed=42)
# ---------------------------------------------------------------------------
def split_queries(pairs: list, corpus: dict, eval_frac: float = EVAL_FRAC,
                  seed: int = SEED):
    """Hold out ``eval_frac`` of the *unique queries* for IR evaluation.

    Returns ``(train_pairs, eval_queries, corpus, relevant_docs)`` where:
      * ``train_pairs``    -> list of pairs whose query is NOT held out,
      * ``eval_queries``   -> ``{qid: query_text}`` for held-out queries,
      * ``corpus``         -> the FULL ``{doc_id: passage}`` haystack (the
                              held-out queries' source passages stay in it,
                              otherwise recall would be meaningless),
      * ``relevant_docs``  -> ``{qid: {doc_id, ...}}`` gold set per query.

    The split is a pure function of ``(pairs, corpus, eval_frac, seed)``, so
    every caller (train.py AND eval.py, for BOTH base models) gets a bit-for-bit
    identical held-out set -> the MiniLM vs BGE comparison is fair.
    """
    pairs = dedup_pairs(pairs)

    # Group by normalized query; keep first-seen original text + its gold docs.
    grouped = {}            # norm_q -> {"text": str, "docs": set, "pairs": list}
    for p in pairs:
        nq = normalize_text(p["query"])
        g = grouped.setdefault(nq, {"text": p["query"], "docs": set(), "pairs": []})
        g["docs"].add(str(p["doc_id"]))
        g["pairs"].append(p)

    norm_queries = sorted(grouped)          # deterministic order before shuffle
    rng = random.Random(seed)
    rng.shuffle(norm_queries)

    n_eval = max(1, int(round(len(norm_queries) * eval_frac)))
    held = set(norm_queries[-n_eval:])

    eval_queries = {}
    relevant_docs = {}
    train_pairs = []
    for idx, nq in enumerate(sorted(grouped)):   # stable qid assignment
        if nq in held:
            qid = f"q{idx:05d}"
            eval_queries[qid] = grouped[nq]["text"]
            relevant_docs[qid] = set(grouped[nq]["docs"])
        else:
            train_pairs.extend(grouped[nq]["pairs"])

    # Held-out integrity: every gold doc MUST be in the corpus.
    missing = {c for docs in relevant_docs.values() for c in docs if c not in corpus}
    if missing:
        raise ValueError(
            f"{len(missing)} held-out gold doc_id(s) absent from corpus, e.g. "
            f"{sorted(missing)[:3]} -- recall would be invalid")

    return train_pairs, eval_queries, dict(corpus), relevant_docs


# ---------------------------------------------------------------------------
# Builders (lazy heavy imports — Colab only)
# ---------------------------------------------------------------------------
def build_train_dataset(train_pairs: list):
    """Build a 2-column ``datasets.Dataset`` for MultipleNegativesRankingLoss.

    MNRL uses column *order* (anchor, positive), not the names. We map
    ``query -> anchor`` and ``passage -> positive``.
    """
    from datasets import Dataset
    return Dataset.from_dict({
        "anchor": [p["query"] for p in train_pairs],
        "positive": [p["passage"] for p in train_pairs],
    })


def build_ir_evaluator(eval_queries: dict, corpus: dict, relevant_docs: dict,
                       name: str = EVAL_NAME):
    """Build an InformationRetrievalEvaluator with a fixed ``name`` so the
    metric keys (``{name}_cosine_*@k``) match across base and fine-tuned runs."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    return InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        show_progress_bar=False,
        batch_size=64,
        corpus_chunk_size=max(1000, len(corpus)),
    )


# ---------------------------------------------------------------------------
# Metric helpers (suffix-based -> robust to cosine/cos_sim key naming)
# ---------------------------------------------------------------------------
# Short name -> key suffix as emitted by InformationRetrievalEvaluator.
METRIC_SUFFIXES = {
    "recall@1": "_recall@1",
    "recall@5": "_recall@5",
    "recall@10": "_recall@10",
    "mrr@10": "_mrr@10",
    "ndcg@10": "_ndcg@10",
    "accuracy@1": "_accuracy@1",
}


def metric(results: dict, suffix: str) -> float:
    """Look up a metric by key *suffix* (e.g. ``_ndcg@10``) so the code is
    immune to the ``cosine`` vs legacy ``cos_sim`` prefix difference."""
    hits = [v for k, v in results.items() if k.endswith(suffix)]
    if not hits:
        raise KeyError(
            f"no metric key ending in '{suffix}'. available: {sorted(results)}")
    return float(hits[0])


def summarize_metrics(results: dict) -> dict:
    """Flatten the evaluator output to short, stable keys for results.json."""
    return {name: metric(results, sfx) for name, sfx in METRIC_SUFFIXES.items()}
