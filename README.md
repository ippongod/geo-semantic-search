# GEO Semantic Search — embedding fine-tune for SEO/GEO retrieval

Fine-tune a sentence-embedding model so a search **query** lands close to the
content that should answer it and far from everything else. This is the core of
semantic search, RAG, and **GEO (Generative Engine Optimization)** — getting the
right passage retrieved before a model ever writes an answer.

Unlike a generative fine-tune, this is **contrastive / retrieval** learning, and
it is measured with the standard information-retrieval metrics:
**Recall@1/5/10, MRR@10, nDCG@10** — reported **base vs fine-tuned** on a fixed
held-out split, for two different base models.

Everything is **$0**: the training data is generated locally with
[Ollama](https://ollama.com/) (`qwen2.5:7b`), and training + evaluation run on a
free Google Colab **T4**. No paid APIs, no keys.

## Why this project is interesting

- **Retrieval, not generation** — `MultipleNegativesRankingLoss` with in-batch
  negatives; a different ML skill from instruction/QLoRA fine-tuning.
- **Real IR evaluation** — `InformationRetrievalEvaluator` with a deterministic
  seed=42, query-level held-out split; the same split is used for every model so
  the comparison is fair.
- **Two baselines** — `all-MiniLM-L6-v2` (22M) and `bge-small-en-v1.5` (33M),
  each fine-tuned and measured, so the gain is shown on a weak and a strong base.
- **No QLoRA / no bitsandbytes** — the models are small enough for a full
  fine-tune, which keeps the Colab stack minimal and robust.
- **Honest numbers** — a metric where the fine-tuned model does *not* beat the
  base is reported as measured, never hidden.

## How it works

```
1. Generate data ($0, local)     2. Train + evaluate ($0, Colab T4)
   Ollama qwen2.5:7b writes          MultipleNegativesRankingLoss on
   passages + the search             (query, passage) pairs; then
   queries that should find  ───►    InformationRetrievalEvaluator
   them. Built into positive         scores base vs fine-tuned with
   (query, passage) pairs.           Recall@k / MRR / nDCG.
```

A document is a passage plus 3–6 search queries that should retrieve it. Each
query becomes a positive `(query, passage)` pair; passages form the corpus that
every query is searched against.

## Repository layout

```
data/
  seed.jsonl       hand-checked English seed documents (passage + queries)
  pairs.jsonl      {query, passage, doc_id} positive pairs
  corpus.jsonl     {doc_id, passage} — one row per unique passage
scripts/
  ir_common.py     single source of truth: schema, dedup, seed=42 split, IR helpers
  make_dataset.py  $0 local dataset generation via Ollama (with GPU/RAM guardrails)
  train.py         SentenceTransformer fine-tune (MNRL, fp16)
  eval.py          IR metrics, base vs fine-tuned -> results.json
  infer.py         semantic-search demo (top-k passages for a query)
  fill_results.py  writes results.json into this README
notebooks/
  colab_train.ipynb  one-click: clone -> install -> train x2 -> eval x2 -> download
requirements.txt   sentence-transformers, datasets, accelerate (torch from Colab)
```

## Quickstart

### A. Generate the dataset locally ($0)

Requires a running Ollama with `qwen2.5:7b` pulled.

```bash
python scripts/make_dataset.py --n 250 --batch 6
```

The script refuses any model larger than ~8B, verifies the model is 100% resident
on the GPU before generating, watches free RAM between batches, and writes
atomically.

### B. Train + evaluate on Colab (free T4)

Open `notebooks/colab_train.ipynb` in Colab → **Runtime → T4 GPU → Run all**.
It trains both models, prints the IR metrics, fills the Results block below, and
downloads the fine-tuned models as zips.

### C. Semantic search with the fine-tuned model

```bash
python scripts/infer.py --model models/geo-minilm \
  --query "where can I get single-origin coffee beans delivered"
```

## Results

<!--RESULTS_START-->
_Run the Colab notebook (step 7) to populate this with real Recall@k / MRR / nDCG
numbers for both models._
<!--RESULTS_END-->

## Approach & models

| Step | Where | Cost |
|---|---|---|
| Dataset generation (`qwen2.5:7b`) | Local GPU + Ollama | $0 |
| Training + IR evaluation | Colab T4 | $0 |
| Publishing | GitHub | $0 |

- **Loss:** `MultipleNegativesRankingLoss` (in-batch negatives — larger batch =
  more negatives = stronger signal; `NO_DUPLICATES` batch sampler).
- **Split:** query-level, seed=42; ~12% of unique queries held out. Held-out
  queries' source passages stay in the corpus so recall is meaningful.
- **Decoding/precision:** fp16 on the T4 (Turing has no bf16).

## Limitations

- Training data is **synthetic** (one local 7B model), so it reflects that
  model's notion of realistic web copy and search intent.
- The corpus is small relative to a production search index; metrics show the
  *relative* before/after gain, not absolute production performance.
- Two small base models are used for speed and reproducibility; larger encoders
  would raise the absolute ceiling.

## License

MIT — see [LICENSE](LICENSE).
