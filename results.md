# Results

### all-MiniLM-L6-v2

_Held-out test set: 154 queries, 270-passage corpus (seed=42). Base: `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity._

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Recall@1 | 74.7% | 87.7% | +13.0 pp |
| Recall@5 | 98.1% | 99.4% | +1.3 pp |
| Recall@10 | 99.4% | 100.0% | +0.7 pp |
| MRR@10 | 85.5% | 93.4% | +7.8 pp |
| nDCG@10 | 89.0% | 95.1% | +6.0 pp |
| Accuracy@1 | 74.7% | 87.7% | +13.0 pp |

### bge-small-en-v1.5

_Held-out test set: 154 queries, 270-passage corpus (seed=42). Base: `BAAI/bge-small-en-v1.5`. Cosine similarity._

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Recall@1 | 84.4% | 92.9% | +8.4 pp |
| Recall@5 | 100.0% | 100.0% | +0.0 pp |
| Recall@10 | 100.0% | 100.0% | +0.0 pp |
| MRR@10 | 91.5% | 96.2% | +4.6 pp |
| nDCG@10 | 93.7% | 97.1% | +3.4 pp |
| Accuracy@1 | 84.4% | 92.9% | +8.4 pp |
