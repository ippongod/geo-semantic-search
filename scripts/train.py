"""Fine-tune a SentenceTransformer for semantic SEO/GEO retrieval.

Contrastive training with MultipleNegativesRankingLoss (in-batch negatives) on
the positive ``(query, passage)`` pairs. No QLoRA / no bitsandbytes — the model
is tiny enough for a full fine-tune, which keeps the Colab stack minimal.

The trainer backend is auto-selected: we try the v3 SentenceTransformerTrainer
with a 2-step smoke run and fall back to the legacy ``model.fit`` if anything in
that path is unavailable on the runtime.
"""

from __future__ import annotations

import argparse

from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                    SentenceTransformerTrainingArguments, InputExample)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import ir_common as ir


def make_args(out, batch, epochs, lr, **ov):
    return SentenceTransformerTrainingArguments(
        output_dir=out,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=True, bf16=False,                       # T4 is Turing: fp16 yes, bf16 no
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # avoid false in-batch negatives
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
        seed=ir.SEED,
        **ov,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="models/geo-minilm")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval_frac", type=float, default=ir.EVAL_FRAC)
    a = ap.parse_args()

    pairs = ir.dedup_pairs(ir.load_pairs(a.pairs))
    ir.validate_pairs(pairs)
    corpus = ir.load_corpus(a.corpus)
    ir.validate_corpus(corpus)

    # Same deterministic split as eval.py -> identical held-out set for every model.
    train_pairs, eq, c, rel = ir.split_queries(pairs, corpus, a.eval_frac, ir.SEED)
    train_ds = ir.build_train_dataset(train_pairs)
    evaluator = ir.build_ir_evaluator(eq, c, rel)
    print(f"[train] base={a.base}  train_pairs={len(train_pairs)}  "
          f"held-out queries={len(eq)}  corpus={len(c)}")

    fresh = lambda: SentenceTransformer(a.base)

    # ---- pick backend with a tiny smoke run ----
    backend = "trainer"
    try:
        sm = fresh()
        from datasets import Dataset
        smoke = Dataset.from_dict({"anchor": train_ds["anchor"][:16],
                                   "positive": train_ds["positive"][:16]})
        SentenceTransformerTrainer(
            model=sm,
            args=make_args(a.out + "_smoke", batch=8, epochs=1, lr=a.lr, max_steps=2),
            train_dataset=smoke,
            loss=MultipleNegativesRankingLoss(sm),
        ).train()
        print("[train] backend = SentenceTransformerTrainer")
    except Exception as e:
        backend = "fit"
        print(f"[train] Trainer smoke failed -> legacy model.fit  ({e})")

    # ---- real training on a fresh model ----
    model = fresh()
    loss = MultipleNegativesRankingLoss(model)
    if backend == "trainer":
        SentenceTransformerTrainer(
            model=model,
            args=make_args(a.out, batch=a.batch, epochs=a.epochs, lr=a.lr),
            train_dataset=train_ds,
            loss=loss,
            evaluator=evaluator,
        ).train()
    else:
        from torch.utils.data import DataLoader
        examples = [InputExample(texts=[p["query"], p["passage"]]) for p in train_pairs]
        dl = DataLoader(examples, shuffle=True, batch_size=a.batch)
        model.fit(
            train_objectives=[(dl, loss)],
            epochs=a.epochs,
            warmup_steps=int(0.1 * len(dl) * a.epochs),
            optimizer_params={"lr": a.lr},
            use_amp=True,
            evaluator=evaluator,
            output_path=a.out,
            show_progress_bar=True,
        )

    model.save_pretrained(a.out)
    print(f"[train] saved fine-tuned model -> {a.out}  (backend={backend})")


if __name__ == "__main__":
    main()
