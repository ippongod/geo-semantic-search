"""Render results.json into the README (between markers) and results.md.

Idempotent and BOM-tolerant. Produces one before/after table per model family
plus a short qualitative retrieval example, and reports negative deltas honestly.
"""

from __future__ import annotations

import argparse
import json
import os

START = "<!--RESULTS_START-->"
END = "<!--RESULTS_END-->"

DISPLAY = {"minilm": "all-MiniLM-L6-v2", "bge": "bge-small-en-v1.5"}
ORDER = ["minilm", "bge"]
ROWS = [("Recall@1", "recall@1"), ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"), ("MRR@10", "mrr@10"),
        ("nDCG@10", "ndcg@10"), ("Accuracy@1", "accuracy@1")]


def pct(x):
    return f"{x * 100:.1f}%"


def signed_pp(x):
    return f"{x * 100:+.1f} pp"


def model_block(tag, entry, n_eval, corpus_size, seed):
    name = DISPLAY.get(tag, tag)
    base_id = entry.get("base_model", name)
    lines = [
        f"### {name}",
        "",
        f"_Held-out test set: {n_eval} queries, {corpus_size}-passage corpus "
        f"(seed={seed}). Base: `{base_id}`. Cosine similarity._",
        "",
        "| Metric | Base | Fine-tuned | Δ |",
        "|---|---|---|---|",
    ]
    base, tuned, delta = entry["base"], entry["tuned"], entry["delta"]
    for label, key in ROWS:
        lines.append(f"| {label} | {pct(base[key])} | {pct(tuned[key])} | "
                     f"{signed_pp(delta[key])} |")
    neg = [label for label, key in ROWS if delta[key] < 0]
    if neg:
        lines += ["", f"> Note: fine-tuned did not beat base on {', '.join(neg)} "
                      f"(reported as measured)."]

    ex = (entry.get("examples") or [None])[0]
    if ex:
        lines += ["", "<details><summary>Example retrieval (fine-tuned)</summary>",
                  "", f'Query: **"{ex["query"]}"**', ""]
        for h in ex["top"]:
            mark = "✅" if h["hit"] else "▫️"
            lines.append(f"- {mark} `{h['score']:.3f}` — {h['passage']}")
        lines.append("</details>")
    return "\n".join(lines)


def build_block(results):
    n_eval = results.get("eval_queries", "?")
    corpus_size = results.get("corpus_size", "?")
    seed = results.get("seed", 42)
    models = results.get("models", {})
    parts = []
    for tag in ORDER + [t for t in models if t not in ORDER]:
        if tag in models:
            parts.append(model_block(tag, models[tag], n_eval, corpus_size, seed))
    return "\n\n".join(parts) if parts else "_No results yet._"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--md-out", default="results.md")
    a = ap.parse_args()

    with open(a.results, encoding="utf-8-sig") as fh:
        results = json.load(fh)

    block = build_block(results)

    with open(a.md_out, "w", encoding="utf-8") as fh:
        fh.write("# Results\n\n" + block + "\n")

    if os.path.exists(a.readme):
        with open(a.readme, encoding="utf-8-sig") as fh:
            text = fh.read()
        wrapped = f"{START}\n{block}\n{END}"
        if START in text and END in text:
            pre = text.split(START, 1)[0]
            post = text.split(END, 1)[1]
            text = pre + wrapped + post
        else:
            text = text.rstrip() + "\n\n## Results\n\n" + wrapped + "\n"
        with open(a.readme, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[fill] updated {a.readme} and {a.md_out}")
    else:
        print(f"[fill] wrote {a.md_out} (no README found)")


if __name__ == "__main__":
    main()
