"""Generate a semantic-search dataset locally with Ollama ($0).

The model writes realistic web-page-style *passages* together with the search
*queries* that should retrieve each one. We then expand every document into
positive ``(query, passage)`` pairs sharing a content-addressed ``doc_id``.

Outputs:
  * ``data/pairs.jsonl``  -> ``{"query","passage","doc_id"}`` positive pairs
  * ``data/corpus.jsonl`` -> ``{"doc_id","passage"}`` (one row per unique passage)

Guardrails (this machine is fragile: 8GB VRAM, prior disk corruption):
  * refuses any model > ~8B (would offload to CPU/RAM),
  * verifies the model is 100% resident on the GPU before generating,
  * watches free RAM between batches and stops early if it gets low,
  * atomic writes so an interrupted run never corrupts the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import statistics
import sys

import requests

import ir_common as ir

DEFAULT_HOST = "127.0.0.1:11434"
MAX_PARAM_B = 8.05          # hard ceiling — anything bigger gets refused
KEEP_ALIVE = "5m"

INDUSTRIES = [
    "specialty coffee roaster", "boutique hotel", "dental clinic", "yoga studio",
    "electric bike shop", "vegan bakery", "wedding photographer", "law firm",
    "pet grooming salon", "indoor climbing gym", "craft brewery", "tattoo studio",
    "language school", "moving company", "solar panel installer", "car detailing",
    "physiotherapy practice", "accounting firm", "florist", "bookshop",
    "vintage clothing store", "personal trainer", "interior design studio",
    "plumbing service", "house cleaning service", "music school", "ceramics studio",
    "food truck", "escape room", "co-working space", "vegan restaurant",
    "barbershop", "optician", "veterinary clinic", "landscaping company",
    "SaaS analytics tool", "mobile app agency", "cybersecurity consultancy",
    "online course platform", "subscription meal kit", "handmade jewelry brand",
    "surf school", "mountain bike tours", "wine bar", "artisan cheese shop",
    "roofing contractor", "HVAC repair", "childcare center", "driving school",
    "3D printing service",
    # --- extended set for diversity at scale (fewer near-duplicate businesses) ---
    "sushi restaurant", "ramen shop", "pizzeria", "taco truck", "steakhouse",
    "ice cream parlor", "juice bar", "tea house", "cocktail lounge", "gastropub",
    "bed and breakfast", "hostel", "campground", "ski resort", "spa and wellness center",
    "nail salon", "hair salon", "massage therapy clinic", "chiropractic clinic",
    "acupuncture clinic", "mental health counseling", "fertility clinic",
    "pediatric clinic", "dermatology clinic", "urgent care center",
    "pharmacy", "hearing aid center", "orthodontist", "podiatry clinic",
    "auto repair shop", "tire shop", "motorcycle dealership", "boat rental",
    "bike repair shop", "locksmith", "appliance repair", "pest control service",
    "window cleaning service", "carpet cleaning service", "junk removal service",
    "electrician", "painter and decorator", "general contractor", "tiling service",
    "fence installation", "pool maintenance service", "gutter cleaning service",
    "interior plant service", "tree surgeon", "septic service",
    "art gallery", "framing shop", "record store", "comic book store",
    "board game cafe", "toy store", "hobby and model shop", "stationery shop",
    "candle maker", "soap and skincare brand", "perfumery", "leather goods maker",
    "watch repair shop", "bespoke tailor", "shoe repair shop", "bridal boutique",
    "children's clothing store", "outdoor gear shop", "running store",
    "guitar shop", "piano tuner", "DJ and event services", "party rental company",
    "catering company", "private chef service", "cooking class studio",
    "wine import shop", "butcher shop", "fishmonger", "farm stand",
    "fitness boot camp", "pilates studio", "martial arts dojo", "dance studio",
    "swim school", "tennis academy", "golf instruction", "rock gym",
    "tutoring center", "coding bootcamp", "test prep service", "art school",
    "photography studio", "videography agency", "branding studio", "copywriting agency",
    "digital marketing agency", "SEO consultancy", "IT support company",
    "managed hosting provider", "fintech startup", "logistics startup",
    "e-commerce fulfillment service", "translation agency", "voiceover studio",
    "recruitment agency", "bookkeeping service", "tax preparation service",
    "estate planning attorney", "immigration law firm", "notary service",
    "real estate agency", "property management company", "home staging service",
    "mortgage broker", "insurance brokerage",
]


# ---------------------------------------------------------------------------
# Resource guardrails
# ---------------------------------------------------------------------------
def free_ram_gb():
    """Available physical RAM in GB (Windows ctypes, /proc fallback)."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / (1024 ** 3)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            return None


def parse_billions(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", str(s))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Ollama HTTP
# ---------------------------------------------------------------------------
def _url(host, path):
    host = host if host.startswith("http") else "http://" + host
    return host.rstrip("/") + path


def ollama_get(host, path):
    return requests.get(_url(host, path), timeout=30).json()


def ollama_post(host, path, payload, timeout=300):
    r = requests.post(_url(host, path), json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def preflight(host, model):
    """Verify Ollama is up, the model exists, is <=8B, and lives on the GPU."""
    if not shutil.which("ollama"):
        sys.exit("[guard] 'ollama' not on PATH — install/start Ollama first.")
    try:
        tags = ollama_get(host, "/api/tags")
    except Exception as e:
        sys.exit(f"[guard] Ollama server unreachable at {host}: {e}")

    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(n == model or n.startswith(model) for n in names):
        sys.exit(f"[guard] model '{model}' not installed. have: {names}")

    # Size guard (refuse >8B): try /api/show parameter_size, then the name.
    size_b = None
    try:
        show = ollama_post(host, "/api/show", {"model": model}, timeout=30)
        size_b = parse_billions(show.get("details", {}).get("parameter_size", ""))
    except Exception:
        pass
    if size_b is None:
        size_b = parse_billions(model)
    if size_b is not None and size_b > MAX_PARAM_B:
        sys.exit(f"[guard] '{model}' is ~{size_b}B > {MAX_PARAM_B}B — refused "
                 f"(would offload to CPU/RAM on 8GB VRAM).")
    print(f"[guard] model={model} size~={size_b}B (ok)")

    # Warm up so the model is loaded, then confirm 100% GPU residency.
    print("[guard] warming up model on GPU...")
    ollama_post(host, "/api/generate",
                {"model": model, "prompt": "ok", "stream": False,
                 "keep_alive": KEEP_ALIVE, "options": {"num_predict": 1}})
    frac = 0.0
    for m in ollama_get(host, "/api/ps").get("models", []):
        if m.get("name", "").startswith(model) or m.get("model", "").startswith(model):
            frac = (m.get("size_vram", 0) or 0) / (m.get("size", 0) or 1)
            break
    print(f"[guard] GPU residency = {frac*100:.1f}%")
    if frac < 0.99:
        sys.exit(f"[guard] model only {frac*100:.1f}% on GPU — refusing to "
                 f"generate (CPU offload is slow and risks freezing this PC).")


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You generate training data for a semantic search engine. For each document "
    "you write (a) a realistic PASSAGE — 2 to 4 sentences of website copy for ONE "
    "concrete business, product, or service. Give the business a distinctive name "
    "and concrete, specific details (named offerings, features, audience, location) "
    "like a real homepage/product/service page; avoid generic, interchangeable "
    "copy. Then write (b) 3 to 6 diverse search QUERIES a real person would type to "
    "find THAT EXACT business/product.\n"
    "Hard rules:\n"
    "- Every query must be answerable using ONLY facts explicitly stated in the "
    "passage. Do NOT introduce any detail (a location, product, audience, or "
    "feature) that is not written in the passage.\n"
    "- Every query must specifically single out this passage's business/product. "
    "Never write a generic query that could match thousands of unrelated sites. If "
    "a query does not clearly point at this passage, drop it.\n"
    "- Use plain ASCII punctuation (straight quotes and apostrophes).\n"
    "- Vary the query styles: at least one short keyword query, at least one "
    "natural-language question, and at least one specific long-tail query.\n"
    "- Passages must be distinct businesses across the batch — no near-duplicates.\n"
    "Return ONLY a JSON array of objects with keys \"passage\" (string) and "
    "\"queries\" (array of strings). No prose, no markdown, no code fences."
)


# Normalize typographic punctuation -> plain ASCII for clean keyword-style data.
_PUNCT = {"’": "'", "‘": "'", "“": '"', "”": '"',
          "–": "-", "—": "-", "…": "...", " ": " "}


def tidy_text(s: str) -> str:
    for k, v in _PUNCT.items():
        s = s.replace(k, v)
    return s


def build_messages(seed_examples, industries, avoid, n):
    shots = ""
    if seed_examples:
        shots = "Examples of the exact format:\n" + json.dumps(
            seed_examples[:2], ensure_ascii=False, indent=2) + "\n\n"
    avoid_txt = ""
    if avoid:
        avoid_txt = ("Do NOT reuse or paraphrase these already-covered businesses: "
                     + "; ".join(avoid[-12:]) + ".\n")
    listing = "\n".join(f"{i + 1}. {ind}" for i, ind in enumerate(industries))
    user = (
        f"{shots}"
        f"Generate exactly {n} documents — ONE for EACH of these {n} industries, "
        f"using every industry exactly once:\n{listing}\n"
        f"Give each business a distinct invented name and a different city.\n"
        f"{avoid_txt}"
        f"Remember: each query must specifically retrieve its own passage. "
        f"Return ONLY a JSON array of {n} objects."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------
def extract_objects(text):
    """Return a list of dicts from possibly-messy model output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    # 1) whole text as JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in ("documents", "data", "items", "results"):
                if isinstance(data.get(key), list):
                    return [d for d in data[key] if isinstance(d, dict)]
            if "passage" in data:
                return [data]
    except Exception:
        pass
    # 2) regex scan for balanced-ish {...} objects containing "passage"
    out = []
    for m in re.finditer(r"\{[^{}]*\"passage\"[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Document -> pairs
# ---------------------------------------------------------------------------
def clean_document(obj, min_q, max_q):
    """Validate one generated document; return cleaned dict or None."""
    if not isinstance(obj, dict):
        return None
    passage = tidy_text(str(obj.get("passage", "")).strip())
    if not (ir.MIN_PASSAGE_CHARS <= len(passage) <= ir.MAX_PASSAGE_CHARS):
        return None
    raw_q = obj.get("queries", [])
    if not isinstance(raw_q, list):
        return None
    npass = ir.normalize_text(passage)
    seen, queries = set(), []
    for q in raw_q:
        q = tidy_text(str(q).strip())
        nq = ir.normalize_text(q)
        if not (ir.MIN_QUERY_CHARS <= len(q) <= ir.MAX_QUERY_CHARS):
            continue
        if nq in seen or nq == npass:
            continue
        seen.add(nq)
        queries.append(q)
    if len(queries) < min_q:
        return None
    return {"passage": passage, "queries": queries[:max_q]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of NEW documents to generate")
    ap.add_argument("--batch", type=int, default=6, help="documents requested per call")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--num-predict", type=int, default=1800)
    ap.add_argument("--seed-file", default="data/seed.jsonl")
    ap.add_argument("--out-pairs", default="data/pairs.jsonl")
    ap.add_argument("--out-corpus", default="data/corpus.jsonl")
    ap.add_argument("--min-queries", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=6)
    ap.add_argument("--max-calls", type=int, default=0, help="0 = auto")
    ap.add_argument("--min-free-ram", type=float, default=1.5)
    ap.add_argument("--no-preflight", action="store_true", help="skip Ollama guards (debug)")
    a = ap.parse_args()

    if not a.no_preflight:
        preflight(a.host, a.model)

    # Seed documents (hand-checked) seed the dataset AND act as few-shot examples.
    seed_docs = []
    try:
        for row in ir.load_pairs(a.seed_file):  # seed.jsonl uses {passage, queries}
            d = clean_document(row, a.min_queries, a.max_queries)
            if d:
                seed_docs.append(d)
        print(f"[seed] {len(seed_docs)} valid seed documents loaded")
    except FileNotFoundError:
        print(f"[seed] no seed file at {a.seed_file} (continuing without)")

    documents = list(seed_docs)          # final kept docs (seed first)
    raw_docs_parsed = 0
    avoid = [d["passage"].split(".")[0][:80] for d in seed_docs]

    max_calls = a.max_calls or (3 * math.ceil(a.n / a.batch) + 4)
    call = 0
    generated = 0
    rng_seed = a.seed

    # Deterministic round-robin over a shuffled industry permutation: every
    # industry is covered once before any repeats, minimizing near-duplicate
    # businesses (which would create ambiguous gold labels and unfair IR deltas).
    ind_rng = random.Random(rng_seed)
    ind_queue = []

    while generated < a.n and call < max_calls:
        ram = free_ram_gb()
        if ram is not None and ram < a.min_free_ram:
            print(f"[guard] free RAM {ram:.2f}GB < {a.min_free_ram}GB — stopping early.")
            break
        call += 1
        want = min(a.batch, a.n - generated)
        if len(ind_queue) < want:
            nxt = list(INDUSTRIES)
            ind_rng.shuffle(nxt)
            ind_queue.extend(nxt)
        industries = ind_queue[:want]      # one distinct industry per requested doc
        del ind_queue[:want]
        msgs = build_messages(seed_docs, industries, avoid, want)
        try:
            resp = ollama_post(a.host, "/api/chat", {
                "model": a.model, "messages": msgs, "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": a.temperature, "top_p": a.top_p,
                            "seed": rng_seed + call, "num_predict": a.num_predict},
            })
        except Exception as e:
            print(f"[call {call}] request failed: {e}")
            continue
        content = resp.get("message", {}).get("content", "")
        objs = extract_objects(content)
        raw_docs_parsed += len(objs)
        kept_this = 0
        existing_pass = {ir.normalize_text(d["passage"]) for d in documents}
        for obj in objs:
            d = clean_document(obj, a.min_queries, a.max_queries)
            if not d:
                continue
            npass = ir.normalize_text(d["passage"])
            if npass in existing_pass:
                continue
            existing_pass.add(npass)
            documents.append(d)
            avoid.append(d["passage"].split(".")[0][:80])
            generated += 1
            kept_this += 1
            if generated >= a.n:
                break
        print(f"[call {call}] parsed {len(objs)} -> kept {kept_this} "
              f"(generated {generated}/{a.n}, RAM {ram:.1f}GB)" if ram else
              f"[call {call}] parsed {len(objs)} -> kept {kept_this} (generated {generated}/{a.n})")

    # Build corpus + globally-unique-query pairs.
    corpus, pairs, seen_q = {}, [], set()
    for d in documents:
        did = ir.doc_id_for(d["passage"])
        corpus[did] = d["passage"]
        for q in d["queries"]:
            nq = ir.normalize_text(q)
            if nq in seen_q:        # one gold doc per query -> unambiguous IR
                continue
            seen_q.add(nq)
            pairs.append({"query": q, "passage": d["passage"], "doc_id": did})

    pairs = ir.dedup_pairs(pairs)
    corpus = ir.dedup_corpus(corpus)
    ir.validate_pairs(pairs)
    ir.validate_corpus(corpus)

    ir.atomic_write_jsonl(a.out_pairs, pairs)
    ir.atomic_write_jsonl(a.out_corpus,
                          [{"doc_id": cid, "passage": p} for cid, p in corpus.items()])

    # ---- quality report ----
    qlens = [len(p["query"]) for p in pairs]
    per_doc = {}
    for p in pairs:
        per_doc[p["doc_id"]] = per_doc.get(p["doc_id"], 0) + 1
    print("\n" + "=" * 60)
    print("DATASET QUALITY REPORT")
    print("=" * 60)
    print(f"documents kept (seed+gen) : {len(documents)}  (seed {len(seed_docs)})")
    print(f"unique passages (corpus)  : {len(corpus)}")
    print(f"query-passage pairs       : {len(pairs)}")
    print(f"ollama calls              : {call}")
    print(f"raw docs parsed           : {raw_docs_parsed}")
    if per_doc:
        print(f"avg queries / document    : {statistics.mean(per_doc.values()):.2f}")
    if qlens:
        print(f"query length (chars)      : min {min(qlens)} / "
              f"median {int(statistics.median(qlens))} / max {max(qlens)}")
    print(f"-> {a.out_pairs}")
    print(f"-> {a.out_corpus}")
    print("\n--- sample documents (inspect query<->passage relevance) ---")
    for did, passage in list(corpus.items())[:3]:
        qs = [p["query"] for p in pairs if p["doc_id"] == did]
        print(f"\n[{did}] {passage}")
        for q in qs:
            print(f"   - {q}")


if __name__ == "__main__":
    main()
