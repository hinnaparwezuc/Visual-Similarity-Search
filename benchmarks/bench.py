"""
bench.py — benchmark harness for the visual similarity search project.

Measures any index the same way, so numbers are comparable across
implementations (flat baseline, FAISS, your own HNSW later).

Reports per index:
  build_time_s     how long the index took to build
  build_mem_mb     peak memory allocated during build
  recall_at_k      agreement with exact brute-force search
  label_acc_at_k   optional: accuracy against your hand-labeled query set
  latency p50/p95  per-query search time in milliseconds

USAGE
  1. Fill in load_data() for your own files.
  2. python bench.py
  3. Results print as a table and save to results.json

Adding an index later means writing a class with .build() and .search()
and appending it to the INDEXES list at the bottom. Nothing else changes.
"""

import json
import time
import tracemalloc
from statistics import median

import numpy as np


# ----------------------------------------------------------------------
# 1. DATA  — replace this with your own loading code
# ----------------------------------------------------------------------

def load_data():
    """
    Return:
      db      (N, D) float32, L2-normalized  — your indexed image embeddings
      queries (Q, D) float32, L2-normalized  — your query embeddings
      labels  list of length Q, or None      — labels[i] = set of correct
                                               db indices for queries[i]

    Normalizing to unit length means inner product == cosine similarity,
    which keeps the comparison to FAISS's IndexFlatIP honest.
    """
    # --- replace with e.g. np.load("embeddings.npy") ---
    rng = np.random.default_rng(0)
    db = rng.standard_normal((5000, 512)).astype("float32")
    queries = rng.standard_normal((200, 512)).astype("float32")
    labels = None
    # ---------------------------------------------------

    db /= np.linalg.norm(db, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return db, queries, labels


# ----------------------------------------------------------------------
# 2. INDEXES  — each needs build(db) and search(q, k) -> array of ids
# ----------------------------------------------------------------------

class FlatIndex:
    """Exact brute-force search. The correctness oracle everything else
    is measured against. Slow by design — that is the point."""

    name = "flat (exact)"

    def build(self, db):
        self.db = db

    def search(self, q, k):
        sims = self.db @ q                      # cosine, since rows are unit norm
        top = np.argpartition(-sims, k)[:k]     # O(N) partial select, not a full sort
        return top[np.argsort(-sims[top])]


class FaissFlatIP:
    """FAISS exact inner-product. Same results as FlatIndex, much faster —
    isolates how much of FAISS's speed is just a better implementation
    rather than approximation."""

    name = "faiss IndexFlatIP"

    def build(self, db):
        import faiss
        self.index = faiss.IndexFlatIP(db.shape[1])
        self.index.add(db)

    def search(self, q, k):
        _, ids = self.index.search(q.reshape(1, -1), k)
        return ids[0]


class FaissHNSW:
    """FAISS's HNSW. This is the number your own implementation has to beat,
    or fail to beat for a reason you can name."""

    name = "faiss HNSW"

    def __init__(self, M=32, ef_construction=200, ef_search=64):
        self.M, self.efc, self.efs = M, ef_construction, ef_search
        self.name = f"faiss HNSW (M={M}, efS={ef_search})"

    def build(self, db):
        import faiss
        self.index = faiss.IndexHNSWFlat(db.shape[1], self.M,
                                         faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.efc
        self.index.add(db)
        self.index.hnsw.efSearch = self.efs

    def search(self, q, k):
        _, ids = self.index.search(q.reshape(1, -1), k)
        return ids[0]


# ----------------------------------------------------------------------
# 3. METRICS
# ----------------------------------------------------------------------

def exact_neighbors(db, queries, k):
    """Ground truth for recall. Computed once, reused for every index."""
    sims = queries @ db.T
    return np.argsort(-sims, axis=1)[:, :k]


def recall_at_k(got, truth):
    """Fraction of the true top-k that the index actually returned.
    This is the accuracy an approximate index trades away for speed."""
    hits = sum(len(set(g) & set(t)) for g, t in zip(got, truth))
    return hits / (len(truth) * truth.shape[1])


def label_accuracy(got, labels):
    """Fraction of queries with at least one correct result in the top-k.
    This is the '97% top-3' style number — it measures whether the SEARCH
    is useful, where recall measures whether the INDEX is faithful."""
    ok = sum(1 for g, lab in zip(got, labels) if set(g) & set(lab))
    return ok / len(labels)


def benchmark(index, db, queries, k, truth, labels=None, warmup=5):
    tracemalloc.start()
    t0 = time.perf_counter()
    index.build(db)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for q in queries[:warmup]:                  # warm caches; first call is always slow
        index.search(q, k)

    results, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        ids = index.search(q, k)
        times.append((time.perf_counter() - t0) * 1000)
        results.append(ids)

    times.sort()
    out = {
        "index": index.name,
        "build_time_s": round(build_time, 3),
        "build_mem_mb": round(peak / 1e6, 1),
        "recall_at_k": round(recall_at_k(results, truth), 4),
        "p50_ms": round(median(times), 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
    }
    if labels is not None:
        out["label_acc_at_k"] = round(label_accuracy(results, labels), 4)
    return out
