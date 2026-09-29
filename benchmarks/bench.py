"""
bench.py — measure every index the same way, so the numbers are comparable.

Covers the exact FAISS index this repo builds by default, FAISS's own HNSW,
and the hand-written HNSW in src/hnsw.py.

Reports per index:
  build_time_s     time to construct the index from embeddings
  index_mb         serialized size of the finished index
  recall_at_k      agreement with exact search — index fidelity
  label_acc_at_k   queries with a same-folder hit in the top-k — search
                   usefulness (only when images sit in class subfolders)
  p50_ms / p95_ms  per-query search latency

USAGE
  python src/indexer.py                  # build data/index first
  python benchmarks/bench.py
  python benchmarks/bench.py --queries 500 --k 3

Queries are drawn from the indexed set and evaluated leave-one-out: each
query image is dropped from its own results, matching the exclude_self
behaviour in search.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import pickle
import time
from pathlib import Path
from statistics import median

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hnsw import HNSW  # noqa: E402

DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------

def load_data(index_dir: Path, n_queries: int, seed: int = 0):
    """Read embeddings back out of the index this repo already builds.

    IndexFlatIP stores raw vectors, so reconstruct_n returns them without
    re-running CLIP and without keeping a second copy on disk to drift.

    Returns (db, query_ids, labels):
      db        (N, D) float32, L2-normalized by CLIPEmbedder
      query_ids indices into db used as queries
      labels    parent folder name per image, or None if all images are
                in one flat directory
    """
    metadata_path = index_dir / "metadata.json"
    flat_path = index_dir / "images.faiss"

    if not flat_path.exists():
        raise FileNotFoundError(
            f"No flat index at {flat_path}. The benchmark reads its vectors "
            "from the exact index, so build that one first:\n"
            "    python src/indexer.py"
        )

    index = faiss.read_index(str(flat_path))
    db = index.reconstruct_n(0, index.ntotal).astype("float32")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_paths = metadata["images"]

    # Per-class subdirectories give free ground-truth labels:
    # data/images/dogs/x.jpg -> "dogs". A flat folder gives nothing to
    # measure, so label accuracy is skipped rather than faked.
    folders = [Path(p).parent.name for p in image_paths]
    labels = folders if len(set(folders)) > 1 else None

    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, len(db))
    query_ids = rng.choice(len(db), size=n_queries, replace=False)

    return db, query_ids, labels


# ----------------------------------------------------------------------
# INDEXES — each needs .name, .build(db), .search(vec, k) -> ids
# ----------------------------------------------------------------------

class NumpyFlat:
    """Exact search in plain NumPy: the correctness oracle, and a reference
    for how much of FAISS's speed is implementation rather than algorithm."""

    name = "numpy flat (exact)"

    def build(self, db):
        self.db = db

    def size_bytes(self):
        return self.db.nbytes

    def search(self, q, k):
        k = min(k, len(self.db) - 1)            # argpartition needs k < N
        sims = self.db @ q                      # cosine: rows are unit norm
        top = np.argpartition(-sims, k)[:k]     # partial select beats a sort
        return top[np.argsort(-sims[top])]


class FaissFlat:
    """What this repo builds by default: exhaustive inner-product search."""

    name = "faiss IndexFlatIP"

    def build(self, db):
        self.index = faiss.IndexFlatIP(db.shape[1])
        self.index.add(db)

    def size_bytes(self):
        return faiss.serialize_index(self.index).nbytes

    def search(self, q, k):
        _, ids = self.index.search(q.reshape(1, -1), k)
        return ids[0]


class FaissHNSW:
    """FAISS's HNSW — the reference the hand-written one is judged against."""

    def __init__(self, M=16, ef_construction=200, ef_search=64):
        self.M, self.efc, self.efs = M, ef_construction, ef_search
        self.name = f"faiss HNSW (efS={ef_search})"

    def build(self, db):
        self.index = faiss.IndexHNSWFlat(db.shape[1], self.M,
                                         faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.efc
        self.index.add(db)
        self.index.hnsw.efSearch = self.efs

    def size_bytes(self):
        return faiss.serialize_index(self.index).nbytes

    def search(self, q, k):
        _, ids = self.index.search(q.reshape(1, -1), k)
        return ids[0]


class MyHNSW:
    """The implementation in src/hnsw.py."""

    def __init__(self, M=16, ef_construction=200, ef_search=64,
                 prebuilt=None, prebuilt_build_time=None):
        self.M, self.efc, self.efs = M, ef_construction, ef_search
        self.name = f"hnsw.py (efS={ef_search})"
        self._prebuilt = prebuilt      # reuse a graph across efSearch values
        # When the graph is shared, build() below takes ~0s, so report the
        # time the shared build actually took instead.
        self.build_time_override = prebuilt_build_time

    def build(self, db):
        if self._prebuilt is not None:
            self.index = self._prebuilt
        else:
            self.index = HNSW(dim=db.shape[1], M=self.M,
                              ef_construction=self.efc, seed=0)
            self.index.add(db, show_progress=False)
        self.index.ef_search = self.efs

    def size_bytes(self):
        # Pickled size. The in-memory size is larger, since the graph is
        # Python dicts and lists, but this is comparable to FAISS's file size.
        return len(pickle.dumps(
            {"vectors": self.index.vectors, "layers": self.index.layers},
            protocol=pickle.HIGHEST_PROTOCOL,
        ))

    def search(self, q, k):
        return self.index.search(q, k)


# ----------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------

def exact_neighbors(db, query_ids, k):
    """Ground truth for recall, computed once and shared by every index.
    Each query's own row is excluded, as search.py does."""
    sims = db[query_ids] @ db.T
    sims[np.arange(len(query_ids)), query_ids] = -np.inf
    return np.argsort(-sims, axis=1)[:, :k]


def recall_at_k(got, truth):
    """Share of the true top-k the index actually returned. This is the
    accuracy an approximate index trades away for speed."""
    hits = sum(len(set(g) & set(t)) for g, t in zip(got, truth))
    return hits / (len(truth) * truth.shape[1])


def label_accuracy(got, query_ids, labels):
    """Share of queries with at least one same-class image in the top-k.
    Recall asks whether the index is faithful to exact search; this asks
    whether the search is useful. Two different numbers, both worth having."""
    ok = sum(
        1 for ids, qid in zip(got, query_ids)
        if any(labels[i] == labels[qid] for i in ids if i != qid and i >= 0)
    )
    return ok / len(query_ids)


def benchmark(index, db, query_ids, k, truth, labels=None, warmup=5):
    # tracemalloc was used here before, but it only sees Python-side
    # allocations: FAISS builds in C++, so every FAISS row read 0 MB. The
    # finished index's serialized size is measured instead, the same way
    # for every index.
    t0 = time.perf_counter()
    index.build(db)
    build_time = time.perf_counter() - t0
    override = getattr(index, "build_time_override", None)
    if override is not None:
        build_time = override

    # Ask for k+1, then drop the query itself: it is in the index and would
    # otherwise always take the first slot at similarity 1.0.
    kk = k + 1

    for qid in query_ids[:warmup]:              # warm caches; first call lies
        index.search(db[qid], kk)

    results, times = [], []
    for qid in query_ids:
        q = db[qid]
        t0 = time.perf_counter()
        ids = index.search(q, kk)
        times.append((time.perf_counter() - t0) * 1000)
        # FAISS pads with -1 when it finds fewer than kk neighbours.
        results.append([i for i in ids if i != qid and i >= 0][:k])

    times.sort()
    out = {
        "index": index.name,
        "build_time_s": round(build_time, 3),
        "index_mb": round(index.size_bytes() / 1e6, 2),
        "recall_at_k": round(recall_at_k(results, truth), 4),
        "p50_ms": round(median(times), 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
    }
    if labels is not None:
        out["label_acc_at_k"] = round(
            label_accuracy(results, query_ids, labels), 4
        )
    return out


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------

def print_table(rows):
    cols = ["index", "build_time_s", "index_mb", "recall_at_k",
            "label_acc_at_k", "p50_ms", "p95_ms"]
    cols = [c for c in cols if any(c in r for r in rows)]
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows))
             for c in cols}
    print(" | ".join(c.ljust(width[c]) for c in cols))
    print("-+-".join("-" * width[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark exact and approximate indexes on the same data."
    )
    ap.add_argument("--index-directory", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 64, 256],
                    help="efSearch values to sweep for both HNSW indexes.")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results.json")
    ap.add_argument("--threads", type=int, default=1,
                    help="FAISS threads. Default 1, so FAISS is timed on one "
                         "core like the NumPy and hnsw.py indexes.")
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads)

    db, query_ids, labels = load_data(args.index_directory, args.queries)
    print(f"db={db.shape}  queries={len(query_ids)}  k={args.k}  "
          f"labels={'yes' if labels else 'no (flat image folder)'}\n")

    truth = exact_neighbors(db, query_ids, args.k)

    indexes = [NumpyFlat(), FaissFlat()]
    for ef in args.ef_search:
        indexes.append(FaissHNSW(ef_search=ef))

    # Build the hand-written graph once and reuse it across the efSearch
    # sweep, since efSearch is a query-time parameter. Rebuilding per value
    # would only measure the same build repeatedly.
    print("building hnsw.py graph once for the efSearch sweep...")
    shared = HNSW(dim=db.shape[1], M=16, ef_construction=200, seed=0)
    t0 = time.perf_counter()
    shared.add(db, show_progress=True)
    shared_build_time = time.perf_counter() - t0
    print(f"graph: {shared.stats()}\n")
    for ef in args.ef_search:
        indexes.append(MyHNSW(ef_search=ef, prebuilt=shared,
                              prebuilt_build_time=shared_build_time))

    rows = [benchmark(i, db, query_ids, args.k, truth, labels) for i in indexes]

    print_table(rows)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
