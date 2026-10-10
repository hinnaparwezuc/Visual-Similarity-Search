from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
sys.path.insert(0, str(SOURCE_DIRECTORY))

from embedder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, CLIPEmbedder
from hnsw import HNSW

BACKENDS = ("flat", "hnsw")
HNSW_FILENAME = "hnsw.pkl"


class ImageSearchEngine:
    """Search an image index by text or by example image.

    Two backends are available per query:
      "flat"  exact search with FAISS IndexFlatIP (always available)
      "hnsw"  approximate search with the graph in hnsw.py (available when
              the index was built with --hnsw or --hnsw-only)
    """

    def __init__(self, index_directory: Path):
        index_path = index_directory / "images.faiss"
        metadata_path = index_directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Search index not found. Run `python src/indexer.py` first."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.image_paths = self.metadata["images"]

        if self.index.ntotal != len(self.image_paths):
            raise ValueError(
                "The FAISS index and image metadata contain different "
                "numbers of images."
            )

        # Queries must be embedded with the same model that built the index,
        # otherwise the vectors live in different spaces.
        self.embedder = CLIPEmbedder(
            model_name=self.metadata.get("model_name", DEFAULT_MODEL_NAME),
            pretrained=self.metadata.get("pretrained", DEFAULT_PRETRAINED),
        )

        if self.embedder.dim != self.index.d:
            raise ValueError(
                f"Model produces {self.embedder.dim}-dimensional vectors but "
                f"the index holds {self.index.d}-dimensional vectors. "
                "Rebuild the index with `python src/indexer.py`."
            )

        # The HNSW graph is optional. If it's missing or out of date, exact
        # search still works; asking for HNSW explains how to build it.
        self.hnsw: HNSW | None = None
        self.hnsw_unavailable_reason = (
            "No HNSW index found. Build one with "
            "`python src/indexer.py --hnsw-only`."
        )
        hnsw_path = index_directory / HNSW_FILENAME
        if hnsw_path.exists():
            # hnsw.pkl is a pickle: only load files you built yourself.
            graph = HNSW.load(hnsw_path)
            if graph.ntotal == self.index.ntotal and graph.dim == self.index.d:
                self.hnsw = graph
            else:
                self.hnsw_unavailable_reason = (
                    f"The HNSW index holds {graph.ntotal} images but the main "
                    f"index holds {self.index.ntotal}, so it is out of date. "
                    "Rebuild it with `python src/indexer.py --hnsw-only`."
                )

        # FAISS's first search pays one-off setup costs (thread pool start,
        # etc.). Run one throwaway query now so the first timing reported
        # to the user reflects the real lookup cost.
        warmup = np.zeros((1, self.index.d), dtype=np.float32)
        warmup[0, 0] = 1.0
        self.index.search(warmup, 1)
        if self.hnsw is not None:
            self.hnsw.search(warmup[0], 1)

        self.last_search_ms = 0.0

    @property
    def has_hnsw(self) -> bool:
        return self.hnsw is not None

    def _resolve(self, stored_path: str) -> Path:
        path = Path(stored_path)
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        backend: str = "flat",
        ef_search: int | None = None,
    ) -> list[dict]:
        """Find images that are semantically similar to a text query."""
        query_embedding = self.embedder.encode_text([query])
        return self._search(query_embedding, top_k, backend, ef_search)

    def search_image(
        self,
        image_path: Path,
        top_k: int = 5,
        exclude_self: bool = True,
        backend: str = "flat",
        ef_search: int | None = None,
        exclude_above: float | None = None,
    ) -> list[dict]:
        """Find images that are visually similar to a query image.

        exclude_self   leave out the query file itself if it is indexed
                       (it would always be the top match with score 1.0)
        exclude_above  leave out every result scoring at or above this, e.g.
                       0.999 to hide identical copies of an uploaded image

        Filtered results are replaced, so up to top_k are still returned
        however many are left out.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        query_embedding = self.embedder.encode_images([image_path])
        query_resolved = image_path.resolve()

        def keep(result: dict) -> bool:
            if exclude_above is not None and result["similarity_score"] >= exclude_above:
                return False
            if exclude_self and self._resolve(result["image_path"]) == query_resolved:
                return False
            return True

        # Ask for a few extra, and widen the search until enough survive the
        # filters. The image is embedded only once; each retry is just
        # another index lookup.
        fetch = top_k + 1
        total_ms = 0.0
        while True:
            results = self._search(query_embedding, fetch, backend, ef_search)
            total_ms += self.last_search_ms
            kept = [r for r in results if keep(r)]
            if len(kept) >= top_k or fetch >= self.index.ntotal:
                break
            fetch = min(self.index.ntotal, fetch * 2)

        self.last_search_ms = total_ms
        return kept[:top_k]

    def _search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        backend: str = "flat",
        ef_search: int | None = None,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}; choose one of {BACKENDS}."
            )

        top_k = min(top_k, self.index.ntotal)
        query_embedding = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        )

        # Only the index lookup is timed, not CLIP encoding, so the number
        # compares the two backends rather than the model.
        start = time.perf_counter()

        if backend == "hnsw":
            if self.hnsw is None:
                raise ValueError(self.hnsw_unavailable_reason)
            if ef_search is not None:
                self.hnsw.ef_search = ef_search
            query = query_embedding[0]
            ids = self.hnsw.search(query, top_k)
            # The graph returns ids only; vectors are unit length, so the
            # dot product is the same cosine score the flat index reports.
            scores = self.hnsw.vectors[ids] @ query
            pairs = zip(scores, ids)
        else:
            scores, indices = self.index.search(query_embedding, top_k)
            pairs = zip(scores[0], indices[0])

        self.last_search_ms = (time.perf_counter() - start) * 1000

        results = []

        for score, index_position in pairs:
            if index_position == -1:
                continue

            results.append(
                {
                    "image_path": self.image_paths[index_position],
                    "similarity_score": round(float(score), 4),
                }
            )

        return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for similar images using CLIP and FAISS."
    )

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--text",
        type=str,
        help="Text description used to search for images.",
    )
    query_group.add_argument(
        "--image",
        type=Path,
        help="Image used to search for visually similar images.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return.",
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        default=PROJECT_ROOT / "data" / "index",
        help="Directory containing the FAISS index and metadata.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="flat",
        help="flat = exact FAISS search; hnsw = approximate search with "
             "src/hnsw.py (build it with `indexer.py --hnsw`).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="HNSW candidate pool size: higher is more accurate but slower.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="With --image, keep the query image in results if it is indexed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    search_engine = ImageSearchEngine(args.index_directory)

    if args.text is not None:
        results = search_engine.search_text(
            args.text, args.top_k, args.backend, args.ef_search
        )
    else:
        results = search_engine.search_image(
            args.image,
            args.top_k,
            exclude_self=not args.include_self,
            backend=args.backend,
            ef_search=args.ef_search,
        )

    print(json.dumps(results, indent=2))
    print(
        f"{args.backend} index lookup: {search_engine.last_search_ms:.2f} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
