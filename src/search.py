from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedder import CLIPEmbedder


class ImageSearchEngine:
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
        self.embedder = CLIPEmbedder()

        if self.index.ntotal != len(self.image_paths):
            raise ValueError(
                "The FAISS index and image metadata contain different "
                "numbers of images."
            )

    def search_text(self, query: str, top_k: int = 5) -> list[dict]:
        """Find images that are semantically similar to a text query."""
        query_embedding = self.embedder.encode_text([query])
        return self._search(query_embedding, top_k)

    def search_image(
        self,
        image_path: Path,
        top_k: int = 5,
    ) -> list[dict]:
        """Find images that are visually similar to a query image."""
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        query_embedding = self.embedder.encode_images([image_path])
        return self._search(query_embedding, top_k)

    def _search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        top_k = min(top_k, self.index.ntotal)
        query_embedding = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        )

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    search_engine = ImageSearchEngine(args.index_directory)

    if args.text:
        results = search_engine.search_text(args.text, args.top_k)
    else:
        results = search_engine.search_image(args.image, args.top_k)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
