"""Find groups of near-duplicate images in an existing index.

Every image is compared against every other with FAISS range search, which
returns all pairs above a similarity threshold. Pairs are then merged into
groups with union-find, so if A matches B and B matches C, all three end up
in one group even when A and C fall slightly below the threshold.

No CLIP model is needed: the vectors are read back out of the index.

    python src/duplicates.py
    python src/duplicates.py --threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent

DEFAULT_THRESHOLD = 0.95


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            # Path halving keeps the trees shallow.
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def find_duplicate_groups(
    index: faiss.Index,
    image_paths: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 1024,
) -> list[dict]:
    """Group images whose cosine similarity is at least `threshold`.

    Returns a list of groups, largest first. Each group is
        {"images": [{"image_path": ..., "similarity_score": ...}, ...]}
    where the first image is the group's reference and each score is that
    image's similarity to the reference (1.0 for the reference itself).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if index.ntotal != len(image_paths):
        raise ValueError("The index and image list have different lengths.")

    count = index.ntotal
    if count < 2:
        return []

    vectors = index.reconstruct_n(0, count).astype(np.float32)
    groups = _UnionFind(count)
    matched = np.zeros(count, dtype=bool)

    # range_search returns every neighbour with similarity strictly above
    # the radius. Nudge the radius down so a score exactly equal to the
    # threshold (e.g. 1.0 for identical images) still counts.
    radius = threshold - 1e-6

    for start in range(0, count, batch_size):
        batch = vectors[start : start + batch_size]
        limits, _, neighbours = index.range_search(batch, radius)

        for offset in range(len(batch)):
            query = start + offset
            for neighbour in neighbours[limits[offset] : limits[offset + 1]]:
                if neighbour != query:
                    groups.union(query, int(neighbour))
                    matched[query] = matched[neighbour] = True

    members: dict[int, list[int]] = {}
    for item in np.flatnonzero(matched):
        members.setdefault(groups.find(int(item)), []).append(int(item))

    results = []
    for rows in members.values():
        reference = rows[0]
        scores = vectors[rows] @ vectors[reference]
        results.append(
            {
                "images": [
                    {
                        "image_path": image_paths[row],
                        "similarity_score": round(float(score), 4),
                    }
                    for row, score in zip(rows, scores)
                ]
            }
        )

    results.sort(key=lambda group: (-len(group["images"]),
                                    group["images"][0]["image_path"]))
    return results


def load_index(index_directory: Path) -> tuple[faiss.Index, list[str]]:
    index_path = index_directory / "images.faiss"
    metadata_path = index_directory / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Search index not found. Run `python src/indexer.py` first."
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata["images"]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find groups of near-duplicate images in the index."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity for two images to count as "
             f"duplicates (default {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        default=PROJECT_ROOT / "data" / "index",
        help="Directory containing the FAISS index and metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    index, image_paths = load_index(args.index_directory)
    groups = find_duplicate_groups(index, image_paths, args.threshold)

    print(json.dumps(groups, indent=2))
    duplicates = sum(len(g["images"]) - 1 for g in groups)
    print(
        f"{len(groups)} group(s); {duplicates} image(s) could be removed "
        "while keeping one of each.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
