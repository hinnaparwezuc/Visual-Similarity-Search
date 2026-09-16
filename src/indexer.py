from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

# Allows this file to import embedder.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedder import CLIPEmbedder


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(image_directory: Path) -> list[Path]:
    """Return all supported image files inside a directory."""
    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_directory}"
        )

    image_paths = sorted(
        path
        for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No supported images were found in {image_directory}"
        )

    return image_paths


def build_index(
    image_directory: Path,
    output_directory: Path,
    batch_size: int = 32,
) -> None:
    """Generate image embeddings and save them in a FAISS index."""
    image_paths = find_images(image_directory)

    print(f"Found {len(image_paths)} images.")
    print("Loading CLIP model...")

    embedder = CLIPEmbedder()

    print("Generating image embeddings...")
    embeddings = embedder.encode_images(
        image_paths,
        batch_size=batch_size,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # CLIPEmbedder already normalizes the vectors. Inner-product search
    # therefore produces cosine-similarity rankings.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_directory.mkdir(parents=True, exist_ok=True)

    index_path = output_directory / "images.faiss"
    metadata_path = output_directory / "metadata.json"

    faiss.write_index(index, str(index_path))

    metadata = {
        "image_count": len(image_paths),
        "embedding_dimension": int(embeddings.shape[1]),
        "images": [
            str(path.resolve().relative_to(PROJECT_ROOT))
            if path.resolve().is_relative_to(PROJECT_ROOT)
            else str(path.resolve())
            for path in image_paths
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved image metadata to: {metadata_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from CLIP image embeddings."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data" / "images",
        help="Directory containing the source images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "index",
        help="Directory where the index and metadata will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images processed in each batch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    build_index(
        image_directory=args.images,
        output_directory=args.output,
        batch_size=args.batch_size,
    )
