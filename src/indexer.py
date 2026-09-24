from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

# src/ holds embedder.py; the repo root holds data/.
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
sys.path.insert(0, str(SOURCE_DIRECTORY))

from embedder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, CLIPEmbedder


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


def to_stored_path(path: Path) -> str:
    """Store paths relative to the project root when possible."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def build_index(
    image_directory: Path,
    output_directory: Path,
    batch_size: int = 32,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
) -> None:
    """Generate image embeddings and save them in a FAISS index."""
    image_paths = find_images(image_directory)

    print(f"Found {len(image_paths)} images.")
    print(f"Loading CLIP model {model_name} ({pretrained})...")

    embedder = CLIPEmbedder(model_name=model_name, pretrained=pretrained)

    embeddings, kept_paths, skipped = embedder.encode_image_files(
        image_paths,
        batch_size=batch_size,
    )

    if skipped:
        print(f"Skipped {len(skipped)} unreadable image(s):")
        for path, error in skipped:
            print(f"  - {path}: {error}")

    if not kept_paths:
        raise ValueError("None of the images could be read; no index built.")

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
        "model_name": model_name,
        "pretrained": pretrained,
        "image_count": len(kept_paths),
        "embedding_dimension": int(embeddings.shape[1]),
        "images": [to_stored_path(path) for path in kept_paths],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Indexed {len(kept_paths)} images.")
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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="OpenCLIP model architecture.",
    )
    parser.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help="OpenCLIP pretrained weights tag.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    build_index(
        image_directory=args.images,
        output_directory=args.output,
        batch_size=args.batch_size,
        model_name=args.model,
        pretrained=args.pretrained,
    )
