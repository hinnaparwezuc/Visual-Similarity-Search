from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np

# src/ holds embedder.py; the repo root holds data/.
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
sys.path.insert(0, str(SOURCE_DIRECTORY))

from embedder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, CLIPEmbedder
from hnsw import HNSW

INDEX_FILENAME = "images.faiss"
METADATA_FILENAME = "metadata.json"
HNSW_FILENAME = "hnsw.pkl"

DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 200

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------

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


def file_signature(path: Path) -> list[int]:
    """Size and modification time: cheap to read, and they change whenever
    the file's contents are replaced. Used to spot edited images on --update
    without hashing every file."""
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _write_atomically(path: Path, write) -> None:
    """Write to a temporary file, then swap it in, so an interrupted run
    never leaves a half-written index behind."""
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


# ----------------------------------------------------------------------
# HNSW
# ----------------------------------------------------------------------

def build_hnsw(
    embeddings: np.ndarray,
    output_directory: Path,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
) -> dict:
    """Build the hand-written HNSW graph and save it next to the FAISS index.

    Returns the settings used, for recording in metadata.json.
    """
    print(f"Building HNSW graph (M={m}, ef_construction={ef_construction})...")
    graph = HNSW(
        dim=embeddings.shape[1],
        M=m,
        ef_construction=ef_construction,
        seed=0,
    )
    graph.add(embeddings, show_progress=True)

    hnsw_path = output_directory / HNSW_FILENAME
    _write_atomically(hnsw_path, graph.save)
    print(f"Saved HNSW index to: {hnsw_path}")

    return {"M": m, "ef_construction": ef_construction}


def _load_existing_graph(
    output_directory: Path, expected_count: int
) -> HNSW | None:
    """Return the saved graph if it matches the previous index, else None."""
    hnsw_path = output_directory / HNSW_FILENAME
    if not hnsw_path.exists():
        return None
    try:
        graph = HNSW.load(hnsw_path)
    except Exception:
        return None
    return graph if graph.ntotal == expected_count else None


def add_hnsw_to_existing(
    output_directory: Path,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
) -> None:
    """Build an HNSW graph from an index that already exists.

    IndexFlatIP stores the raw vectors, so they can be read back out instead
    of running CLIP over every image again.
    """
    index_path = output_directory / INDEX_FILENAME
    metadata_path = output_directory / METADATA_FILENAME

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No existing index in {output_directory}. "
            "Run `python src/indexer.py --hnsw` to build both at once."
        )

    index = faiss.read_index(str(index_path))
    embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["hnsw"] = build_hnsw(embeddings, output_directory, m, ef_construction)
    _write_atomically(
        metadata_path,
        lambda p: p.write_text(json.dumps(metadata, indent=2), encoding="utf-8"),
    )


# ----------------------------------------------------------------------
# Incremental planning
# ----------------------------------------------------------------------

def _load_previous(
    output_directory: Path, model_name: str, pretrained: str
) -> tuple[dict, np.ndarray] | None:
    """Load the previous index for --update, or explain why it can't be used.

    Returns (metadata, vectors), or None when a full rebuild is needed.
    """
    index_path = output_directory / INDEX_FILENAME
    metadata_path = output_directory / METADATA_FILENAME

    if not index_path.exists() or not metadata_path.exists():
        print("No existing index found; building from scratch.")
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if "signatures" not in metadata:
        print("The existing index predates --update support; "
              "rebuilding from scratch once.")
        return None

    if (metadata.get("model_name"), metadata.get("pretrained")) != (
        model_name, pretrained
    ):
        print(f"The existing index was built with "
              f"{metadata.get('model_name')} ({metadata.get('pretrained')}); "
              "rebuilding from scratch for the new model.")
        return None

    index = faiss.read_index(str(index_path))
    if index.ntotal != len(metadata["images"]):
        print("The existing index and metadata disagree; "
              "rebuilding from scratch.")
        return None

    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    return metadata, vectors


def _plan_update(
    image_paths: list[Path], previous: dict
) -> tuple[list[int], list[Path]]:
    """Split the current images into rows to keep and files to embed.

    Returns (kept_rows, to_embed): row numbers from the previous index whose
    files are unchanged, and paths that are new or were modified.
    """
    previous_rows = {
        stored: (row, signature)
        for row, (stored, signature) in enumerate(
            zip(previous["images"], previous["signatures"])
        )
    }

    kept_rows: list[int] = []
    to_embed: list[Path] = []

    for path in image_paths:
        entry = previous_rows.get(to_stored_path(path))
        if entry is not None and entry[1] == file_signature(path):
            kept_rows.append(entry[0])
        else:
            to_embed.append(path)

    # Keep rows in their original order. If nothing was removed or changed,
    # the old rows then come first, unchanged, which lets the HNSW graph be
    # extended rather than rebuilt.
    kept_rows.sort()
    return kept_rows, to_embed


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------

def build_index(
    image_directory: Path,
    output_directory: Path,
    batch_size: int = 32,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    with_hnsw: bool = False,
    hnsw_m: int | None = None,
    hnsw_ef_construction: int | None = None,
    update: bool = False,
) -> dict:
    """Generate image embeddings and save them in a FAISS index.

    With update=True, only images that are new or have changed since the
    last build are embedded; deleted images are dropped. An existing HNSW
    graph is kept in sync automatically.

    Returns counts of what happened: added, removed, unchanged, skipped.
    """
    image_paths = find_images(image_directory)
    print(f"Found {len(image_paths)} images.")

    previous = _load_previous(output_directory, model_name, pretrained) if update else None

    if previous is not None:
        previous_metadata, previous_vectors = previous
        previous_count = len(previous_metadata["images"])
        kept_rows, to_embed = _plan_update(image_paths, previous_metadata)
        removed_count = previous_count - len(kept_rows)
    else:
        previous_metadata, previous_vectors = None, None
        previous_count = 0
        kept_rows, to_embed = [], image_paths
        removed_count = 0

    print(f"Unchanged: {len(kept_rows)}  To embed: {len(to_embed)}  "
          f"Removed: {removed_count}")

    # ---- embed only what's needed -----------------------------------
    new_vectors = np.empty((0, 0), dtype=np.float32)
    new_paths: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    if to_embed:
        print(f"Loading CLIP model {model_name} ({pretrained})...")
        embedder = CLIPEmbedder(model_name=model_name, pretrained=pretrained)
        new_vectors, new_paths, skipped = embedder.encode_image_files(
            to_embed, batch_size=batch_size
        )

    if skipped:
        print(f"Skipped {len(skipped)} unreadable image(s):")
        for path, error in skipped:
            print(f"  - {path}: {error}")

    summary = {
        "added": len(new_paths),
        "removed": removed_count,
        "unchanged": len(kept_rows),
        "skipped": len(skipped),
    }

    # ---- HNSW settings ----------------------------------------------
    previous_hnsw = (previous_metadata or {}).get("hnsw", {})
    hnsw_m = hnsw_m or previous_hnsw.get("M", DEFAULT_HNSW_M)
    hnsw_ef_construction = hnsw_ef_construction or previous_hnsw.get(
        "ef_construction", DEFAULT_HNSW_EF_CONSTRUCTION
    )
    hnsw_path = output_directory / HNSW_FILENAME
    existing_graph = (
        _load_existing_graph(output_directory, previous_count)
        if previous is not None else None
    )

    nothing_changed = not new_paths and removed_count == 0
    if previous is not None and nothing_changed and not (
        with_hnsw and existing_graph is None
    ):
        print("Index is already up to date.")
        return summary

    # ---- combine and write the flat index ----------------------------
    if previous is not None:
        kept_vectors = previous_vectors[kept_rows]
        kept_images = [previous_metadata["images"][r] for r in kept_rows]
        kept_signatures = [previous_metadata["signatures"][r] for r in kept_rows]
    else:
        kept_vectors = np.empty((0, 0), dtype=np.float32)
        kept_images, kept_signatures = [], []

    parts = [v for v in (kept_vectors, new_vectors) if v.size]
    if not parts:
        raise ValueError("None of the images could be read; no index built.")

    embeddings = np.ascontiguousarray(np.vstack(parts), dtype=np.float32)
    images = kept_images + [to_stored_path(p) for p in new_paths]
    signatures = kept_signatures + [file_signature(p) for p in new_paths]

    # CLIPEmbedder already normalizes the vectors. Inner-product search
    # therefore produces cosine-similarity rankings.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_directory.mkdir(parents=True, exist_ok=True)
    index_path = output_directory / INDEX_FILENAME
    metadata_path = output_directory / METADATA_FILENAME

    _write_atomically(index_path, lambda p: faiss.write_index(index, str(p)))

    metadata = {
        "model_name": model_name,
        "pretrained": pretrained,
        "image_count": len(images),
        "embedding_dimension": int(embeddings.shape[1]),
        "images": images,
        "signatures": signatures,
    }

    # ---- keep the HNSW graph in sync ---------------------------------
    keep_graph_in_sync = with_hnsw or existing_graph is not None
    settings_match = (
        existing_graph is not None
        and existing_graph.M == hnsw_m
        and existing_graph.ef_construction == hnsw_ef_construction
    )

    if keep_graph_in_sync:
        if existing_graph is not None and removed_count == 0 and settings_match:
            # Pure additions: append to the graph instead of rebuilding.
            if new_paths:
                print(f"Adding {len(new_paths)} image(s) to the HNSW graph...")
                existing_graph.add(new_vectors, show_progress=True)
                _write_atomically(hnsw_path, existing_graph.save)
                print(f"Saved HNSW index to: {hnsw_path}")
            metadata["hnsw"] = {"M": hnsw_m, "ef_construction": hnsw_ef_construction}
        else:
            # HNSW graphs don't support deleting nodes, so any removal or
            # change means rebuilding from the stored vectors (no CLIP).
            metadata["hnsw"] = build_hnsw(
                embeddings, output_directory, hnsw_m, hnsw_ef_construction
            )
    elif hnsw_path.exists():
        # An old graph would describe a different set of images.
        hnsw_path.unlink()
        print("Removed the old HNSW index, which no longer matches. "
              "Re-run with --hnsw to rebuild it.")

    _write_atomically(
        metadata_path,
        lambda p: p.write_text(json.dumps(metadata, indent=2), encoding="utf-8"),
    )

    print(f"Indexed {len(images)} images "
          f"({summary['added']} added, {summary['removed']} removed).")
    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved image metadata to: {metadata_path}")
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--update",
        action="store_true",
        help="Only embed new or changed images and drop deleted ones, "
             "instead of rebuilding everything.",
    )
    hnsw_group = parser.add_mutually_exclusive_group()
    hnsw_group.add_argument(
        "--hnsw",
        action="store_true",
        help="Also build an HNSW index for approximate search.",
    )
    hnsw_group.add_argument(
        "--hnsw-only",
        action="store_true",
        help="Add an HNSW index to the existing index without re-embedding.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=None,
        help=f"HNSW neighbours per node, layer 0 keeps twice this "
             f"(default {DEFAULT_HNSW_M}).",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=None,
        help=f"HNSW candidate pool size while building "
             f"(default {DEFAULT_HNSW_EF_CONSTRUCTION}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)

    if args.hnsw_only:
        add_hnsw_to_existing(
            output_directory=args.output,
            m=args.hnsw_m or DEFAULT_HNSW_M,
            ef_construction=args.hnsw_ef_construction or DEFAULT_HNSW_EF_CONSTRUCTION,
        )
    else:
        build_index(
            image_directory=args.images,
            output_directory=args.output,
            batch_size=args.batch_size,
            model_name=args.model,
            pretrained=args.pretrained,
            with_hnsw=args.hnsw,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            update=args.update,
        )


if __name__ == "__main__":
    main()
