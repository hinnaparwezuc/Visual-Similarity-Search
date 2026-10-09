"""A stand-in for src/embedder.py so the tests run without PyTorch or CLIP.

Images are embedded from their actual pixels (an 8x8 grayscale thumbnail,
mean-centred and L2-normalized), so identical files get identical vectors
and lightly edited copies get very similar ones. That is enough to test the
indexing, search and duplicate logic end to end. Text is embedded from a
hash, so the same query always gives the same vector.

The public interface mirrors CLIPEmbedder exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DIMENSION = 64


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _embed_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        thumbnail = image.convert("L").resize((8, 8), Image.BILINEAR)
    return _normalize(np.asarray(thumbnail, dtype=np.float32).reshape(-1))


class CLIPEmbedder:
    # Counts every image embedded, across instances, so tests can check
    # that --update only embeds what changed.
    images_embedded = 0

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cpu"
        self.dim = DIMENSION

    def encode_images(
        self, paths: Iterable[str | Path], batch_size: int = 32
    ) -> np.ndarray:
        vectors = []
        for path in paths:
            try:
                vectors.append(_embed_image(path))
            except Exception as error:
                raise RuntimeError(f"Could not open image {path}: {error}") from error
        CLIPEmbedder.images_embedded += len(vectors)
        return np.stack(vectors)

    def encode_image_files(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[Path], list[tuple[Path, str]]]:
        vectors, kept, skipped = [], [], []
        for path in map(Path, paths):
            try:
                vectors.append(_embed_image(path))
                kept.append(path)
            except Exception as error:
                skipped.append((path, str(error)))
        CLIPEmbedder.images_embedded += len(vectors)
        if not vectors:
            return np.empty((0, self.dim), dtype=np.float32), kept, skipped
        return np.stack(vectors), kept, skipped

    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors.append(_normalize(rng.normal(size=self.dim)))
        return np.stack(vectors)
