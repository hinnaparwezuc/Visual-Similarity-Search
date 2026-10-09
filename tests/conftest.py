"""Shared test setup.

The real embedder needs PyTorch and downloads CLIP weights, which is slow
and unnecessary for testing the indexing and search logic. Registering the
fake under the name `embedder` before any src module is imported means
`from embedder import CLIPEmbedder` picks up the fake instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TESTS_DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = TESTS_DIRECTORY.parent / "src"

sys.path.insert(0, str(SOURCE_DIRECTORY))
sys.path.insert(0, str(TESTS_DIRECTORY))

import fake_embedder  # noqa: E402

sys.modules["embedder"] = fake_embedder


def make_image(path: Path, seed: int, size: int = 32) -> Path:
    """Write a random-noise image. Different seeds give unrelated images."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


def make_near_copy(source: Path, destination: Path, noise: int = 6) -> Path:
    """Write a lightly altered copy, like a re-saved or edited export."""
    pixels = np.asarray(Image.open(source).convert("RGB"), dtype=np.int16)
    rng = np.random.default_rng(12345)
    pixels = np.clip(pixels + rng.integers(-noise, noise + 1, pixels.shape), 0, 255)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(destination)
    return destination


@pytest.fixture
def image_directory(tmp_path: Path) -> Path:
    """Ten unrelated images across two subfolders."""
    directory = tmp_path / "images"
    for number in range(10):
        folder = "cats" if number % 2 else "dogs"
        make_image(directory / folder / f"image{number}.png", seed=number)
    return directory


@pytest.fixture
def index_directory(tmp_path: Path) -> Path:
    return tmp_path / "index"


@pytest.fixture(autouse=True)
def reset_embed_counter():
    fake_embedder.CLIPEmbedder.images_embedded = 0
    yield
