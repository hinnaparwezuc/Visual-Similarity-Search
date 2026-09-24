from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class CLIPEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
        self.dim = self.model.visual.output_dim

    @torch.no_grad()
    def _embed_tensors(self, tensors: List[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(tensors).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def _load(self, path: str | Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.preprocess(image)

    def encode_images(
        self, paths: Iterable[str | Path], batch_size: int = 32
    ) -> np.ndarray:
        """Embed images strictly: any unreadable file raises an error.

        Used for query images, where a bad file should be reported.
        """
        paths = list(paths)
        all_embeds: List[np.ndarray] = []

        for i in range(0, len(paths), batch_size):
            tensors = []
            for p in paths[i : i + batch_size]:
                try:
                    tensors.append(self._load(p))
                except Exception as e:
                    raise RuntimeError(f"Could not open image {p}: {e}") from e
            all_embeds.append(self._embed_tensors(tensors))

        return np.concatenate(all_embeds, axis=0)

    def encode_image_files(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[Path], list[tuple[Path, str]]]:
        """Embed a collection of images, skipping any that can't be read.

        Returns (embeddings, kept_paths, skipped) where `skipped` is a list
        of (path, error message). Row i of `embeddings` belongs to
        kept_paths[i].
        """
        paths = [Path(p) for p in paths]
        all_embeds: List[np.ndarray] = []
        kept: list[Path] = []
        skipped: list[tuple[Path, str]] = []

        batches = range(0, len(paths), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Embedding", unit="batch")

        for i in batches:
            tensors = []
            for p in paths[i : i + batch_size]:
                try:
                    tensors.append(self._load(p))
                    kept.append(p)
                except Exception as e:
                    skipped.append((p, str(e)))
            if tensors:
                all_embeds.append(self._embed_tensors(tensors))

        if not all_embeds:
            return np.empty((0, self.dim), dtype="float32"), kept, skipped

        return np.concatenate(all_embeds, axis=0), kept, skipped

    @torch.no_grad()
    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")
