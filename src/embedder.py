from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import open_clip
import torch
from PIL import Image


class CLIPEmbedder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
        self.dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode_images(self, paths: Iterable[str | Path], batch_size: int = 32) -> np.ndarray:
        paths = list(paths)
        all_embeds: List[np.ndarray] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"Could not open image {p}: {e}")
                imgs.append(self.preprocess(img))
            batch = torch.stack(imgs).to(self.device)

            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy().astype("float32"))

        return np.concatenate(all_embeds, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")
