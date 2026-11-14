"""CLIP image and text embeddings, cached on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def load_clip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.eval().to(device), preprocess, tokenizer


def embed_images(
    image_names: pd.Series, images_dir: Path, cache_path: Path, clip, batch_size: int, device: str
) -> pd.DataFrame:
    """L2-normalised image embeddings indexed by image name (`Style/painting.jpg`).

    Only names missing from the cache are embedded; the cache is updated in place.
    Names with no file in `images_dir` are skipped.
    """
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    names = [
        n for n in pd.unique(image_names) if n not in cached.index and (images_dir / n).exists()
    ]
    if names:
        model, preprocess, _ = clip
        chunks = []
        with torch.no_grad():
            for i in tqdm(range(0, len(names), batch_size), desc="embedding images"):
                batch = names[i : i + batch_size]
                imgs = [preprocess(Image.open(images_dir / n).convert("RGB")) for n in batch]
                emb = model.encode_image(torch.stack(imgs).to(device)).float()
                chunks.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        new = pd.DataFrame(np.vstack(chunks), index=pd.Index(names, name="image_name"))
        new.columns = new.columns.astype(str)
        cached = pd.concat([cached, new])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path)
    return cached.loc[cached.index.intersection(pd.unique(image_names))]


def embed_texts(texts: list[str], clip, device: str) -> np.ndarray:
    """L2-normalised text embeddings, one row per text."""
    model, _, tokenizer = clip
    with torch.no_grad():
        emb = model.encode_text(tokenizer(texts).to(device)).float()
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


def concept_scores(image_emb: pd.DataFrame, text_emb: np.ndarray, concepts) -> pd.DataFrame:
    """Cosine similarity of every image with every concept, z-scored per concept
    across images (so a score says how much more than the average painting)."""
    sim = image_emb.to_numpy() @ text_emb.T
    sim = (sim - sim.mean(axis=0)) / sim.std(axis=0)
    return pd.DataFrame(sim, index=image_emb.index, columns=list(concepts))
