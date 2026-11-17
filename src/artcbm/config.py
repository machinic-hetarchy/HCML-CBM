"""Load config.yaml into a frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    annotations_csv: Path
    annotations_parquet: Path
    images_dir: Path
    languages: tuple[str, ...]
    emotions: tuple[str, ...]
    min_annotations_per_language: int
    max_paintings: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    pretrained: str
    batch_size: int
    cache_dir: Path
    multilingual_model: str
    multilingual_pretrained: str


@dataclass(frozen=True)
class ConceptsConfig:
    path: Path
    templates: dict[str, str]
    translated: dict[str, Path]


@dataclass(frozen=True)
class AnalysisConfig:
    n_permutations: int
    cv_folds: int
    n_bootstrap: int
    n_concept_components: int
    logreg_C: float
    forward_Cs: list[float]
    min_paintings_per_style: int


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    data: DataConfig
    embedding: EmbeddingConfig
    concepts: ConceptsConfig
    analysis: AnalysisConfig


def _abs(path: str) -> Path:
    """Resolve paths in the config relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    d, e = raw["data"], raw["embedding"]
    return Config(
        seed=raw["seed"],
        device=raw["device"],
        data=DataConfig(
            annotations_csv=_abs(d["annotations_csv"]),
            annotations_parquet=_abs(d["annotations_parquet"]),
            images_dir=_abs(d["images_dir"]),
            languages=tuple(d["languages"]),
            emotions=tuple(d["emotions"]),
            min_annotations_per_language=d["min_annotations_per_language"],
            max_paintings=d["max_paintings"],
        ),
        embedding=EmbeddingConfig(
            model=e["model"],
            pretrained=e["pretrained"],
            batch_size=e["batch_size"],
            cache_dir=_abs(e["cache_dir"]),
            multilingual_model=e["multilingual_model"],
            multilingual_pretrained=e["multilingual_pretrained"],
        ),
        concepts=ConceptsConfig(
            path=_abs(raw["concepts"]["path"]),
            templates=dict(raw["concepts"]["templates"]),
            translated={k: _abs(v) for k, v in raw["concepts"]["translated"].items()},
        ),
        analysis=AnalysisConfig(**raw["analysis"]),
    )


def resolve_device(device: str) -> str:
    """'auto' picks mps on Apple Silicon, else cpu."""
    if device != "auto":
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    """Seed python, numpy and torch. Pass `random_state=seed` to scikit-learn explicitly."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
