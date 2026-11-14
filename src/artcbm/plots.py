"""Shared figure style: one colour per language, recessive axes, PNG export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from artcbm.config import REPO_ROOT

FIG_DIR = REPO_ROOT / "outputs" / "figures"

# First three categorical slots of a colour-blind-checked palette (validated all-pairs).
LANG_COLORS = {"english": "#2a78d6", "arabic": "#eb6834", "chinese": "#1baf7a"}
NEUTRAL = "#898781"  # nulls, baselines, residual
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def savefig(fig, name: str) -> Path:
    """Save to outputs/figures/<name>.png and return the path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
