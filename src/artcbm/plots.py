"""Shared figure style: one colour per language, recessive axes, PNG export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from artcbm.config import REPO_ROOT

FIG_DIR = REPO_ROOT / "outputs" / "figures"

# Language colours, matching the static method figures in notebooks/figures/methods/.
LANG_COLORS = {"english": "#378ADD", "arabic": "#D85A30", "chinese": "#1D9E75"}
NEUTRAL = "#898781"  # nulls, baselines, residual
# Models and vocabularies (kept apart from the language colours; validated all-pairs).
MODEL_COLORS = {"bottleneck": "#6250d6", "black box": "#eda100", "other people": "#008300"}
VOCAB_COLORS = {  # the filtered native list shares the native colour (drawn hatched)
    "generic": NEUTRAL,
    "translated": "#eda100",
    "native": "#6250d6",
    "native, no feeling words": "#6250d6",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _font_family() -> list[str]:
    """Default font, plus a CJK fallback when one is installed (for Chinese text)."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    fallbacks = ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS"]
    return ["DejaVu Sans"] + [f for f in fallbacks if f in installed][:1]


def setup_style() -> None:
    import logging

    # The CJK fallback font has no bold face; matplotlib picks the nearest and says so.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    try:  # sharper figures in the notebook (2x pixel density); no effect outside Jupyter
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("retina")
    except ImportError:
        pass
    plt.rcParams.update(
        {
            "font.family": _font_family(),
            "figure.dpi": 110,
            "savefig.dpi": 200,
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
