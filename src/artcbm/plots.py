"""Shared figure style: one colour per language, recessive axes, PNG export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from artcbm.config import REPO_ROOT

FIG_DIR = REPO_ROOT / "outputs" / "figures"

# First three categorical slots of a colour-blind-checked palette (validated all-pairs).
LANG_COLORS = {"english": "#2a78d6", "arabic": "#eb6834", "chinese": "#1baf7a"}
NEUTRAL = "#898781"  # nulls, baselines, residual
# Models and vocabularies (kept apart from the language colours; validated all-pairs).
MODEL_COLORS = {"bottleneck": "#6250d6", "black box": "#eda100", "other people": "#008300"}
VOCAB_COLORS = {"generic": NEUTRAL, "translated": "#eda100", "native": "#6250d6"}
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
    plt.rcParams.update(
        {
            "font.family": _font_family(),
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


CHAIN = [
    (
        "Labels",
        "Becker; Bowker & Star",
        "Do the languages use the emotion options differently?",
        "2, 3",
    ),
    (
        "Language as a category",
        "Hacking",
        "Does the chosen emotion give the language away?",
        "4, 5",
    ),
    (
        "Language-specific concepts",
        "Derrida",
        "Do another encoder or other vocabularies change the picture?",
        "7, 8",
    ),
    ("Checks", "Latour; Sokal", "Does each effect beat its shuffled version?", "all"),
    (
        "The painting's share",
        "Koh et al.; Nisbett & Wilson",
        "How much does the painting decide, and does the model follow it?",
        "6, 9",
    ),
]


def argument_chain(results: dict | None = None, with_theory: bool = False):
    """Five questions as boxes left to right, with a dashed arrow from the last back to
    the first. `results` maps box number (1-5) to a one-line result shown under the
    question. `with_theory` adds the theory names under each title (for the report)."""
    import textwrap

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    height = 3.8 if results else 2.6
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set_xlim(0, 12.25)
    ax.set_ylim(0, height)
    ax.axis("off")
    top, bottom, line, width, gap = height - 0.1, 0.75, 0.2, 2.15, 0.3

    def write(x, y, text, size, color, wrap, **kw):
        text = textwrap.fill(text, wrap)
        ax.text(x, y, text, ha="center", va="top", fontsize=size, color=color, **kw)
        return text.count("\n") + 1

    for i, (title, theory, question, parts) in enumerate(CHAIN, start=1):
        x = 0.15 + (i - 1) * (width + gap)
        mid = x + width / 2
        box = FancyBboxPatch(
            (x, bottom),
            width,
            top - bottom,
            fc="#f3f2ee",
            ec="#c3c2b7",
            boxstyle="round,pad=0.02,rounding_size=0.12",
            lw=1.2,
        )
        ax.add_patch(box)
        y = top - 0.18
        y -= line * (write(mid, y, title, 10.5, INK, 22, fontweight="bold") + 0.15)
        if with_theory:
            y -= line * (write(mid, y, theory, 8.5, NEUTRAL, 30) + 0.1)
        y -= line * (write(mid, y - 0.05, question, 9.5, INK_2, 26) + 0.6)
        y -= line * (
            write(mid, y, "all sections" if parts == "all" else f"sections {parts}", 9, NEUTRAL, 30)
            + 0.4
        )
        if results and i in results:
            write(mid, y, results[i], 9.5, INK, 25)
        if i < len(CHAIN):
            ax.annotate(
                "",
                xy=(x + width + gap - 0.02, top - 0.3),
                xytext=(x + width + 0.02, top - 0.3),
                arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.3},
            )
    first_x = 0.15 + width / 2
    last_x = 0.15 + 4 * (width + gap) + width / 2
    ax.add_patch(
        FancyArrowPatch(
            (last_x, bottom - 0.03),
            (first_x, bottom - 0.03),
            connectionstyle="arc3,rad=-0.06",
            arrowstyle="-|>",
            mutation_scale=14,
            color=INK_2,
            lw=1.3,
            ls="--",
        )
    )
    if results:
        ax.text(
            (first_x + last_x) / 2,
            0.2,
            "the language signal sits mostly in the labels",
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK_2,
            style="italic",
        )
    fig.tight_layout()
    return fig
