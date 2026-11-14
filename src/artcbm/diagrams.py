"""Method diagrams on made-up labels: what each test compares and what each shuffle does.

No real paintings or annotations are used here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from artcbm.plots import INK, INK_2, LANG_COLORS

LANGS = ["english", "arabic", "chinese"]
# Toy emotions: a letter plus a colour (not the language colours). Dark fills get white text.
TOY = {
    "C": ("contentment", "#eda100", INK),
    "S": ("sadness", "#6250d6", "white"),
    "A": ("awe", "#e87ba4", INK),
    "F": ("fear", "#008300", "white"),
    "E": ("something else", "#52514e", "white"),
}
# 3 paintings x 3 languages x 5 annotators. Painting A reads sad, B content, C fearful;
# the Chinese annotators lean towards contentment everywhere (a label habit).
TOY_LABELS = np.array(
    [
        [list("SSFSA"), list("SASSE"), list("SCSCS")],
        [list("CACCE"), list("AACAC"), list("CCCCA")],
        [list("FSFAF"), list("FFEFS"), list("CFCFC")],
    ]
)
PAINTINGS = ["painting A", "painting B", "painting C"]


def _cell_x(lang: int, k: int) -> float:
    return lang * 6 + k


def draw_grid(ax, labels, title, faded=None, boxes=()):
    """Draw paintings (rows) x annotators grouped by language (columns).
    `faded` is a set of letters drawn in light grey; `boxes` are (row, lang, k0, k1, style,
    text) outlines around annotator groups."""
    faded = faded or set()
    for row in range(labels.shape[0]):
        y = labels.shape[0] - 1 - row
        for lang in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                letter = labels[row, lang, k]
                _, fill, text = TOY[letter]
                if letter in faded:
                    fill, text = "#e1e0d9", "#898781"
                ax.add_patch(Rectangle((_cell_x(lang, k), y), 0.9, 0.9, color=fill, lw=0))
                ax.text(
                    _cell_x(lang, k) + 0.45,
                    y + 0.45,
                    letter,
                    ha="center",
                    va="center",
                    color=text,
                    fontsize=10,
                    fontweight="bold",
                )
        ax.text(-0.4, y + 0.45, PAINTINGS[row], ha="right", va="center", fontsize=10, color=INK_2)
    for lang, name in enumerate(LANGS):
        ax.add_patch(
            Rectangle(
                (_cell_x(lang, 0), labels.shape[0] + 0.15), 4.9, 0.12, color=LANG_COLORS[name], lw=0
            )
        )
        ax.text(
            _cell_x(lang, 2) + 0.45,
            labels.shape[0] + 0.5,
            name,
            ha="center",
            fontsize=10,
            color=INK,
        )
    for row, lang, k0, k1, style, text in boxes:
        y = labels.shape[0] - 1 - row
        ax.add_patch(
            Rectangle(
                (_cell_x(lang, k0) - 0.08, y - 0.08),
                k1 - k0 + 0.06 + 0.9,
                1.06,
                fill=False,
                lw=2,
                ls=style,
                ec=INK,
            )
        )
        if text:
            ax.text(
                _cell_x(lang, k0) + (k1 - k0 + 0.9) / 2,
                y - 0.35,
                text,
                ha="center",
                va="top",
                fontsize=9,
                color=INK,
            )
    ax.set_xlim(-4.2, 17.6)
    ax.set_ylim(-1.3, labels.shape[0] + 1.1)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=4)
    ax.axis("off")


def _legend(fig, letters, y=0.0):
    handles = [Rectangle((0, 0), 1, 1, color=TOY[x][1]) for x in letters]
    names = [f"{x} = {TOY[x][0]}" for x in letters]
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=len(letters),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, y),
    )


def _shuffle_within_painting(labels, rng):
    n_p, n_l, n = labels.shape
    return rng.permuted(labels.reshape(n_p, n_l * n), axis=1).reshape(n_p, n_l, n)


def _shuffle_within_language(labels, rng):
    out = labels.copy()
    for lang in range(labels.shape[1]):
        out[:, lang, :] = rng.permuted(labels[:, lang, :].ravel()).reshape(labels.shape[0], -1)
    return out


def disagreement_method(seed: int = 3):
    """Section 2: within vs across comparisons, and the two shuffles."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    boxes = [(2, 0, 0, 1, "-", "within"), (2, 0, 2, 3, "-", ""), (2, 2, 0, 1, "--", "across")]
    draw_grid(axes[0], TOY_LABELS, "Real labels", boxes=boxes)
    draw_grid(axes[1], _shuffle_within_painting(TOY_LABELS, rng), "If language didn't matter")
    draw_grid(axes[2], _shuffle_within_language(TOY_LABELS, rng), "If only label habits mattered")
    axes[1].annotate(
        "",
        xy=(16.5, -0.75),
        xytext=(0, -0.75),
        arrowprops={"arrowstyle": "<->", "color": INK_2, "lw": 1.5},
    )
    axes[1].text(
        8.2,
        -1.25,
        "labels swap between languages, within each painting",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )
    for lang in range(3):
        axes[2].annotate(
            "",
            xy=(_cell_x(lang, 4) + 1.25, 3.0),
            xytext=(_cell_x(lang, 4) + 1.25, 0),
            arrowprops={"arrowstyle": "<->", "color": INK_2, "lw": 1.5},
        )
    axes[2].text(
        8.2,
        -1.25,
        "labels move between paintings, within each language",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )
    axes[0].text(
        8.2,
        -1.25,
        "compare the emotion mix of small groups of annotators",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )
    _legend(fig, "CSAFE", y=-0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def something_else_method(seed: int = 5):
    """Section 3: the share of 'something else' per language, before and after shuffling."""
    rng = np.random.default_rng(seed)
    shuffled = _shuffle_within_painting(TOY_LABELS, rng)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    others = set("CSAF")
    for ax, labels, title in [
        (axes[0], TOY_LABELS, "Real labels"),
        (axes[1], shuffled, "If language didn't matter"),
    ]:
        draw_grid(ax, labels, title, faded=others)
        for lang in range(3):
            share = (labels[:, lang, :] == "E").mean()
            ax.text(
                _cell_x(lang, 2) + 0.45,
                -0.6,
                f"{share:.0%}",
                ha="center",
                fontsize=11,
                color=INK,
                fontweight="bold",
            )
        ax.text(
            8.2,
            -1.25,
            "share of 'something else' per language",
            ha="center",
            fontsize=9.5,
            color=INK_2,
        )
    _legend(fig, "E", y=-0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def _box(ax, x, y, w, h, text, fc="#f3f2ee", ec="#c3c2b7", fontsize=10.5, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
    )


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=INK_2, lw=1.4
        )
    )


BAR = "#52514e"  # concept scores: neutral, never a language colour


def _bars(ax, x, y, values, w=0.16, h=0.55, color=BAR):
    for i, v in enumerate(values):
        ax.add_patch(Rectangle((x + i * (w + 0.05), y), w, h * v, color=color, lw=0))


def inverted_cbm_method():
    """Section 4: what the language classifier sees, and why concepts alone cannot work."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
    # left: the pipeline
    _box(ax1, 0.1, 2.9, 1.8, 1.4, "painting")
    _arrow(ax1, 2.0, 3.6, 2.7, 3.6)
    _box(ax1, 2.8, 2.7, 2.2, 1.8, "")
    _bars(ax1, 3.05, 3.05, [0.9, 0.3, 0.7, 0.2, 0.5, 0.8])
    ax1.text(3.9, 4.15, "concept scores", ha="center", fontsize=10, color=INK)
    _box(ax1, 2.8, 0.6, 2.2, 1.2, "emotion chosen:\nfear")
    _arrow(ax1, 5.1, 3.6, 6.0, 2.8)
    _arrow(ax1, 5.1, 1.2, 6.0, 2.2)
    _box(ax1, 6.1, 1.8, 1.7, 1.4, "classifier", fc="#ebe9f8", ec="#6250d6", weight="bold")
    _arrow(ax1, 7.9, 2.5, 8.4, 2.5)
    for i, lang in enumerate(LANGS):
        ax1.add_patch(plt.Circle((8.65, 3.4 - i * 0.9), 0.13, color=LANG_COLORS[lang]))
        ax1.text(8.9, 3.4 - i * 0.9, lang + "?", va="center", fontsize=10.5, color=INK)
    ax1.text(
        5.0,
        0.05,
        "inputs tried: concepts · emotion · both · emotion × concepts",
        ha="center",
        fontsize=10,
        color=INK_2,
    )
    ax1.set_title(
        "What the classifier sees, for one annotation", fontsize=12, fontweight="bold", color=INK
    )
    # right: concepts alone
    for i in range(15):
        lang = LANGS[i // 5]
        x, y = 0.4 + (i % 5) * 0.75, 3.6 - (i // 5) * 1.25
        _bars(ax2, x, y, [0.9, 0.3, 0.7, 0.5], w=0.11, h=0.6)
        ax2.add_patch(plt.Circle((x + 0.25, y - 0.22), 0.1, color=LANG_COLORS[lang]))
    ax2.text(
        6.2,
        3.1,
        "15 annotations of one painting:\nthe same concept scores for all,\n"
        "5 from each language.\n\nAny guess is right for 5 of 15,\nwhich is chance.",
        ha="left",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    ax2.set_title(
        "Why concepts alone must land on chance", fontsize=12, fontweight="bold", color=INK
    )
    fig.tight_layout()
    return fig


def variance_method():
    """Section 6: the split of one emotion choice (sadness yes/no) into five parts."""
    y = (TOY_LABELS == "S").astype(float)  # paintings x languages x annotators
    grand = y.mean()
    cell = y.mean(axis=2)
    paint = cell.mean(axis=1, keepdims=True) - grand
    lang = cell.mean(axis=0, keepdims=True) - grand
    inter = cell - grand - paint - lang
    panels = [
        ("the data:\nwho picked sadness", None),
        ("average", np.full((3, 3), grand)),
        ("painting", np.repeat(paint, 3, axis=1)),
        ("language", np.repeat(lang, 3, axis=0)),
        ("painting × language", inter),
        ("left over", None),
    ]
    signs = ["=", "+", "+", "+", "+"]
    fig, axes = plt.subplots(1, 6, figsize=(13.5, 3.3))
    cmap = plt.get_cmap("RdBu_r")
    for i, (ax, (name, grid)) in enumerate(zip(axes, panels, strict=True)):
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.6, 3.5)
        ax.axis("off")
        ax.set_title(name, fontsize=11.5, fontweight="bold", color=INK)
        for r in range(3):
            for c in range(3):
                yy = 2 - r
                if name.startswith("the data") or name == "left over":
                    ax.add_patch(Rectangle((c, yy), 0.94, 0.94, fc="#f3f2ee", ec="#e1e0d9"))
                    for k in range(5):
                        val = y[r, c, k]
                        if name == "left over":
                            dev = val - cell[r, c]
                            if abs(dev) < 1e-9:
                                color = "#d9d8d2"
                            else:
                                color = "#d6604d" if dev > 0 else "#4393c3"
                        else:
                            color = TOY["S"][1] if val else "#d9d8d2"
                        ax.add_patch(
                            plt.Circle((c + 0.15 + k * 0.16, yy + 0.47), 0.07, color=color)
                        )
                else:
                    v = grid[r, c]
                    color = "#d9d8d2" if name == "average" else cmap(0.5 + v / 1.2)
                    ax.add_patch(Rectangle((c, yy), 0.94, 0.94, color=color))
                    ax.text(
                        c + 0.47,
                        yy + 0.47,
                        f"{v:+.2f}" if name != "average" else f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=INK,
                    )
        for c, lname in enumerate(LANGS):
            ax.add_patch(Rectangle((c, 3.02), 0.94, 0.1, color=LANG_COLORS[lname], lw=0))
        if i == 0:
            for r, pname in enumerate(["A", "B", "C"]):
                ax.text(-0.15, 2.47 - r, pname, ha="right", va="center", fontsize=10, color=INK_2)
        if i < len(signs):
            ax.text(3.45, 1.4, signs[i], fontsize=20, ha="center", va="center", color=INK_2)
    fig.text(
        0.5,
        0.02,
        "rows: paintings A, B, C · columns: english, arabic, chinese · "
        "red = above average, blue = below",
        ha="center",
        fontsize=10,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def swap_method(seed: int = 1):
    """Section 5: concept profiles move between paintings; languages and emotions stay."""
    rows = [
        ("english", "sadness", "A"),
        ("chinese", "contentment", "B"),
        ("arabic", "fear", "C"),
        ("chinese", "awe", "A"),
    ]
    profiles = {"A": [0.9, 0.2, 0.7, 0.4], "B": [0.3, 0.9, 0.2, 0.8], "C": [0.6, 0.5, 0.9, 0.1]}
    swap = {"A": "C", "B": "A", "C": "B"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    for ax, title, mapping in [
        (axes[0], "Real pairing", {k: k for k in profiles}),
        (axes[1], "Swapped profiles", swap),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.4, len(rows) + 0.8)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
        for j, head in enumerate(["language", "emotion", "painting", "concept profile used"]):
            ax.text(
                [0.3, 2.5, 4.7, 6.6][j],
                len(rows) + 0.2,
                head,
                fontsize=10,
                color=INK_2,
                fontweight="bold",
            )
        for i, (lang, emo, painting) in enumerate(rows):
            yy = len(rows) - 1 - i
            ax.add_patch(plt.Circle((0.45, yy + 0.3), 0.12, color=LANG_COLORS[lang]))
            ax.text(0.7, yy + 0.3, lang, va="center", fontsize=10.5, color=INK)
            ax.text(2.5, yy + 0.3, emo, va="center", fontsize=10.5, color=INK)
            ax.text(4.7, yy + 0.3, painting, va="center", fontsize=10.5, color=INK)
            used = mapping[painting]
            _bars(
                ax,
                6.6,
                yy,
                profiles[used],
                w=0.22,
                h=0.6,
            )
            weight = "normal" if used == painting else "bold"
            ax.text(
                7.75,
                yy + 0.3,
                f"from {used}",
                va="center",
                fontsize=10.5,
                color=INK,
                fontweight=weight,
            )
    axes[1].text(
        5,
        -0.35,
        "label habits stay intact; only the link to what is in the painting is broken",
        ha="center",
        fontsize=10,
        color=INK_2,
    )
    fig.tight_layout()
    return fig


EMOTION_LETTERS = ["Am", "Aw", "Co", "Ex", "An", "Di", "Fe", "Sa", "SE"]
HIGHLIGHT = "#6250d6"  # the bottleneck / an edit


def _prob_bars(ax, x, y, probs, chosen=None, w=0.17, h=1.0, labels=True):
    """Nine small bars for an emotion distribution; the chosen emotion outlined."""
    for i, p in enumerate(probs):
        ax.add_patch(Rectangle((x + i * (w + 0.04), y), w, h * p, color=BAR, lw=0))
        if chosen == i:
            ax.add_patch(
                Rectangle(
                    (x + i * (w + 0.04) - 0.03, y - 0.03),
                    w + 0.06,
                    h + 0.06,
                    fill=False,
                    ec=HIGHLIGHT,
                    lw=1.6,
                )
            )
        if not labels:
            continue
        ax.text(
            x + i * (w + 0.04) + w / 2,
            y - 0.18,
            EMOTION_LETTERS[i][:2],
            ha="center",
            va="top",
            fontsize=6.5,
            color=INK_2,
        )


def forward_cbm_method():
    """Section 1: concept scores -> one emotion head per language, and how it is scored."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
    _box(ax1, 0.1, 2.0, 1.6, 1.2, "painting")
    _arrow(ax1, 1.8, 2.6, 2.4, 2.6)
    _box(ax1, 2.5, 1.7, 2.0, 1.8, "")
    _bars(ax1, 2.75, 2.0, [0.9, 0.3, 0.7, 0.2, 0.5, 0.8])
    ax1.text(3.5, 3.15, "concept scores", ha="center", fontsize=10, color=INK)
    probs = {
        "english": [0.1, 0.15, 0.3, 0.07, 0.02, 0.04, 0.08, 0.12, 0.12],
        "arabic": [0.17, 0.19, 0.3, 0.08, 0.01, 0.02, 0.03, 0.07, 0.13],
        "chinese": [0.05, 0.08, 0.5, 0.04, 0.01, 0.06, 0.1, 0.14, 0.02],
    }
    for i, lang in enumerate(LANGS):
        yy = 3.7 - i * 1.55
        _arrow(ax1, 4.6, 2.6, 5.4, yy + 0.35)
        ax1.add_patch(plt.Circle((5.65, yy + 0.35), 0.12, color=LANG_COLORS[lang]))
        ax1.text(5.85, yy + 0.35, f"{lang} head", va="center", fontsize=10, color=INK)
        _prob_bars(ax1, 7.6, yy, probs[lang], h=0.9, labels=False)
    ax1.text(8.5, 4.75, "emotion mix", ha="center", fontsize=10, color=INK)
    ax1.set_title(
        "The bottleneck: what is in the painting → each language's emotions",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    # right: scoring one held-out annotation that chose sadness
    habits = [0.1, 0.15, 0.36, 0.06, 0.01, 0.04, 0.07, 0.11, 0.1]
    model = [0.05, 0.08, 0.2, 0.04, 0.01, 0.07, 0.17, 0.32, 0.06]
    ax2.text(0.2, 4.55, "one unseen annotation chose sadness (Sa)", fontsize=10.5, color=INK)
    _prob_bars(ax2, 0.1, 2.6, habits, chosen=7, w=0.36, h=1.4)
    ax2.text(1.95, 1.95, "habits-only guess", ha="center", fontsize=10, color=INK)
    _prob_bars(ax2, 5.4, 2.6, model, chosen=7, w=0.36, h=1.4)
    ax2.text(7.25, 1.95, "bottleneck", ha="center", fontsize=10, color=INK)
    ax2.text(
        5.0,
        0.9,
        "score = share of the habits-only error that the bottleneck removes\n"
        "(0 = no better than each language's overall mix, higher is better)",
        ha="center",
        fontsize=10,
        color=INK_2,
    )
    ax2.set_title("How it is scored", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def encoder_method():
    """Section 7: the same concepts through two encoders, into the same three tests."""
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _box(ax, 0.1, 1.4, 2.4, 1.2, "same paintings,\nsame concept list,\nsame English prompts")
    for i, name in enumerate(["original CLIP\n(mostly English)", "multilingual CLIP"]):
        yy = 2.55 - i * 1.7
        _arrow(ax, 2.6, 2.0, 3.6, yy + 0.4)
        _box(ax, 3.7, yy, 2.4, 0.9, name, fc="#ebe9f8", ec=HIGHLIGHT)
        _arrow(ax, 6.2, yy + 0.45, 7.0, yy + 0.45)
        _box(ax, 7.1, yy, 2.0, 0.9, "concept scores")
        _arrow(ax, 9.2, yy + 0.45, 9.9, 2.0)
    _box(
        ax,
        10.0,
        0.9,
        2.4,
        2.2,
        "the same tests:\n\nsection 1 bottleneck\nsection 4 classifier\nsection 5 swap test",
    )
    ax.set_title(
        "Swap the encoder, keep everything else", fontsize=12, fontweight="bold", color=INK
    )
    fig.tight_layout()
    return fig


def vocabulary_method():
    """Section 8: which concept list each language's head uses, and where native words
    come from."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.axis("off")
    rows = {
        "generic": ["a boat · the sea · red", "a boat · the sea · red", "a boat · the sea · red"],
        "translated": ["a boat · the sea · red", "قارب · البحر · أحمر", "小船 · 大海 · 红色"],
        "native": ["colors · woman · dark", "رجل · لون · سماء", "感到 · 满足 · 女人"],
    }
    for j, lang in enumerate(LANGS):
        x = 2.2 + j * 2.6
        ax1.add_patch(Rectangle((x, 4.35), 2.4, 0.12, color=LANG_COLORS[lang], lw=0))
        ax1.text(x + 1.2, 4.6, f"{lang} head", ha="center", fontsize=10, color=INK)
    for i, (name, cells) in enumerate(rows.items()):
        yy = 3.3 - i * 1.25
        ax1.text(0.1, yy + 0.4, name, va="center", fontsize=10.5, color=INK, fontweight="bold")
        for j, text in enumerate(cells):
            _box(ax1, 2.2 + j * 2.6, yy, 2.4, 0.8, text, fontsize=9.5)
    ax1.text(
        5.0,
        0.0,
        "all three scored with the multilingual CLIP; same list length",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )
    ax1.set_title(
        "Which concept list each language's head uses", fontsize=12, fontweight="bold", color=INK
    )
    # right: where native words come from
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis("off")
    _box(ax2, 0.3, 2.6, 4.2, 1.6, "479 other paintings\n(their explanations)", fc="#f3f2ee")
    _box(ax2, 5.5, 2.6, 4.2, 1.6, "our 840 paintings\n(tested)", fc="#ebe9f8", ec=HIGHLIGHT)
    _arrow(ax2, 2.4, 2.5, 2.4, 1.4)
    ax2.text(
        2.4,
        0.9,
        "most common words\nper language → native lists",
        ha="center",
        fontsize=10,
        color=INK,
    )
    ax2.plot([5.0, 5.0], [0.4, 4.5], color=INK_2, lw=1.2, ls="--")
    ax2.text(
        7.6, 1.3, "never used for mining,\nso nothing leaks", ha="center", fontsize=10, color=INK_2
    )
    ax2.set_title("Where native words come from", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def edit_method():
    """Section 9: real edits raise the mentioned concepts; sham edits raise random ones."""
    names = ["a boat", "the sea", "the sky", "a cat", "a bridge", "red"]
    base = [0.45, 0.35, 0.6, 0.15, 0.25, 0.3]
    high = 0.85
    edits = {
        "concept scores": set(),
        "real edit": {"a boat", "the sea"},
        "sham edit": {"a cat", "a bridge"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (title, raised) in zip(axes, edits.items(), strict=True):
        for i, (name, v) in enumerate(zip(names, base, strict=True)):
            ax.add_patch(Rectangle((i, 0), 0.7, v, color=BAR, lw=0))
            if name in raised:
                ax.add_patch(Rectangle((i, v), 0.7, high - v, color=HIGHLIGHT, lw=0))
            ax.text(
                i + 0.35, -0.05, name, ha="right", va="top", rotation=40, fontsize=9.5, color=INK
            )
        ax.axhline(high, color=INK_2, lw=1, ls=":")
        ax.text(5.9, high + 0.02, "high", ha="right", va="bottom", fontsize=9, color=INK_2)
        ax.set_xlim(-0.3, 6)
        ax.set_ylim(-0.55, 1.15)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    axes[0].text(
        2.85,
        1.03,
        "English explanation: “a lonely boat on a grey sea”",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )
    axes[1].text(
        2.85, 1.03, "raise what the explanation mentions", ha="center", fontsize=9.5, color=INK_2
    )
    axes[2].text(
        2.85, 1.03, "raise as many random concepts", ha="center", fontsize=9.5, color=INK_2
    )
    fig.text(
        0.5,
        0.0,
        "then compare the bottleneck's predictions for the Arabic and Chinese "
        "annotators of the same painting",
        ha="center",
        fontsize=10,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
