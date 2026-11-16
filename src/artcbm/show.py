"""Small HTML cards for paintings and annotations, and the static method figures.

HTML instead of matplotlib so Arabic and Chinese text render correctly.
"""

from __future__ import annotations

import base64
import html
import io
import textwrap
from pathlib import Path

import pandas as pd
from IPython.display import HTML, SVG, display
from PIL import Image

from artcbm.config import REPO_ROOT
from artcbm.plots import LANG_COLORS

CELL = "padding:5px 12px;vertical-align:top;text-align:left;border:none;"
SCALE = 1.4  # thumbnails and card widths, relative to the sizes given by callers
TEXT = "font-size:1.1em;line-height:1.45"
TABLE = "border-collapse:collapse;" + TEXT
METHODS_DIR = REPO_ROOT / "notebooks" / "figures" / "methods"


def method_figure(name: str) -> None:
    """Show one of the static method figures by file stem, e.g. "03_shuffle_tests"."""
    display(SVG(filename=str(METHODS_DIR / f"{name}.svg")))


def chain_results(results: dict[int, str], width: int = 24, max_lines: int = 3) -> None:
    """Show the argument chain with one result per box (boxes 1 to 5), filled into the
    SVG template. Each result is wrapped to `max_lines` lines of about `width` characters;
    a result that needs more lines raises an error instead of being cut off."""
    svg = (METHODS_DIR / "02_argument_chain_results.svg.tmpl").read_text()
    for box in range(1, 6):
        wrapped = textwrap.wrap(results.get(box, ""), width=width)
        if len(wrapped) > max_lines:
            raise ValueError(f"result for box {box} needs {len(wrapped)} lines: {wrapped}")
        wrapped += [""] * (max_lines - len(wrapped))
        for i, line in enumerate(wrapped, start=1):
            svg = svg.replace(f"{{r{box}_{i}}}", html.escape(line, quote=False))
    display(SVG(data=svg))


def thumb(path: Path, height: int = 200) -> str:
    """An <img> tag with a small embedded JPEG thumbnail."""
    height = int(height * SCALE)
    img = Image.open(path).convert("RGB")
    img.thumbnail((height * 3, height))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{data}" style="height:{height}px;border-radius:4px">'


def title(painting: str) -> str:
    """'edvard-munch_children-in-the-street-1915' -> 'Edvard Munch: Children in the street 1915'."""
    artist, _, name = painting.partition("_")
    artist = " ".join(w.capitalize() for w in artist.split("-"))
    return f"{artist}: {name.replace('-', ' ').capitalize()}"


def lang_label(lang: str) -> str:
    dot = f'<span style="color:{LANG_COLORS.get(lang, "#888")};font-size:1.2em">●</span>'
    return f'<span style="white-space:nowrap">{dot} {lang}</span>'


def figure(image_html: str, caption: str, width: int | None = None) -> str:
    style = f"width:{int(width * SCALE)}px;" if width else ""
    return (
        f'<div style="display:inline-block;margin:6px 10px 6px 0;vertical-align:top;{style}">'
        f'{image_html}<div style="{TEXT};opacity:0.85;margin-top:6px">{caption}</div>'
        "</div>"
    )


def row(*blocks: str) -> str:
    """Lay blocks out side by side, wrapping on narrow screens."""
    inner = "".join(f'<div style="margin-right:18px">{b}</div>' for b in blocks)
    return f'<div style="display:flex;flex-wrap:wrap;align-items:flex-start">{inner}</div>'


def annotation_table(
    rows: pd.DataFrame, show_caption: bool = True, translations: dict | None = None
) -> str:
    """Table of annotations: language, emotion and (optionally) the explanation.

    `translations` maps an explanation to an English translation shown beneath it.
    """
    translations = translations or {}
    head = "<th style='" + CELL + "'>language</th><th style='" + CELL + "'>emotion</th>"
    if show_caption:
        head += "<th style='" + CELL + "'>explanation</th>"
    body = ""
    for r in rows.itertuples():
        cells = (
            f"<td style='{CELL}'>{lang_label(r.language)}</td>"
            f"<td style='{CELL}'><b>{html.escape(r.emotion)}</b></td>"
        )
        if show_caption:
            cells += f"<td style='{CELL}'>{explanation(r.caption, translations)}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table style='{TABLE}'><tr>{head}</tr>{body}</table>"


def explanation(text: str, translations: dict | None = None) -> str:
    """An explanation in its own script, with an English translation below if given."""
    out = f"<div dir='auto'>{html.escape(text)}</div>"
    if translations and text in translations:
        out += (
            "<div style='opacity:0.7;font-style:italic'>(trans.) "
            f"{html.escape(translations[text])}</div>"
        )
    return out


def emotion_counts(df: pd.DataFrame, painting: str, languages) -> str:
    """For one painting: each language's labels as 'sadness ×3, fear ×2'."""
    out = ""
    for lang in languages:
        counts = df[(df["painting"] == painting) & (df["language"] == lang)]["emotion"]
        text = ", ".join(f"{e} ×{n}" for e, n in counts.value_counts().items())
        out += f"<tr><td style='{CELL}'>{lang_label(lang)}</td><td style='{CELL}'>{text}</td></tr>"
    return f"<table style='{TABLE}'>{out}</table>"


def show(*blocks: str) -> None:
    display(HTML(row(*blocks)))


def bottleneck_steps(
    df, probs, pushes: dict, scores_row: pd.Series, painting: str, languages, image_html: str
) -> str:
    """One painting through the bottleneck: the image, its strongest concepts, and per
    language the predicted emotions, the concepts that push hardest toward the top one,
    and what the annotators chose. `pushes` maps language -> emotions x concepts.
    A concept marked (low) pushes because the painting scores below average on it."""
    concepts = ", ".join(
        f"{c} <span style='opacity:0.6'>{v:+.1f}</span>" for c, v in scores_row.nlargest(6).items()
    )
    step1 = (
        f"<div style='{TEXT};margin:0 0 8px 12px'><b>1. Concepts CLIP sees:</b> {concepts}</div>"
    )
    head = (
        f"<tr><th style='{CELL}'></th><th style='{CELL}'>2. predicted</th>"
        f"<th style='{CELL}'>pushed most by</th><th style='{CELL}'>chosen by people</th></tr>"
    )
    body = ""
    for lang in languages:
        rows = ((df["painting"] == painting) & (df["language"] == lang)).to_numpy()
        pred = probs[rows].iloc[0].nlargest(2)
        top = pred.index[0]
        drivers = pushes[lang].loc[top].nlargest(3)
        chosen = df.loc[rows, "emotion"].value_counts()
        body += (
            f"<tr><td style='{CELL}'>{lang_label(lang)}</td>"
            f"<td style='{CELL}'>"
            + ", ".join(f"{e} {100 * v:.0f}%" for e, v in pred.items())
            + f"</td><td style='{CELL}'>"
            + ", ".join(c if scores_row[c] >= 0 else f"{c} (low)" for c in drivers.index)
            + "</td>"
            f"<td style='{CELL}'>"
            + ", ".join(f"{e} ×{n}" for e, n in chosen.items())
            + "</td></tr>"
        )
    step2 = f"<table style='{TABLE}'>{head}{body}</table>"
    steps = f"<div style='flex:1;min-width:0'>{step1}{step2}</div>"
    return (
        "<div style='display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px;"
        "border-top:1px solid rgba(128,128,128,0.3);padding:10px 0'>"
        f"{figure(image_html, title(painting))}{steps}</div>"
    )
