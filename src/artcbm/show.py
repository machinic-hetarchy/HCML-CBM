"""Small HTML cards for showing paintings and annotations inside the notebook.

HTML instead of matplotlib so Arabic and Chinese text render correctly.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display
from PIL import Image

from artcbm.plots import LANG_COLORS

CELL = "padding:4px 10px;vertical-align:top;text-align:left;border:none;"


def thumb(path: Path, height: int = 200) -> str:
    """An <img> tag with a small embedded JPEG thumbnail."""
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
    style = f"width:{width}px;" if width else ""
    return (
        f'<div style="display:inline-block;margin:6px 10px 6px 0;vertical-align:top;{style}">'
        f'{image_html}<div style="font-size:0.85em;opacity:0.8;margin-top:4px">{caption}</div>'
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
            text = f"<div dir='auto'>{html.escape(r.caption)}</div>"
            if r.caption in translations:
                text += (
                    "<div style='opacity:0.7;font-style:italic'>(trans.) "
                    f"{html.escape(translations[r.caption])}</div>"
                )
            cells += f"<td style='{CELL}'>{text}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table style='border-collapse:collapse;font-size:0.9em'><tr>{head}</tr>{body}</table>"


def emotion_counts(df: pd.DataFrame, painting: str, languages) -> str:
    """For one painting: each language's labels as 'sadness ×3, fear ×2'."""
    out = ""
    for lang in languages:
        counts = df[(df["painting"] == painting) & (df["language"] == lang)]["emotion"]
        text = ", ".join(f"{e} ×{n}" for e, n in counts.value_counts().items())
        out += f"<tr><td style='{CELL}'>{lang_label(lang)}</td><td style='{CELL}'>{text}</td></tr>"
    return f"<table style='border-collapse:collapse;font-size:0.9em'>{out}</table>"


def show(*blocks: str) -> None:
    display(HTML(row(*blocks)))
