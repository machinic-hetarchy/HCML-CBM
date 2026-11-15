"""Regenerate the method figures (static SVGs) in notebooks/figures/methods/. Run only after editing a figure."""

import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "figures" / "methods"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
TEXT, TEXT2, MUTED = "#2C2C2A", "#5F5E5A", "#888780"
CARD_FILL, CARD_STROKE = "#FFFFFF", "#E8E6DF"
BOX_FILL, BOX_STROKE = "#F6F5F1", "#B4B2A9"
ACC_FILL, ACC_STROKE, ACC_TEXT = "#EEEDFE", "#534AB7", "#3C3489"
ACC_MID = "#AFA9EC"
BAR = "#B4B2A9"
ARROW = "#888780"

LANG = {
    "english": dict(fill="#E6F1FB", stroke="#185FA5", text="#0C447C", dot="#378ADD"),
    "arabic": dict(fill="#FAECE7", stroke="#993C1D", text="#712B13", dot="#D85A30"),
    "chinese": dict(fill="#E1F5EE", stroke="#0F6E56", text="#085041", dot="#1D9E75"),
}
LANGS = ["english", "arabic", "chinese"]

EMO = {  # letter: (fill, text, name)
    "C": ("#FAC775", "#633806", "contentment"),
    "S": ("#CECBF6", "#3C3489", "sadness"),
    "A": ("#F4C0D1", "#72243E", "awe"),
    "F": ("#C0DD97", "#27500A", "fear"),
    "E": ("#D3D1C7", "#444441", "something else"),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, body, title, desc):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" font-family="{FONT}" '
        f'style="max-width:100%;height:auto">'
        f"<title>{esc(title)}</title><desc>{esc(desc)}</desc>"
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" '
        'stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" '
        'stroke-linejoin="round"/></marker></defs>'
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="{CARD_FILL}" '
        f'stroke="{CARD_STROKE}"/>' + "".join(body) + "</svg>"
    )


def text(x, y, s, size=14, color=TEXT, anchor="start", weight=400, italic=False):
    st = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}" dominant-baseline="central"{st}>{esc(s)}</text>'
    )


def lines(x, y, rows, size=14, color=TEXT, anchor="start", weight=400, lh=None):
    lh = lh or size * 1.4
    return [text(x, y + i * lh, r, size, color, anchor, weight) for i, r in enumerate(rows)]


def box(x, y, w, h, fill=BOX_FILL, stroke=BOX_STROKE, rx=8, sw=0.75, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def arrow(x1, y1, x2, y2, color=ARROW, dash=None, both=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    start = ' marker-start="url(#arr)"' if both else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.25"'
        f'{d} marker-end="url(#arr)"{start}/>'
    )


def path_arrow(d, color=ARROW, dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.25"{ds} '
        'marker-end="url(#arr)"/>'
    )


def dot(x, y, r, fill):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>'


def bars(x, y_base, values, bw, gap, hmax, fill=BAR, highlight=None, hl_fill=None):
    out = []
    for i, v in enumerate(values):
        h = max(1.5, v * hmax)
        f = hl_fill if highlight and i in highlight else fill
        out.append(
            f'<rect x="{x + i * (bw + gap):.1f}" y="{y_base - h:.1f}" width="{bw}" '
            f'height="{h:.1f}" rx="1.5" fill="{f}"/>'
        )
    return out


def lang_pill(x, y, w, lang, label=None, h=32):
    c = LANG[lang]
    return [
        box(x, y, w, h, c["fill"], c["stroke"], rx=h / 2),
        dot(x + 16, y + h / 2, 5, c["dot"]),
        text(x + 28, y + h / 2, label or lang, 14, c["text"]),
    ]


def save(name, content):
    (OUT / f"{name}.svg").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------- 1. bottleneck
def fig_bottleneck():
    W, H = 960, 340
    b = []
    b += [text(24, 30, "The bottleneck", 14, TEXT, weight=500)]
    b += [text(24, 52, "what is in the painting goes in; each language's head gives its own emotion mix", 12, TEXT2)]
    b += [box(24, 150, 104, 52), text(76, 176, "painting", 14, TEXT, "middle")]
    b += [arrow(132, 176, 162, 176)]
    b += [box(166, 116, 132, 120), text(232, 136, "concept scores", 12, TEXT2, "middle")]
    b += bars(190, 220, [0.75, 0.3, 0.6, 0.2, 0.45, 0.85], 10, 5, 62)
    mixes = {
        "english": [0.25, 0.3, 0.55, 0.15, 0.05, 0.1, 0.2, 0.35, 0.3],
        "arabic": [0.35, 0.4, 0.5, 0.15, 0.03, 0.05, 0.08, 0.2, 0.3],
        "chinese": [0.1, 0.15, 0.95, 0.08, 0.04, 0.12, 0.25, 0.4, 0.06],
    }
    ys = {"english": 90, "arabic": 160, "chinese": 230}
    for lang in LANGS:
        y = ys[lang]
        b += [arrow(302, 176, 338, y + 16)]
        b += lang_pill(342, y, 132, lang, f"{lang} head")
        b += bars(492, y + 30, mixes[lang], 9, 3, 34)
    b += [text(546, 74, "emotion mix", 12, TEXT2, "middle")]
    # scoring panel
    X = 640
    b += [f'<line x1="{X - 16}" y1="24" x2="{X - 16}" y2="{H - 24}" stroke="{CARD_STROKE}" stroke-width="1"/>']
    b += [text(X, 30, "How it is scored", 14, TEXT, weight=500)]
    b += [text(X, 52, "one unseen annotation chose sadness", 12, TEXT2)]
    habits = [0.18, 0.22, 0.5, 0.12, 0.03, 0.08, 0.12, 0.16, 0.16]
    model = [0.12, 0.15, 0.38, 0.08, 0.02, 0.1, 0.22, 0.42, 0.1]
    for i, (vals, lab) in enumerate([(habits, "habits-only guess"), (model, "bottleneck")]):
        x0 = X + 8 + i * 150
        b += bars(x0, 210, vals, 11, 3, 110, highlight={7}, hl_fill=ACC_MID)
        sx = x0 + 7 * 14 - 2
        b += [box(sx, 96, 15, 118, "none", ACC_STROKE, rx=3, sw=1.25)]
        b += [text(x0 + 62, 232, lab, 12, TEXT2, "middle")]
    b += [text(X + 8 + 7 * 14 + 5, 86, "sadness", 11, ACC_TEXT, "middle")]
    b += lines(X, 272, ["score = share of the habits-only error", "that the bottleneck removes"], 12, TEXT)
    b += [text(X, 312, "0 = no better than each language's overall mix; higher is better", 11, MUTED)]
    save("01_bottleneck", svg(W, H, b, "The concept bottleneck and how it is scored",
                             "A painting's concept scores feed one head per language; each head's emotion mix is compared with a habits-only guess on an unseen annotation."))


# ---------------------------------------------------------------- 2. argument chain
CHAIN = [
    ("Labels", "Do the languages use the emotion options differently?", "sections 2, 3"),
    ("Language as a category", "Does the chosen emotion give the language away?", "sections 4, 5"),
    ("Language-specific concepts", "Do another encoder or other vocabularies change the picture?", "sections 7, 8"),
    ("Checks", "Does each effect beat its shuffled version?", "all sections"),
    ("The painting's share", "How much does the painting decide, and does the model follow it?", "sections 6, 9"),
]


def wrap(s, width):
    words, rows, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) > width:
            rows.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        rows.append(cur)
    return rows


def fig_chain(results=False):
    W = 960
    bw, gap, x0 = 168, 20, 22
    blocks = []
    for i, (title, q, secs) in enumerate(CHAIN):
        blocks.append((wrap(title, 19), wrap(q, 24), secs))
    tmax = max(len(t) for t, _, _ in blocks)
    qmax = max(len(q) for _, q, _ in blocks)
    body_h = 30 + tmax * 19 + 10 + qmax * 17 + 26
    bh = body_h + (70 if results else 0)
    H = 24 + bh + 96
    b = []
    for i, (trows, qrows, secs) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        b += [box(x, 24, bw, bh, rx=10)]
        b += [text(x + 14, 42, str(i + 1), 11, MUTED, weight=500)]
        y = 62
        for r in trows:
            b += [text(x + 14, y, r, 14, TEXT, weight=500)]
            y += 19
        y = 62 + tmax * 19 + 8
        for r in qrows:
            b += [text(x + 14, y, r, 12, TEXT2)]
            y += 17
        y = 62 + tmax * 19 + 8 + qmax * 17 + 6
        b += [text(x + 14, y, secs, 11, MUTED)]
        if results:
            y += 22
            b += [f'<line x1="{x + 14}" y1="{y - 10}" x2="{x + bw - 14}" y2="{y - 10}" stroke="{CARD_STROKE}"/>']
            for k in range(3):
                b += [text(x + 14, y + 6 + k * 17, "{" + f"r{i + 1}_{k + 1}" + "}", 12, ACC_TEXT, weight=500)]
        if i < 4:
            b += [arrow(x + bw + 3, 62, x + bw + gap - 3, 62)]
    xa = x0 + 4 * (bw + gap) + bw / 2
    xb = x0 + bw / 2
    yb = 24 + bh + 4
    b += [path_arrow(f"M{xa} {yb} C {xa} {yb + 70}, {xb} {yb + 70}, {xb} {yb + 6}", dash="5 4")]
    if results:
        b += [text(W / 2, yb + 64, "the language signal sits mostly in the labels", 12, TEXT2, "middle")]
    content = svg(W, H, b, "How the sections fit together",
                  "Five questions in sequence, from label habits to the painting's share, with a loop back to labels.")
    if results:
        (OUT / "02_argument_chain_results.svg.tmpl").write_text(content, encoding="utf-8")
    else:
        save("02_argument_chain", content)


# ---------------------------------------------------------------- shared label grid
REAL = {
    "A": {"english": "SSFSA", "arabic": "SASSE", "chinese": "SCSCS"},
    "B": {"english": "CACCE", "arabic": "AACAC", "chinese": "CCCCA"},
    "C": {"english": "FSFAF", "arabic": "FFEFS", "chinese": "CFCFC"},
}


def shuffle_languages(real, seed):
    rng = random.Random(seed)
    out = {}
    for p, d in real.items():
        pool = list("".join(d[l] for l in LANGS))
        rng.shuffle(pool)
        out[p] = {l: "".join(pool[i * 5:(i + 1) * 5]) for i, l in enumerate(LANGS)}
    return out


def shuffle_habits(real, seed):
    rng = random.Random(seed)
    out = {p: {} for p in real}
    for l in LANGS:
        pool = list("".join(real[p][l] for p in "ABC"))
        rng.shuffle(pool)
        for i, p in enumerate("ABC"):
            out[p][l] = "".join(pool[i * 5:(i + 1) * 5])
    return out


SQ, SG, GG = 18, 2, 12  # square, square gap, group gap
GROUP_W = 5 * SQ + 4 * SG
RL = 22


def grid(x, y, data, mode="emotion", row_labels=True, heads=True):
    """mode: 'emotion' colours every square; 'else' greys all but E."""
    b = []
    lx = x + (RL if row_labels else 0)
    for j, l in enumerate(LANGS):
        gx = lx + j * (GROUP_W + GG)
        if heads:
            b += [text(gx + GROUP_W / 2, y, l, 12, LANG[l]["text"], "middle")]
            b += [f'<rect x="{gx}" y="{y + 10}" width="{GROUP_W}" height="3" rx="1.5" fill="{LANG[l]["dot"]}"/>']
    for r, p in enumerate("ABC"):
        ry = y + 22 + r * (SQ + 10)
        if row_labels:
            b += [text(x + 4, ry + SQ / 2, p, 12, TEXT2)]
        for j, l in enumerate(LANGS):
            gx = lx + j * (GROUP_W + GG)
            for k, ch in enumerate(data[p][l]):
                sx = gx + k * (SQ + SG)
                if mode == "emotion":
                    fill, tc, _ = EMO[ch]
                else:
                    fill, tc = ("#444441", "#FFFFFF") if ch == "E" else ("#F1EFE8", "#B4B2A9")
                b += [f'<rect x="{sx}" y="{ry}" width="{SQ}" height="{SQ}" rx="3" fill="{fill}"/>']
                b += [text(sx + SQ / 2, ry + SQ / 2 + 0.5, ch, 11, tc, "middle", 500)]
    return b


def grid_width(row_labels=True):
    return (RL if row_labels else 0) + 3 * GROUP_W + 2 * GG


def legend(x, y, items):
    b, cx = [], x
    for letter, (fill, tc, name) in items:
        b += [f'<rect x="{cx}" y="{y - SQ / 2}" width="{SQ}" height="{SQ}" rx="3" fill="{fill}"/>']
        b += [text(cx + SQ / 2, y + 0.5, letter, 11, tc, "middle", 500)]
        lab = f"{name}"
        b += [text(cx + SQ + 7, y, lab, 12, TEXT2)]
        cx += SQ + 7 + len(lab) * 7 + 24
    return b


# ---------------------------------------------------------------- 3. shuffle tests
def fig_shuffles():
    gw = grid_width()
    gap = 34
    W = 24 * 2 + 3 * gw + 2 * gap
    H = 280
    lang_sh = shuffle_languages(REAL, 3)
    hab_sh = shuffle_habits(REAL, 7)
    panels = [
        ("Real labels", REAL, "rows: paintings A, B, C; compare small groups within or across languages"),
        ("If language didn't matter", lang_sh, "labels swap between languages, within each painting"),
        ("If only label habits mattered", hab_sh, "labels move between paintings, within each language"),
    ]
    b = []
    for i, (title, data, cap) in enumerate(panels):
        x = 24 + i * (gw + gap)
        b += [text(x, 30, title, 14, TEXT, weight=500)]
        b += grid(x, 58, data)
        b += [text(x, 196, cap, 11, TEXT2)]
    # within / across marks on the real panel
    lx = 24 + RL
    b += [box(lx - 2, 58 + 22 + 2 * (SQ + 10) - 2, 2 * SQ + SG + 4, SQ + 4, "none", TEXT, rx=4, sw=1.25)]
    b += [box(lx + 3 * (SQ + SG) - 2, 58 + 22 + 2 * (SQ + 10) - 2, 2 * SQ + SG + 4, SQ + 4, "none", TEXT, rx=4, sw=1.25)]
    ax = lx + 2 * (GROUP_W + GG)
    b += [box(ax - 2, 58 + 22 + 2 * (SQ + 10) - 2, 2 * SQ + SG + 4, SQ + 4, "none", TEXT, rx=4, sw=1.25, dash="3 2")]
    b += [text(lx + GROUP_W / 2, 176, "within", 11, TEXT, "middle")]
    b += [text(ax + SQ, 176, "across (vs English)", 11, TEXT, "middle")]
    # arrows on shuffle panels
    x2 = 24 + gw + gap + RL
    b += [arrow(x2 + 6, 172, x2 + 3 * GROUP_W + 2 * GG - 6, 172, both=True)]
    x3 = 24 + 2 * (gw + gap) + RL
    for j in range(3):
        gx = x3 + j * (GROUP_W + GG) + GROUP_W + 5
        b += [arrow(gx, 84, gx, 150, both=True)] if j == 2 else []
    b += legend(24, 240, [(k, EMO[k]) for k in "CSAFE"])
    save("03_shuffle_tests", svg(W, H, b, "Disagreement test and its two shuffles",
                                 "Made-up labels for three paintings in three languages; one shuffle swaps labels between languages, the other moves labels between paintings within a language."))


# ---------------------------------------------------------------- 4. something else
SHUF_ELSE = {
    "A": {"english": "SSCSS", "arabic": "ESFAS", "chinese": "ASCSS"},
    "B": {"english": "ECCCA", "arabic": "CCACA", "chinese": "CCACA"},
    "C": {"english": "FSFFC", "arabic": "FSFFF", "chinese": "FCEAC"},
}


def fig_something_else():
    gw = grid_width()
    gap = 60
    W = 24 * 2 + 2 * gw + gap
    H = 270
    b = []
    for i, (title, data) in enumerate([("Real labels", REAL), ("If language didn't matter", SHUF_ELSE)]):
        x = 24 + i * (gw + gap)
        b += [text(x, 30, title, 14, TEXT, weight=500)]
        b += grid(x, 58, data, mode="else")
        lx = x + RL
        for j, l in enumerate(LANGS):
            n = sum(data[p][l].count("E") for p in "ABC")
            b += [text(lx + j * (GROUP_W + GG) + GROUP_W / 2, 182, f"{round(100 * n / 15)}%", 14, TEXT, "middle", 500)]
        b += [text(x + RL, 206, "share of “something else” per language", 11, TEXT2)]
    b += [f'<rect x="24" y="{240 - SQ / 2}" width="{SQ}" height="{SQ}" rx="3" fill="#444441"/>',
          text(24 + SQ / 2, 240.5, "E", 11, "#FFFFFF", "middle", 500), text(24 + SQ + 7, 240, "something else", 12, TEXT2)]
    save("04_something_else", svg(W, H, b, "Counting “something else” per language",
                                  "Made-up labels with the catch-all option highlighted; real shares differ by language, shuffled shares even out."))


# ---------------------------------------------------------------- 5. classifier input
def fig_classifier():
    W, H = 960, 330
    b = [text(24, 30, "What the classifier sees, for one annotation", 14, TEXT, weight=500)]
    b += [box(24, 74, 104, 52), text(76, 100, "painting", 14, TEXT, "middle")]
    b += [arrow(132, 100, 160, 100)]
    b += [box(164, 62, 124, 76), text(226, 80, "concept scores", 12, TEXT2, "middle")]
    b += bars(194, 128, [0.7, 0.3, 0.55, 0.2, 0.8], 9, 5, 34)
    b += [box(164, 180, 124, 56), text(226, 200, "emotion chosen", 12, TEXT2, "middle"), text(226, 220, "fear", 14, TEXT, "middle", 500)]
    b += [arrow(292, 100, 326, 148), arrow(292, 208, 326, 168)]
    b += [box(330, 134, 116, 48, ACC_FILL, ACC_STROKE), text(388, 158, "classifier", 14, ACC_TEXT, "middle", 500)]
    b += [arrow(450, 158, 474, 158)]
    for k, l in enumerate(LANGS):
        y = 112 + k * 46
        b += [dot(490, y, 6, LANG[l]["dot"]), text(502, y, f"{l}?", 14, LANG[l]["text"])]
    b += [text(24, 276, "inputs tried: concepts · emotion · both · emotion × concepts", 12, TEXT2)]
    X = 610
    b += [f'<line x1="{X - 22}" y1="24" x2="{X - 22}" y2="{H - 24}" stroke="{CARD_STROKE}" stroke-width="1"/>']
    b += [text(X, 30, "Why concepts alone land on chance", 14, TEXT, weight=500)]
    prof = [0.8, 0.3, 0.6, 0.45]
    for r, l in enumerate(LANGS):
        for k in range(5):
            gx = X + k * 38
            gy = 96 + r * 62
            b += bars(gx, gy, prof, 5, 2, 30)
            b += [dot(gx + 13, gy + 12, 4, LANG[l]["dot"])]
    b += lines(X, 268, ["15 annotations of one painting, 5 per language,",
                        "all with the same concept scores:",
                        "any guess is right 5 times in 15, which is chance."], 12, TEXT2, lh=18)
    save("05_classifier_input", svg(W, H, b, "Language classifier inputs",
                                    "Concept scores and the chosen emotion feed a classifier that guesses the annotator's language; concept scores alone cannot beat chance."))


# ---------------------------------------------------------------- 6. swap test
PROF = {"A": [0.9, 0.25, 0.7, 0.4], "B": [0.3, 0.9, 0.2, 0.8], "C": [0.6, 0.7, 0.95, 0.12]}


def fig_swap():
    rows = [("english", "sadness", "A"), ("chinese", "contentment", "B"),
            ("arabic", "fear", "C"), ("chinese", "awe", "A")]
    swapped = ["C", "A", "B", "C"]
    W, H = 960, 300
    b = []
    for t, (title, x) in enumerate([("Real pairing", 24), ("Swapped profiles", 510)]):
        b += [text(x, 30, title, 14, TEXT, weight=500)]
        cols = [("language", 0), ("emotion", 104), ("painting", 210), ("concept profile", 290)]
        for name, dx in cols:
            b += [text(x + dx, 64, name, 11, MUTED)]
        b += [f'<line x1="{x}" y1="76" x2="{x + 420}" y2="76" stroke="{CARD_STROKE}"/>']
        for i, (lang, emo, p) in enumerate(rows):
            y = 104 + i * 44
            b += [dot(x + 6, y, 5, LANG[lang]["dot"]), text(x + 18, y, lang, 13, LANG[lang]["text"])]
            b += [text(x + 104, y, emo, 13, TEXT), text(x + 210, y, p, 13, TEXT)]
            src = p if t == 0 else swapped[i]
            hl = t == 1
            b += bars(x + 290, y + 14, PROF[src], 9, 3, 28, fill=ACC_MID if hl else BAR)
            b += [text(x + 344, y, f"from {src}", 13, ACC_TEXT if hl else TEXT2, weight=500 if hl else 400)]
    b += [arrow(452, 170, 494, 170)]
    b += [text(510, 278, "label habits stay intact; only the link to what is in the painting is broken", 11, TEXT2)]
    save("06_swap_test", svg(W, H, b, "Swap test",
                             "Each annotation keeps its language and emotion; only the concept profile it is paired with comes from another painting."))


# ---------------------------------------------------------------- 7. variance split
def signed_fill(v):
    a = abs(v)
    if a < 1e-9:
        return "#F1EFE8"
    if v > 0:
        return "#F0997B" if a >= 0.3 else "#F5C4B3" if a >= 0.08 else "#FAECE7"
    return "#85B7EB" if a >= 0.3 else "#B5D4F4" if a >= 0.08 else "#E6F1FB"


def fig_variance():
    picks = {
        "A": {"english": [1, 1, 0, 1, 0], "arabic": [1, 0, 1, 1, 0], "chinese": [1, 0, 0, 1, 1]},
        "B": {l: [0] * 5 for l in LANGS},
        "C": {"english": [0, 1, 0, 0, 0], "arabic": [0, 0, 0, 0, 1], "chinese": [0] * 5},
    }
    avg = 0.24
    pnt = {"A": 0.36, "B": -0.24, "C": -0.11}
    lng = {"english": 0.02, "arabic": 0.02, "chinese": -0.04}
    itr = {"A": [-0.02, -0.02, 0.04], "B": [-0.02, -0.02, 0.04], "C": [0.04, 0.04, -0.09]}
    CW, CH = 44, 40
    GW = 3 * CW
    OP = 30
    x0 = 44
    W = x0 + 6 * GW + 5 * OP + 24
    H = 300
    titles = ["who picked sadness", "average", "painting", "language", "painting × language", "left over"]
    ops = ["=", "+", "+", "+", "+"]
    b = []
    for g in range(6):
        gx = x0 + g * (GW + OP)
        b += [text(gx + GW / 2, 34, titles[g], 13 if g else 13, TEXT, "middle", 500)]
        for j, l in enumerate(LANGS):
            b += [f'<rect x="{gx + j * CW + 2}" y="54" width="{CW - 4}" height="3" rx="1.5" fill="{LANG[l]["dot"]}"/>']
        for r, p in enumerate("ABC"):
            cy = 64 + r * CH
            if g == 0:
                b += [text(24, cy + CH / 2, p, 12, TEXT2)]
            for j, l in enumerate(LANGS):
                cx = gx + j * CW
                mean = sum(picks[p][l]) / 5
                if g == 0:
                    b += [box(cx + 1, cy + 1, CW - 2, CH - 2, "#F6F5F1", "#E8E6DF", rx=4)]
                    for k, v in enumerate(picks[p][l]):
                        b += [dot(cx + 8 + k * 7, cy + CH / 2, 3, "#7F77DD" if v else "#D3D1C7")]
                elif g == 5:
                    b += [box(cx + 1, cy + 1, CW - 2, CH - 2, "#F6F5F1", "#E8E6DF", rx=4)]
                    for k, v in enumerate(picks[p][l]):
                        res = v - mean
                        col = "#D85A30" if res > 1e-9 else "#378ADD" if res < -1e-9 else "#D3D1C7"
                        b += [dot(cx + 8 + k * 7, cy + CH / 2, 3, col)]
                else:
                    v = {1: avg, 2: pnt[p], 3: lng[l], 4: itr[p][j]}[g]
                    fill = "#F1EFE8" if g == 1 else signed_fill(v)
                    b += [f'<rect x="{cx + 1}" y="{cy + 1}" width="{CW - 2}" height="{CH - 2}" rx="4" fill="{fill}"/>']
                    s = f"{v:.2f}" if g == 1 else f"{v:+.2f}"
                    b += [text(cx + CW / 2, cy + CH / 2, s, 11, TEXT, "middle")]
        if g < 5:
            b += [text(gx + GW + OP / 2, 64 + 1.5 * CH, ops[g], 20, MUTED, "middle")]
    b += [text(24, 212, "rows: paintings A, B, C · columns: english, arabic, chinese", 11, TEXT2)]
    b += [text(24, 232, "coral = above average, blue = below; a share is how much a part varies across all annotations", 11, TEXT2)]
    b += [dot(30, 262, 4, "#7F77DD"), text(40, 262, "picked sadness", 11, TEXT2),
          dot(148, 262, 4, "#D85A30"), text(158, 262, "above its cell's average", 11, TEXT2),
          dot(318, 262, 4, "#378ADD"), text(328, 262, "below it", 11, TEXT2)]
    save("07_variance_split", svg(W, H, b, "Variance split for one emotion",
                                  "Made-up choices for one emotion split into an average, a painting part, a language part, a painting by language part and what is left over."))


# ---------------------------------------------------------------- 8. encoder swap
def fig_encoder():
    W, H = 960, 250
    b = [text(24, 30, "Swap the encoder, keep everything else", 14, TEXT, weight=500)]
    b += [box(24, 92, 196, 76)] + lines(122, 112, ["same paintings", "same concept list", "same English prompts"], 12, TEXT, "middle", lh=18)
    for k, (lab, sub) in enumerate([("original CLIP", "mostly English"), ("multilingual CLIP", "also reads Arabic, Chinese")]):
        y = 62 + k * 92
        b += [arrow(224, 130, 280, y + 26)]
        b += [box(284, y, 190, 52, ACC_FILL, ACC_STROKE), text(379, y + 18, lab, 14, ACC_TEXT, "middle", 500), text(379, y + 36, sub, 11, ACC_TEXT, "middle")]
        b += [arrow(478, y + 26, 518, y + 26)]
        b += [box(522, y, 150, 52), text(597, y + 26, "concept scores", 13, TEXT, "middle")]
        b += [arrow(676, y + 26, 722, 130)]
    b += [box(726, 76, 210, 108)] + [text(831, 98, "the same tests", 13, TEXT, "middle", 500)]
    b += lines(831, 124, ["section 1 bottleneck", "section 4 classifier", "section 5 swap test"], 12, TEXT2, "middle", lh=18)
    save("08_encoder_swap", svg(W, H, b, "Encoder swap",
                                "The same paintings, concepts and prompts go through two CLIP encoders, and the same three tests are rerun on each."))


# ---------------------------------------------------------------- 9. vocabularies
def fig_vocab():
    W, H = 960, 400
    b = [text(24, 30, "Which concept list each language's head uses", 14, TEXT, weight=500)]
    words = {
        "generic": ["a boat · the sea · red"] * 3,
        "translated": ["a boat · the sea · red", "قارب · البحر · أحمر", "小船 · 大海 · 红色"],
        "native": ["colors · man · beautiful", "رجل · لون · جميلة", "感到 · 满足 · 女人"],
        "native, no|feeling words": ["colors · man · woman", "رجل · لون · اشجار", "女人 · 穿着 · 男人"],
    }
    cw, x0 = 152, 134
    for j, l in enumerate(LANGS):
        cx = x0 + j * (cw + 12)
        b += [text(cx + cw / 2, 62, f"{l} head", 12, LANG[l]["text"], "middle")]
        b += [f'<rect x="{cx}" y="72" width="{cw}" height="3" rx="1.5" fill="{LANG[l]["dot"]}"/>']
    for r, (name, ws) in enumerate(words.items()):
        y = 88 + r * 62
        label = name.split("|")
        b += lines(24, y + 22 - (len(label) - 1) * 8, label, 13, TEXT, weight=500, lh=16)
        for j in range(3):
            cx = x0 + j * (cw + 12)
            b += [box(cx, y, cw, 44), text(cx + cw / 2, y + 22, ws[j], 12, TEXT, "middle")]
    b += [text(24, 362, "all four scored with the multilingual CLIP; same list length", 11, TEXT2)]
    X = 660
    b += [text(X, 30, "Where native words come from", 14, TEXT, weight=500)]
    b += [box(X, 70, 150, 64)] + lines(X + 75, 92, ["71,244 other paintings", "(original ArtELingo)"], 12, TEXT, "middle", lh=18)
    b += [arrow(X + 75, 138, X + 75, 170)]
    b += lines(X + 75, 188, ["most common words", "that are concepts", "→ native lists"], 12, TEXT2, "middle", lh=17)
    b += [arrow(X + 75, 236, X + 75, 266)]
    b += [box(X, 270, 150, 64, ACC_FILL, ACC_STROKE)] + lines(X + 75, 292, ["feeling words labelled", "and dropped"], 12, ACC_TEXT, "middle", lh=18)
    b += [f'<line x1="{X + 170}" y1="62" x2="{X + 170}" y2="340" stroke="{BOX_STROKE}" stroke-dasharray="4 4"/>']
    b += [box(X + 186, 70, 92, 64, ACC_FILL, ACC_STROKE)] + lines(X + 232, 92, ["our 840", "paintings"], 12, ACC_TEXT, "middle", lh=18)
    b += lines(X + 232, 168, ["tested here;", "never mined,", "so nothing leaks"], 11, TEXT2, "middle", lh=16)
    save("09_vocabularies", svg(W, H, b, "Four concept vocabularies",
                                "Generic, translated, native and native-without-feeling-words concept lists per language head; native words are mined only from paintings outside the tested set."))


# ---------------------------------------------------------------- 10. edit test
def fig_edits():
    W, H = 960, 300
    concepts = ["boat", "sea", "sky", "cat", "bridge", "red"]
    base = [0.52, 0.4, 0.7, 0.18, 0.3, 0.36]
    panels = [("Concept scores", "explanation: “a lonely boat on a grey sea”", set()),
              ("Real edit", "raise what the explanation mentions", {0, 1}),
              ("Sham edit", "raise as many random concepts", {3, 4})]
    pw, gap = 290, 26
    b = []
    for i, (title, sub, raised) in enumerate(panels):
        x = 24 + i * (pw + gap)
        b += [text(x, 30, title, 14, TEXT, weight=500), text(x, 52, sub, 11, TEXT2)]
        ytop, ybase, hmax = 84, 224, 140
        b += [f'<line x1="{x}" y1="{ytop}" x2="{x + pw}" y2="{ytop}" stroke="{MUTED}" stroke-dasharray="2 3"/>',
              text(x + pw, ytop - 10, "high", 11, MUTED, "end")]
        bw, bg = 34, 13
        for k, v in enumerate(base):
            bx = x + 6 + k * (bw + bg)
            h = v * hmax
            b += [f'<rect x="{bx}" y="{ybase - h}" width="{bw}" height="{h}" rx="2" fill="{BAR}"/>']
            if k in raised:
                b += [f'<rect x="{bx}" y="{ytop}" width="{bw}" height="{ybase - h - ytop}" rx="2" fill="{ACC_MID}"/>']
            b += [text(bx + bw / 2, ybase + 16, concepts[k], 11, TEXT2, "middle")]
    b += [text(W / 2, 274, "then compare the bottleneck's predictions for the Arabic and Chinese annotators of the same painting", 12, TEXT2, "middle")]
    save("10_edit_test", svg(W, H, b, "Concept edit test",
                             "Real edits raise the concepts an English explanation mentions; sham edits raise as many random concepts."))


if __name__ == "__main__":
    fig_bottleneck()
    fig_chain(False)
    fig_chain(True)
    fig_shuffles()
    fig_something_else()
    fig_classifier()
    fig_swap()
    fig_variance()
    fig_encoder()
    fig_vocab()
    fig_edits()
    print(sorted(p.name for p in OUT.iterdir()))
