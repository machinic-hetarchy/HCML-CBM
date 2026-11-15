"""Label native words, blind to the model's labels.

    uv run python scripts/check_feeling_words.py chinese
    uv run python scripts/check_feeling_words.py english
    uv run python scripts/check_feeling_words.py chinese --from 1   # revisit from word 1

y = feeling word: names an emotion, a mood or a judgement ("sad", "peaceful", "beautiful",
    "boring").
n = concept: something one could point to in the painting, including visible qualities
    such as "dark" or "smile".
x = not a concept at all: function words, fillers and fragments ("don't", "maybe", 不由).

Other keys: Enter = keep the current answer (when revisiting), b = back one word,
q = save and quit. Answers are saved after every word to
concepts/native_v1_<language>_check.tsv, and a rerun continues where you stopped.
"""

from __future__ import annotations

import argparse

import pandas as pd

from artcbm.config import REPO_ROOT

KEYS = {"y": "feeling", "n": "concept", "x": "none"}
SHORT = {v: k for k, v in KEYS.items()}
GUIDE = __doc__.split("\n\n")[2]
N_CHECK = 300  # the most common words, which make up most of each list


def main(language: str, start: int | None) -> None:
    path = REPO_ROOT / "concepts" / f"native_v1_{language}_check.tsv"
    if path.exists():
        sheet = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"label": str})
    else:
        words = pd.read_csv(REPO_ROOT / "concepts" / f"native_v1_{language}.tsv", sep="\t")
        sheet = words[["rank", "word"]].head(N_CHECK).assign(label="")
    print(GUIDE, "\n\nEnter = keep, b = back, q = save and quit\n")
    if start is None:
        i = next((k for k, v in enumerate(sheet["label"]) if v == ""), len(sheet))
    else:
        i = max(start - 1, 0)
    while i < len(sheet):
        current = sheet.at[i, "label"]
        shown = f" [{SHORT[current]}]" if current else ""
        prompt = f"[{i + 1}/{len(sheet)}]  {sheet.at[i, 'word']}{shown}  > "
        answer = input(prompt).strip().lower()
        if answer in KEYS:
            sheet.at[i, "label"] = KEYS[answer]
            i += 1
        elif answer == "" and current:
            i += 1
        elif answer == "b":
            i = max(i - 1, 0)
        elif answer == "q":
            break
        else:
            print("  y, n, x, b or q")
            continue
        sheet.to_csv(path, sep="\t", index=False)
    sheet.to_csv(path, sep="\t", index=False)
    done = (sheet["label"] != "").sum()
    print(f"\n{done} of {len(sheet)} labelled, saved to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label native words (blind check).")
    parser.add_argument("language", choices=["chinese", "english", "arabic"])
    parser.add_argument("--from", dest="start", type=int, help="revisit from this word number")
    args = parser.parse_args()
    main(args.language, args.start)
