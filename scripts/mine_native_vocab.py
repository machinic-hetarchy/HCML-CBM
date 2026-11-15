"""Mine native concept vocabularies from the original ArtELingo release.

Counts, per language, the words that appear in the most explanations, using only
paintings outside the notebook's set (so nothing leaks into the paintings we test on).
Writes concepts/native_v1_<language>.tsv (rank, word, explanations). The feeling-word
labels are added to the same files by hand afterwards; rerunning overwrites them, so run
this only to rebuild the lists from scratch.

    uv run python scripts/mine_native_vocab.py
"""

from __future__ import annotations

import pandas as pd

from artcbm import data, vocab
from artcbm.config import REPO_ROOT, load_config

RELEASE = REPO_ROOT / "data" / "raw" / "artelingo_release.csv"
N_WORDS = 500  # more than the vocabulary length, so labelled words can be dropped


def main() -> None:
    cfg = load_config()
    langs = list(cfg.data.languages)
    ours = data.filter_paintings(
        data.load_annotations(cfg),
        langs,
        cfg.data.min_annotations_per_language,
        cfg.data.max_paintings,
        cfg.seed,
    )["painting"].unique()
    release = pd.read_csv(RELEASE, usecols=["painting", "language", "utterance"])
    outside = release[~release["painting"].isin(ours) & release["language"].isin(langs)]
    print(f"{outside['painting'].nunique()} paintings outside our {len(ours)}")
    for lang in langs:
        captions = outside.loc[outside["language"] == lang, "utterance"].drop_duplicates()
        words, counts = vocab.mine_vocabulary(captions, lang, N_WORDS, with_counts=True)
        out = pd.DataFrame(
            {"rank": range(1, len(words) + 1), "word": words, "explanations": counts}
        )
        path = REPO_ROOT / "concepts" / f"native_v1_{lang}.tsv"
        out.to_csv(path, sep="\t", index=False)
        print(f"{lang}: {len(captions)} explanations -> {path.name}")


if __name__ == "__main__":
    main()
