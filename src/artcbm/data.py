"""Loading, filtering and balancing ArtELingo-28 annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from artcbm.config import Config

COLUMNS = ["painting", "image_name", "art_style", "genre", "language", "emotion", "caption"]


def load_annotations(cfg: Config) -> pd.DataFrame:
    """Read the annotation table for the configured languages, cached as parquet.

    Checks that the language and emotion strings in config.yaml exist in the data.
    """
    path = cfg.data.annotations_parquet
    if path.exists():
        return pd.read_parquet(path)

    raw = pd.read_csv(cfg.data.annotations_csv, usecols=COLUMNS)
    missing = set(cfg.data.languages) - set(raw["language"])
    if missing:
        raise ValueError(f"Languages not in data: {missing}")
    df = raw[raw["language"].isin(cfg.data.languages)].reset_index(drop=True)
    unexpected = set(df["emotion"]) - set(cfg.data.emotions)
    if unexpected:
        raise ValueError(f"Emotion labels not in config: {unexpected}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def filter_paintings(
    df: pd.DataFrame, languages, min_per_language: int, max_paintings: int, seed: int
) -> pd.DataFrame:
    """Keep paintings with at least `min_per_language` annotations in every language,
    then subsample to at most `max_paintings` paintings."""
    counts = df.groupby(["painting", "language"]).size().unstack(fill_value=0)
    ok = counts.index[(counts[list(languages)] >= min_per_language).all(axis=1)]
    if len(ok) > max_paintings:
        rng = np.random.default_rng(seed)
        ok = pd.Index(rng.choice(ok, size=max_paintings, replace=False))
    return df[df["painting"].isin(ok)].reset_index(drop=True)


def balance_per_painting(df: pd.DataFrame, n_per_language: int, seed: int) -> pd.DataFrame:
    """Sample exactly `n_per_language` annotations for every (painting, language) cell.

    Afterwards every painting contributes the same number of rows per language, so a
    model cannot predict language from which paintings a language happened to annotate.
    """
    out = df.groupby(["painting", "language"], group_keys=False).sample(
        n=n_per_language, random_state=seed
    )
    return out.sort_values(["painting", "language"]).reset_index(drop=True)


def label_array(df: pd.DataFrame, languages, emotions) -> np.ndarray:
    """Integer emotion codes shaped (n_paintings, n_languages, n_per_language).

    Expects a balanced table from `balance_per_painting`, sorted by painting.
    """
    code = {e: i for i, e in enumerate(emotions)}
    lang = {lg: i for i, lg in enumerate(languages)}
    paintings = df["painting"].unique()
    n = df.groupby(["painting", "language"]).size().iloc[0]
    arr = np.empty((len(paintings), len(languages), n), dtype=int)
    p_idx = pd.Series(np.arange(len(paintings)), index=paintings)
    d = df.assign(_rank=df.groupby(["painting", "language"]).cumcount())
    arr[
        p_idx[d["painting"]].to_numpy(),
        d["language"].map(lang).to_numpy(),
        d["_rank"].to_numpy(),
    ] = d["emotion"].map(code).to_numpy()
    return arr


WIKIART_IMG = "https://uploads.wikiart.org/images/{artist}/{title}.jpg"
WIKIART_PAGE = "https://www.wikiart.org/en/{artist}/{title}"
USER_AGENT = "artcbm-research/0.1 (university course project)"


def _wikiart_image_url(artist: str, title: str, session) -> str | None:
    """Direct upload URL if it exists, else the og:image of the painting's page."""
    import re

    url = WIKIART_IMG.format(artist=artist, title=title)
    if session.head(url, timeout=20).status_code == 200:
        return url
    page = session.get(WIKIART_PAGE.format(artist=artist, title=title), timeout=20)
    m = re.search(r'<meta property="og:image" content="([^"]+)"', page.text)
    if page.ok and m and "/images/" in m.group(1):
        return m.group(1)
    return None


def fetch_wikiart_images(image_names, images_dir, delay: float = 1.0) -> pd.DataFrame:
    """Download `Style/artist_title.jpg` images from wikiart.org into `images_dir`.

    Skips files already on disk and paintings already recorded as gone, waits `delay`
    seconds between requests, and writes the paintings it could not get to
    `images_dir/missing.csv` (also returned). Timeouts are retried on the next call.
    """
    import io
    import time

    import requests
    from PIL import Image
    from tqdm import tqdm

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    missing, known_gone = [], set()
    if (images_dir / "missing.csv").exists():
        prev = pd.read_csv(images_dir / "missing.csv")
        gone = prev[prev["reason"] == "not on wikiart.org"]
        missing, known_gone = gone.to_dict("records"), set(gone["image_name"])
    todo = sorted(n for n in set(image_names) - known_gone if not (images_dir / n).exists())
    for name in tqdm(todo, desc="wikiart.org", disable=not todo):
        path = images_dir / name
        artist, title = name.split("/", 1)[1].removesuffix(".jpg").split("_", 1)
        try:
            url = _wikiart_image_url(artist, title, session)
            if url is None:
                raise FileNotFoundError("not on wikiart.org")
            img = Image.open(io.BytesIO(session.get(url, timeout=60).content)).convert("RGB")
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, quality=95)
        except Exception as err:  # record and move on; a rerun retries missing ones
            page = WIKIART_PAGE.format(artist=artist, title=title)
            missing.append({"image_name": name, "page": page, "reason": str(err)[:80]})
        time.sleep(delay)
    out = pd.DataFrame(missing, columns=["image_name", "page", "reason"])
    images_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(images_dir / "missing.csv", index=False)
    return out
