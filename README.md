# ArtELingo CBM

Course project for Human-Centred Machine Learning. One notebook,
[`notebooks/01_tuesday_poc.ipynb`](notebooks/01_tuesday_poc.ipynb), builds a concept
bottleneck model on ArtELingo (painting → visual concepts → emotion) and tests the
assumptions behind it. It is committed with its outputs, so you can read it on GitHub
without running anything.

## Run it

You need [uv](https://docs.astral.sh/uv/) and one data file.

1. Clone and install (uv fetches Python 3.11 if needed):

   ```bash
   git clone <repo-url> && cd CBM
   uv sync
   ```

2. Put the ArtELingo-28 training file at `data/raw/artelingo28_train_final.csv`. It is
   on the team drive (originally from [artelingo.org](https://www.artelingo.org/)). It
   is the only file you have to provide. `data/` is gitignored, so it never gets
   committed.

3. Open the notebook and run Kernel → Restart & Run All:

   ```bash
   uv run jupyter lab notebooks/01_tuesday_poc.ipynb
   ```

**The first run takes longer**, roughly half an hour. It downloads the 840 paintings
from wikiart.org (slowly, one request a second) and two CLIP models (about 1.5 GB), and
caches everything in `data/`. After that a full run takes about 5 minutes on a Mac
laptop. No accounts or API keys are needed.

## What else is here

- `config.yaml`: every setting the notebook uses.
- `src/artcbm/`: the helper functions the notebook calls.
- `concepts/`: the concept lists. `generic_v1.txt` is the main one. The `native_v1_*`
  files are word lists mined from annotators' explanations, with their labels.
- `scripts/`: you don't need these to run the notebook.
  - `mine_native_vocab.py` rebuilds the native word lists. It needs the original
    ArtELingo release at `data/raw/artelingo_release.csv` (1 GB).
  - `check_feeling_words.py` lets you label native words by hand, with y/n/x answers.
  - `make_method_figures.py` regenerates the method diagrams in
    `notebooks/figures/methods/`.
- `outputs/figures/`: the notebook's result figures as PNGs.

Before committing, run `uv run ruff check . && uv run ruff format .`
