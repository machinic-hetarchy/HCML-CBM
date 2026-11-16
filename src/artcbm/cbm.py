"""Forward concept bottleneck: concept scores -> each language's emotion distribution."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from artcbm.analysis import _parallel, logreg


def _per_language(features, languages) -> dict:
    return features if isinstance(features, dict) else {lang: features for lang in languages}


def _folds(df, n_folds, seed):
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(folds.split(df, groups=df["painting"]))


def _brier(model, X, y, emotions) -> float:
    onehot = (np.asarray(y)[:, None] == np.asarray(model.classes_)[None, :]).astype(float)
    return ((onehot - model.predict_proba(X)) ** 2).sum(axis=1).mean()


def choose_C(X, y, groups, Cs, seed, n_inner=3) -> float:
    """The C with the lowest squared error in an inner GroupKFold (training rows only)."""
    inner = GroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    splits = list(inner.split(X, y, groups))
    loss = [
        np.mean([_brier(logreg(c).fit(X[a], y[a]), X[b], y[b], None) for a, b in splits])
        for c in Cs
    ]
    return Cs[int(np.argmin(loss))]


def forward_cv(df, features, emotions, languages, n_folds, C, seed):
    """Out-of-fold emotion probabilities for every annotation.

    One logistic model per language (its own head) maps a painting's features to that
    language's emotion choices. Training on individual annotations is the same as fitting
    each painting's soft label with cross-entropy. Folds are by painting and shared by all
    languages. `features` is a painting x feature table, or a dict of them per language.
    `C` is one value, or a list to choose from inside each training fold.

    Returns (probs, fits): probs is rows x emotions; fits is a list of
    (test_rows, language, model) for re-predicting with edited features.
    """
    features = _per_language(features, languages)
    emotions, lang, y = list(emotions), df["language"].to_numpy(), df["emotion"].to_numpy()
    paintings = df["painting"].to_numpy()
    probs, fits = np.zeros((len(df), len(emotions))), []
    for train, test in _folds(df, n_folds, seed):
        for language in languages:
            tr, te = train[lang[train] == language], test[lang[test] == language]
            table = features[language]
            X_tr = table.loc[paintings[tr]].to_numpy()
            c = choose_C(X_tr, y[tr], paintings[tr], C, seed) if isinstance(C, list) else C
            model = logreg(c).fit(X_tr, y[tr])
            cols = [emotions.index(c) for c in model.classes_]
            probs[np.ix_(te, cols)] = model.predict_proba(
                table.loc[df["painting"].iloc[te]].to_numpy()
            )
            fits.append((te, language, model))
    return pd.DataFrame(probs, index=df.index, columns=emotions), fits


def repredict(df, fits, features, emotions, languages=None) -> pd.DataFrame:
    """Probabilities from already fitted fold models, for (possibly edited) features.
    Only rows of `languages` are filled (all by default); the rest are NaN."""
    features = _per_language(features, list({f[1] for f in fits}))
    emotions = list(emotions)
    probs = np.full((len(df), len(emotions)), np.nan)
    for te, language, model in fits:
        if languages is not None and language not in languages:
            continue
        cols = [emotions.index(c) for c in model.classes_]
        probs[te] = 0.0
        probs[np.ix_(te, cols)] = model.predict_proba(
            features[language].loc[df["painting"].iloc[te]].to_numpy()
        )
    return pd.DataFrame(probs, index=df.index, columns=emotions)


def prior_cv(df, emotions, n_folds, seed) -> pd.DataFrame:
    """Baseline: each language's emotion mix in the training folds."""
    probs = pd.DataFrame(0.0, index=df.index, columns=list(emotions))
    for train, test in _folds(df, n_folds, seed):
        mix = df.iloc[train].groupby("language")["emotion"].value_counts(normalize=True)
        mix = mix.unstack(fill_value=0).reindex(columns=list(emotions), fill_value=0)
        probs.iloc[test] = mix.loc[df["language"].iloc[test]].to_numpy()
    return probs


def other_annotators(df, emotions, n_folds, seed, ks=(1, 2, 5, 10, 20, 50, 100)):
    """Human reference: predict each annotation from everyone else who labelled the same
    painting (all languages), pulled toward the annotator's language mix by k
    pseudo-annotations: (others' counts + k * language mix) / (others + k).
    k is chosen on the training folds, the language mix comes from them too."""
    emotions = list(emotions)
    onehot = pd.get_dummies(df["emotion"]).reindex(columns=emotions, fill_value=0)
    onehot = onehot.to_numpy(float)
    totals = pd.DataFrame(onehot).groupby(df["painting"].to_numpy()).transform("sum")
    others = totals.to_numpy() - onehot
    n_others = others.sum(axis=1, keepdims=True)
    probs = np.zeros_like(onehot)
    for train, test in _folds(df, n_folds, seed):
        mix = df.iloc[train].groupby("language")["emotion"].value_counts(normalize=True)
        mix = mix.unstack(fill_value=0).reindex(columns=emotions, fill_value=0)
        prior = mix.loc[df["language"]].to_numpy()

        def predict(k, rows, prior=prior):
            return (others[rows] + k * prior[rows]) / (n_others[rows] + k)

        sse = [((onehot[train] - predict(k, train)) ** 2).sum() for k in ks]
        probs[test] = predict(ks[int(np.argmin(sse))], test)
    return pd.DataFrame(probs, index=df.index, columns=emotions)


def explained(df, probs, baseline, emotions, rows=None) -> pd.Series:
    """Share of the baseline's squared error that the model removes, per emotion and
    summed over emotions (1 - SSE_model / SSE_baseline). 0 = no better than baseline."""
    onehot = pd.get_dummies(df["emotion"]).reindex(columns=list(emotions), fill_value=0)
    onehot = onehot.astype(float)
    if rows is not None:
        onehot, probs, baseline = onehot[rows], probs[rows], baseline[rows]
    sse = ((onehot - probs) ** 2).sum()
    sse_base = ((onehot - baseline) ** 2).sum()
    out = 1 - sse / sse_base
    out["all emotions"] = 1 - sse.sum() / sse_base.sum()
    return out


def forward_swap_null(df, features, emotions, languages, n_swap, n_folds, C, seed):
    """Overall explained share when each painting gets a random other painting's
    features (labels untouched), n_swap times."""
    baseline = prior_cv(df, emotions, n_folds, seed)
    tasks = (
        (df, features, emotions, languages, n_folds, C, seed, seed + 1 + k, baseline)
        for k in range(n_swap)
    )
    return _parallel(_one_forward_swap, tasks)


def _one_forward_swap(df, features, emotions, languages, n_folds, C, seed, swap_seed, baseline):
    rng = np.random.default_rng(swap_seed)
    swapped = {}
    for lang, table in _per_language(features, languages).items():
        table = table.copy()
        table.index = rng.permutation(table.index)
        swapped[lang] = table
    probs, _ = forward_cv(df, swapped, emotions, languages, n_folds, C, seed)
    return explained(df, probs, baseline, emotions)["all emotions"]


def bootstrap_difference(df, probs_a, probs_b, baseline, emotions, n_boot, seed, rows=None):
    """Bootstrap over paintings of explained(a) - explained(b), all emotions."""
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), bool) if rows is None else np.asarray(rows)
    paintings = df["painting"].to_numpy()
    by_painting = pd.Series(np.arange(len(df))).groupby(paintings).indices
    names = np.array(list(by_painting))
    onehot = pd.get_dummies(df["emotion"]).reindex(columns=list(emotions), fill_value=0)
    err = {
        k: ((onehot.to_numpy(float) - p.to_numpy()) ** 2).sum(axis=1)
        for k, p in [("a", probs_a), ("b", probs_b), ("base", baseline)]
    }
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([by_painting[p] for p in rng.choice(names, len(names))])
        idx = idx[keep[idx]]
        base = err["base"][idx].sum()
        out[i] = (base - err["a"][idx].sum()) / base - (base - err["b"][idx].sum()) / base
    return out


# ---------------------------------------------------------------- interventions


def keyword(concept: str) -> str | None:
    """The single content word of a concept ('a horse' -> 'horse'), or None if it has
    several ('warm colours'), since those cannot be matched reliably in text."""
    words = [w for w in concept.lower().split() if w not in {"a", "an", "the"}]
    return words[0] if len(words) == 1 else None


def mentions(captions: pd.DataFrame, concepts) -> pd.DataFrame:
    """painting x concept: does any explanation of the painting mention the concept's
    keyword (singular or plural, whole word)?"""
    keys = {c: keyword(c) for c in concepts if keyword(c)}
    text = captions.groupby("painting")["caption"].apply(lambda s: " ".join(s).lower())
    return pd.DataFrame(
        {c: text.str.contains(rf"\b{re.escape(k)}(?:s|es)?\b", regex=True) for c, k in keys.items()}
    )


def raise_concepts(scores: pd.DataFrame, chosen: pd.DataFrame, level: pd.Series):
    """Set the chosen concepts (painting x concept, bool) to at least `level`."""
    edited = scores.astype(float)
    cols = list(chosen.columns)
    block = edited.loc[chosen.index, cols]
    raised = np.maximum(block.to_numpy(), level[cols].to_numpy())
    edited.loc[chosen.index, cols] = np.where(chosen.to_numpy(), raised, block.to_numpy())
    return edited


def sham_choice(mentioned: pd.DataFrame, rng) -> pd.DataFrame:
    """For each painting, as many random concepts as it has mentions, drawn from the
    mentionable concepts it does not mention."""
    m = mentioned.to_numpy()
    out = np.zeros_like(m)
    for i, row in enumerate(m):
        free = np.flatnonzero(~row)
        k = min(row.sum(), len(free))
        out[i, rng.choice(free, k, replace=False)] = True
    return pd.DataFrame(out, index=mentioned.index, columns=mentioned.columns)


def concept_pushes(df, fits, features, painting: str, language: str) -> pd.DataFrame:
    """How much each concept pushes one painting's prediction toward each emotion, in the
    fold model that never saw the painting: (coefficient - its mean over emotions) x the
    painting's standardised concept score. Returns emotions x concepts."""
    for te, lang, model in fits:
        if lang == language and (df["painting"].iloc[te] == painting).any():
            x = model[0].transform(features.loc[[painting]].to_numpy())[0]
            coef = model[-1].coef_
            push = (coef - coef.mean(axis=0)) * x
            return pd.DataFrame(push, index=model[-1].classes_, columns=features.columns)
    raise KeyError(f"no fold model for {painting} ({language})")
