"""Divergences, variance split, inverted CBM and permutation nulls."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 1. disagreement map


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (base 2, range 0 to 1) along the last axis."""
    m = (p + q) / 2

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / b), 0.0)
        return t.sum(axis=-1)

    return (kl(p, m) + kl(q, m)) / 2


def split_half_divergence(labels: np.ndarray, n_emotions: int, n_boot: int, seed: int):
    """Within-language vs cross-language JSD with matched sample sizes.

    `labels` has shape (paintings, languages, n). Each resample shuffles every
    (painting, language) cell and takes two disjoint halves of size h = n // 2.
    Within-language compares a language's two halves; cross-language compares half 1
    of one language with half 2 of another. Both sides compare two samples of size h,
    so small-sample bias in JSD affects them equally.

    Returns (within, cross, pairs): within is (paintings, languages), cross is
    (paintings, n_pairs), both averaged over resamples.
    """
    rng = np.random.default_rng(seed)
    n_p, n_l, n = labels.shape
    h = n // 2
    pairs = list(combinations(range(n_l), 2))
    eye = np.eye(n_emotions)
    within = np.zeros((n_p, n_l))
    cross = np.zeros((n_p, len(pairs)))
    for _ in range(n_boot):
        shuffled = rng.permuted(labels, axis=2)
        d1 = eye[shuffled[:, :, :h]].mean(axis=2)  # (paintings, languages, emotions)
        d2 = eye[shuffled[:, :, h : 2 * h]].mean(axis=2)
        within += jsd(d1, d2)
        cross += np.stack([jsd(d1[:, i], d2[:, j]) for i, j in pairs], axis=1)
    return within / n_boot, cross / n_boot, pairs


def shuffle_within_painting(labels: np.ndarray, rng) -> np.ndarray:
    """Reassign each painting's annotations to languages at random (same cell sizes)."""
    n_p, n_l, n = labels.shape
    return rng.permuted(labels.reshape(n_p, n_l * n), axis=1).reshape(n_p, n_l, n)


def divergence_gap_null(labels, n_emotions, n_perm, n_boot, seed) -> np.ndarray:
    """Null distribution of mean(cross) - mean(within) when language labels are
    shuffled within each painting, i.e. when languages are exchangeable."""
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_perm)
    for k in range(n_perm):
        w, c, _ = split_half_divergence(
            shuffle_within_painting(labels, rng), n_emotions, n_boot, seed + 1 + k
        )
        gaps[k] = c.mean() - w.mean()
    return gaps


# ---------------------------------------------------------------- 3. variance split


def variance_split(df: pd.DataFrame, emotions) -> pd.DataFrame:
    """Two-way decomposition of one-hot emotion indicators into painting, language,
    painting x language and residual sums of squares.

    Assumes a balanced design (same n in every painting x language cell), where the
    decomposition is exact and equals sequential OLS sums of squares. The last row
    sums the SS over all emotion indicators.

    `noise_floor` is the interaction share expected with no true interaction:
    df_interaction * residual mean square / total SS.
    """
    n_p, n_l = df["painting"].nunique(), df["language"].nunique()
    n = len(df) // (n_p * n_l)
    y = pd.get_dummies(df["emotion"]).reindex(columns=list(emotions), fill_value=0)
    y = y.astype(float)
    y[["painting", "language"]] = df[["painting", "language"]]

    grand = y[list(emotions)].mean()
    m_p = y.groupby("painting")[list(emotions)].mean()
    m_l = y.groupby("language")[list(emotions)].mean()
    m_pl = y.groupby(["painting", "language"])[list(emotions)].mean()

    ss_p = n_l * n * ((m_p - grand) ** 2).sum()
    ss_l = n_p * n * ((m_l - grand) ** 2).sum()
    inter = (
        m_pl
        - m_p.reindex(m_pl.index, level="painting").to_numpy()
        - m_l.reindex(m_pl.index, level="language").to_numpy()
        + grand
    )
    ss_pl = n * (inter**2).sum()
    cell = m_pl.loc[list(zip(y["painting"], y["language"], strict=True))].to_numpy()
    ss_res = pd.Series(((y[list(emotions)].to_numpy() - cell) ** 2).sum(axis=0), emotions)

    out = pd.DataFrame(
        {"painting": ss_p, "language": ss_l, "interaction": ss_pl, "residual": ss_res}
    )
    out.loc["all emotions"] = out.sum()
    total = out.sum(axis=1)
    df_pl, df_res = (n_p - 1) * (n_l - 1), n_p * n_l * (n - 1)
    shares = out.div(total, axis=0)
    shares["noise_floor"] = df_pl * (out["residual"] / df_res) / total
    return shares


# ---------------------------------------------------------------- 2/4. inverted CBM


def make_features(df, kind: str, emotions, concept_scores: pd.DataFrame | None = None):
    """Feature matrix for the inverted CBM.

    kind: "emotion" (one-hot), "concepts" (painting-level concept scores),
    "both" (concepts + emotion) or "interaction" (emotion + concept x emotion).
    Returns (X, feature_names).
    """
    e = pd.get_dummies(df["emotion"]).reindex(columns=list(emotions), fill_value=0)
    e_names = [f"emo={x}" for x in emotions]
    e = e.to_numpy(dtype=float)
    if kind == "emotion":
        return e, e_names
    c = concept_scores.loc[df["painting"]].to_numpy()
    c_names = list(concept_scores.columns)
    if kind == "concepts":
        return c, c_names
    if kind == "both":
        return np.hstack([c, e]), c_names + e_names
    if kind == "interaction":
        ce = (c[:, :, None] * e[:, None, :]).reshape(len(df), -1)
        ce_names = [f"{cn} | {en}" for cn in c_names for en in emotions]
        return np.hstack([e, ce]), e_names + ce_names
    raise ValueError(kind)


def concept_components(scores: pd.DataFrame, n: int, seed: int):
    """Summarise correlated concept scores into `n` principal components.

    Returns (components, loadings): components is paintings x n, z-scored;
    loadings is n x concepts. Uses no labels, only the painting images.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n, random_state=seed).fit(scores.to_numpy())
    names = [f"D{i + 1}" for i in range(n)]
    comp = pca.transform(scores.to_numpy())
    comp = (comp - comp.mean(axis=0)) / comp.std(axis=0)
    components = pd.DataFrame(comp, index=scores.index, columns=names)
    loadings = pd.DataFrame(pca.components_, index=names, columns=scores.columns)
    return components, loadings


def describe_components(loadings: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """The concepts at the high and low end of each component."""
    return pd.DataFrame(
        {
            "high": [", ".join(row.nlargest(k).index) for _, row in loadings.iterrows()],
            "low": [", ".join(row.nsmallest(k).index) for _, row in loadings.iterrows()],
        },
        index=loadings.index,
    )


def logreg(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def cv_balanced_accuracy(X, y, groups, n_folds, C, seed) -> float:
    """Balanced accuracy of pooled out-of-fold predictions, GroupKFold by painting."""
    pred = np.empty_like(y)
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(X, y, groups):
        pred[test] = logreg(C).fit(X[train], y[train]).predict(X[test])
    return balanced_accuracy_score(y, pred)


def shuffle_labels_within(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Permute labels within each group (painting)."""
    out = y.copy()
    for idx in pd.Series(np.arange(len(y))).groupby(groups).indices.values():
        out[idx] = y[rng.permutation(idx)]
    return out


def _parallel(fn, args_per_task, n_jobs=-1) -> np.ndarray:
    """Run fn over tasks in parallel. Arrays go in as arguments so joblib memory-maps
    them; one BLAS thread per worker avoids oversubscribing the cores."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return np.array(Parallel(n_jobs=n_jobs)(delayed(fn)(*a) for a in args_per_task))


def permutation_null(X, y, groups, n_perm, n_folds, C, seed) -> np.ndarray:
    """Balanced accuracies after shuffling language within painting, n_perm times."""
    return _parallel(
        _one_permutation, ((X, y, groups, n_folds, C, seed, seed + 1 + k) for k in range(n_perm))
    )


def _one_permutation(X, y, groups, n_folds, C, seed, perm_seed):
    y_perm = shuffle_labels_within(y, groups, np.random.default_rng(perm_seed))
    return cv_balanced_accuracy(X, y_perm, groups, n_folds, C, seed)


def concept_swap_null(df, emotions, components, n_swap, n_folds, C, seed, kind="interaction"):
    """Balanced accuracies when each painting gets a random other painting's concept
    profile. Languages and emotions stay as labelled, so label habits are untouched;
    only the link between painting content and the annotations is broken."""
    return _parallel(
        _one_swap,
        ((df, emotions, components, kind, n_folds, C, seed, seed + 1 + k) for k in range(n_swap)),
    )


def _one_swap(df, emotions, components, kind, n_folds, C, seed, swap_seed):
    swapped = components.copy()
    swapped.index = np.random.default_rng(swap_seed).permutation(components.index)
    X, _ = make_features(df, kind, emotions, swapped)
    y, groups = df["language"].to_numpy(), df["painting"].to_numpy()
    return cv_balanced_accuracy(X, y, groups, n_folds, C, seed)


def permutation_p(real: float, null: np.ndarray) -> float:
    """One-sided p-value with the +1 correction."""
    return (1 + (null >= real).sum()) / (1 + len(null))
