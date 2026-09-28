"""
Part 2 utilities for the SIT307 11.1HD study: a leakage-free, nested
cross-validated stacking protocol. Builds on heart_utils.py (Part 1).
"""

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from heart_utils import (FEATURES, TARGET, make_preprocessor, make_base_classifiers,
                         make_stacking, evaluate, row_keys)

BASE_ORDER = ["LR", "DT", "RF", "XGB", "NB", "KNN"]

# 1. Patient identity (exact duplicate rows = same record)
def add_record_id(df):
    """Assign the same record_id to all exact copies of a row."""
    keys = row_keys(df[FEATURES + [TARGET]])
    codes, _ = pd.factorize(keys)
    return df.assign(record_id=codes)


def deduplicate(df):
    """Keep one copy of every record (first occurrence)."""
    return df.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)


# 2. Hyperparameter grids (tuned with inner stratified CV, training data only)
#    Small, interpretable grids chosen to control model complexity, the main
#    driver of memorisation on this small dataset.
PARAM_GRIDS = {
    "LR":  {"clf__C": [0.01, 0.1, 1, 10]},
    "DT":  {"clf__max_depth": [2, 3, 4, 5, None], "clf__min_samples_leaf": [1, 5, 10]},
    "RF":  {"clf__n_estimators": [200], "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.5]},
    "XGB": {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.05, 0.1]},
    "NB":  {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    "KNN": {"clf__n_neighbors": [5, 9, 15, 21, 31], "clf__weights": ["uniform", "distance"]},
}
TUNING_SCORING = "roc_auc"   # threshold-independent; matches the probability inputs of stacking


def tune_base_models(X_train, y_train, seed, inner_splits=5, n_jobs=-1):
    """
    Inner loop of nested CV: grid-search every base classifier on the
    training fold only. Returns {name: tuned classifier (unfitted clone)} and
    {name: best params}.
    """
    base = make_base_classifiers(seed)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    tuned, best = {}, {}
    for name in BASE_ORDER:
        pipe = Pipeline([("prep", make_preprocessor(seed)), ("clf", base[name])])
        gs = GridSearchCV(pipe, PARAM_GRIDS[name], scoring=TUNING_SCORING, cv=inner,
                          n_jobs=n_jobs, refit=False)
        gs.fit(X_train, y_train)
        params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        tuned[name] = clone(base[name]).set_params(**params)
        best[name] = params
    return tuned, best


def make_tuned_pipelines(tuned, seed, stack_cv=5):
    """Pipelines for the six tuned classifiers and the stacking of all six."""
    pipes = {n: Pipeline([("prep", make_preprocessor(seed)), ("clf", clone(c))])
             for n, c in tuned.items()}
    stack = StackingClassifier(
        estimators=[(n, clone(c)) for n, c in tuned.items()],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=StratifiedKFold(n_splits=stack_cv, shuffle=True, random_state=seed),
        stack_method="predict_proba", n_jobs=1)
    pipes["Stacking"] = Pipeline([("prep", make_preprocessor(seed)), ("clf", stack)])
    return pipes


def make_default_pipelines(seed):
    """Paper configuration (library defaults), for the leakage-free re-evaluation."""
    pipes = {n: Pipeline([("prep", make_preprocessor(seed)), ("clf", c)])
             for n, c in make_base_classifiers(seed).items()}
    pipes["Stacking"] = Pipeline([("prep", make_preprocessor(seed)), ("clf", make_stacking(seed))])
    return pipes


def score_model(pipe, X_train, y_train, X_test, y_test):
    model = clone(pipe).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    res = evaluate(y_test, y_pred, y_score)
    res["Brier"] = brier_score_loss(y_test, y_score)
    return res


# 3. Statistics
def corrected_resampled_ttest(a, b, n_train, n_test):
    """
    Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV.
    a, b: per-fold scores of two models on the SAME outer folds.
    The variance term is inflated by n_test/n_train because training sets of
    different folds overlap, which makes the ordinary paired t-test too liberal.
    Returns mean difference (a - b), t statistic, two-sided p-value.
    """
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return d.mean(), np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(np.abs(t), df=J - 1)
    return d.mean(), t, p


def mean_ci(x, n_train, n_test, level=0.95):
    """Mean and corrected CI half-width (same variance correction as above)."""
    x = np.asarray(x); J = len(x)
    se = np.sqrt((1.0 / J + n_test / n_train) * np.var(x, ddof=1))
    h = stats.t.ppf(0.5 + level / 2, df=J - 1) * se
    return x.mean(), h


# 4. One outer fold of the nested CV (used by the Part 2 notebook)
MODEL_ORDER_ALL = ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking"]
OOF_KEYS = ["Stacking (tuned)", "Stacking (default)", "LR (tuned)"]


def run_outer_fold(fold_id, tr, te, X, y, n_splits):
    """
    Default and tuned configurations on one outer fold.
    Tuning uses the outer training fold only; the outer test fold is used once, for scoring.
    Returns (metric records, best params, out-of-fold scores/predictions for OOF_KEYS).
    """
    rep, seed = fold_id // n_splits, fold_id
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    tuned, best = tune_base_models(Xtr, ytr, seed)
    records, oof = [], {}
    for config, pipes in [("Default", make_default_pipelines(seed)),
                          ("Tuned", make_tuned_pipelines(tuned, seed))]:
        for name in MODEL_ORDER_ALL:
            model = clone(pipes[name]).fit(Xtr, ytr)
            y_pred = model.predict(Xte)
            y_score = model.predict_proba(Xte)[:, 1]
            res = evaluate(yte, y_pred, y_score)
            res["Brier"] = brier_score_loss(yte, y_score)
            records.append({"fold": fold_id, "repeat": rep, "config": config, "model": name,
                            "n_train": len(tr), "n_test": len(te), **res})
            key = f"{name} ({config.lower()})"
            if key in OOF_KEYS:
                oof[key] = (np.asarray(te), y_score, y_pred)
    params = {"fold": fold_id, **{n: str(p) for n, p in best.items()}}
    return records, params, oof


def holm_adjust(pvalues):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
