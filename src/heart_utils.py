"""
Shared utilities for the SIT307 11.1HD reproduction study of
Bhagat et al. (2025), "An efficient stacking-based ensemble technique for
early heart attack prediction", Multimedia Tools and Applications 84:36351-36375.

Used by both notebooks:
  - notebooks/part1_reproduction.ipynb  (faithful reproduction of the paper)
  - notebooks/part2_proposed.ipynb      (proposed leakage-free protocol)

Everything that is fitted (imputer, encoder, scaler, classifiers) lives inside
an sklearn Pipeline, so it is fitted on training data only.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)
from xgboost import XGBClassifier

# Column groups (names follow the Kaggle file; Table 9 of the paper)
TARGET = "target"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Nominal categories -> one-hot (no natural order between codes)
NOMINAL = ["cp", "restecg", "slope", "thal"]
# Continuous / count features -> min-max normalised
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Binary 0/1 features -> passed through unchanged
BINARY = ["sex", "fbs", "exang"]

# Codes outside the valid range stated in Table 9 of the paper.
# Table 9: ca in 0..3, thal in 1..3. The Kaggle file contains ca=4 and thal=0,
# which correspond to missing entries ("?") in the original UCI data.
INVALID_CODES = {"ca": 4, "thal": 0}
VALID_RANGE = {"ca": (0, 3), "thal": (1, 3)}

MODEL_ORDER = ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking"]


def load_data(path):
    """Load the Kaggle heart.csv file."""
    df = pd.read_csv(path)
    assert list(df.columns) == FEATURES + [TARGET], "Unexpected column layout"
    return df


# Preprocessing
class RegressionImputer(BaseEstimator, TransformerMixin):
    """
    Section 3.1 of the paper: 'A well-known basic linear regression imputation
    method is used to handle missing values ... [it] employs the correlation
    matrix to establish attribute value correlation and then predicts missing
    values using existing attribute values.'

    Implementation (assumption, the paper gives no code):
      1. Invalid codes (ca=4, thal=0) are marked as missing.
      2. Each missing value is predicted from the other attributes with a
         linear regression model (sklearn IterativeImputer + LinearRegression).
      3. Imputed categorical codes are rounded and clipped to the valid range.
    Fitted on the training split only.
    """

    def __init__(self, random_state=0):
        self.random_state = random_state

    def _mark(self, X):
        X = X.copy().astype(float)
        for col, bad in INVALID_CODES.items():
            X.loc[X[col] == bad, col] = np.nan
        return X

    def fit(self, X, y=None):
        self.imputer_ = IterativeImputer(estimator=LinearRegression(),
                                         max_iter=10, random_state=self.random_state)
        self.imputer_.fit(self._mark(X))
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        Xm = self._mark(X)
        out = pd.DataFrame(self.imputer_.transform(Xm), columns=self.columns_, index=X.index)
        for col, (lo, hi) in VALID_RANGE.items():
            out[col] = out[col].round().clip(lo, hi)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def make_preprocessor(random_state=0):
    """Regression imputation -> one-hot (nominal) + min-max (numeric) + passthrough (binary)."""
    encode = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
            ("numeric", MinMaxScaler(), NUMERIC),
            ("binary", "passthrough", BINARY),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("impute", RegressionImputer(random_state=random_state)),
                     ("encode", encode)])


# Models (paper Section 3.3 and 3.4.1). Hyperparameters are not reported in
# the paper, so library defaults are used unless the paper states otherwise.
def make_base_classifiers(random_state=0):
    return {
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        # Eq. (2) and Table 3 of the paper describe entropy-based splitting
        "DT": DecisionTreeClassifier(criterion="entropy", random_state=random_state),
        "RF": RandomForestClassifier(random_state=random_state, n_jobs=1),
        "XGB": XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=1),
        "NB": GaussianNB(),
        # Table 7: Euclidean distance; K not reported -> sklearn default K=5
        "KNN": KNeighborsClassifier(metric="euclidean"),
    }


def make_stacking(random_state=0, cv=5):
    """
    Section 3.4.1 / Table 8: all six classifiers are stacked and a
    meta-classifier is trained on their out-of-fold predictions
    ('5-fold stacking', Section 5). The meta-classifier is not named in the
    paper; Logistic Regression is assumed.
    """
    base = make_base_classifiers(random_state)
    return StackingClassifier(
        estimators=list(base.items()),
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=cv, stack_method="predict_proba", n_jobs=1,
    )


def make_all_pipelines(random_state=0):
    """preprocessor + classifier, one Pipeline per model (incl. stacking)."""
    models = make_base_classifiers(random_state)
    models["Stacking"] = make_stacking(random_state)
    return {name: Pipeline([("prep", make_preprocessor(random_state)), ("clf", clf)])
            for name, clf in models.items()}


# Evaluation
def evaluate(y_true, y_pred, y_score):
    """Metrics used in Table 11 of the paper. AUC uses predicted probabilities."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def fit_and_score(pipe, X_train, y_train, X_test, y_test):
    """Fit a (cloned) pipeline on train, return metrics, predictions and scores on test."""
    model = clone(pipe).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return model, evaluate(y_test, y_pred, y_score), y_pred, y_score


def metrics_from_confusion(tn, fp, fn, tp):
    """Recompute all metrics from a reported confusion matrix (consistency check)."""
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan
    return {"Accuracy": acc, "Precision": prec, "Recall": rec,
            "F1": f1, "Specificity": spec, "MCC": mcc}


def row_keys(df):
    """String key for each row, used to detect exact duplicates across splits."""
    return df.astype(str).agg("|".join, axis=1)


def overlap_rate(train_df, test_df):
    """Share of test rows that have an identical copy in the training rows."""
    train_keys = set(row_keys(train_df))
    return row_keys(test_df).isin(train_keys).mean()
