
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42


@dataclass
class CVResult:
    y_true: np.ndarray
    y_proba: np.ndarray
    roc_auc: float
    pr_auc: float
    brier: float

# initialzie pipeline,IMPORTANT: missing features are replaced by median. maybe
# change to better heuristic in final
def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transformers = [("num", numeric_pipeline, numeric_features)]
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers)

# xgboost
def build_xgb_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )

# log reg
def build_logreg_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> CVResult:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    return CVResult(
        y_true=y.values,
        y_proba=y_proba,
        roc_auc=roc_auc_score(y, y_proba),
        pr_auc=average_precision_score(y, y_proba),
        brier=brier_score_loss(y, y_proba),
    )


def train_calibrated(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    calibrated.fit(X, y)
    return calibrated


def get_post_preprocess_feature_names(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    pre: ColumnTransformer = pipeline.named_steps["preprocess"]
    names: List[str] = list(numeric_features)
    if categorical_features:
        ohe = pre.named_transformers_["cat"]
        names.extend(ohe.get_feature_names_out(categorical_features).tolist())
    return names

# get feature activationcontribution by attenuating from shap algo
def compute_shap_from_fitted(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    pre: ColumnTransformer = fitted_pipeline.named_steps["preprocess"]
    clf: XGBClassifier = fitted_pipeline.named_steps["clf"]
    X_transformed = pre.transform(X)
    explainer = shap.TreeExplainer(clf)
    raw_shap = explainer.shap_values(X_transformed)

    post_names = get_post_preprocess_feature_names(
        fitted_pipeline, numeric_features, categorical_features
    )
    shap_post = pd.DataFrame(raw_shap, columns=post_names, index=X.index)

    per_feature = pd.DataFrame(index=X.index)
    for feat in numeric_features:
        per_feature[feat] = shap_post[feat]
    for feat in categorical_features:
        cols = [c for c in shap_post.columns if c.startswith(f"{feat}_")]
        per_feature[feat] = shap_post[cols].sum(axis=1)

    return per_feature, raw_shap


def metrics_dict(name: str, cv_result: CVResult) -> Dict[str, float | str]:
    return {
        "model": name,
        "n": len(cv_result.y_true),
        "n_positive": int(cv_result.y_true.sum()),
        "roc_auc": round(float(cv_result.roc_auc), 4),
        "pr_auc": round(float(cv_result.pr_auc), 4),
        "brier": round(float(cv_result.brier), 4),
    }
