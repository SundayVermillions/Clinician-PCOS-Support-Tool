"""Clinical metric helpers.

Provides:
    * Subgroup definitions (age, BMI with South Asian thresholds, hyperandrogenism)
    * Rotterdam rule baseline
    * Per-subgroup metric computation
    * Threshold selection helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# -----------------------------------------------------------------------------
# Subgroup definitions
# -----------------------------------------------------------------------------

# South-Asian BMI thresholds (WHO Asia-Pacific, ICMR Indian guideline).
# Overweight at ≥23, obese at ≥27.5 — lower than the Western 25/30 cutoffs.
BMI_LEAN_CUT = 23.0
BMI_OBESE_CUT = 27.5

AGE_YOUNG_CUT = 25.0
AGE_OLDER_CUT = 35.0

HYPERANDROGENISM_SIGN_COLUMNS: List[str] = [
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
]


def assign_age_band(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    if age < AGE_YOUNG_CUT:
        return f"young (<{int(AGE_YOUNG_CUT)})"
    if age < AGE_OLDER_CUT:
        return f"mid ({int(AGE_YOUNG_CUT)}-{int(AGE_OLDER_CUT - 1)})"
    return f"older ({int(AGE_OLDER_CUT)}+)"


def assign_bmi_band(bmi: float) -> str:
    if pd.isna(bmi):
        return "unknown"
    if bmi < BMI_LEAN_CUT:
        return f"lean (<{BMI_LEAN_CUT})"
    if bmi < BMI_OBESE_CUT:
        return f"overweight ({BMI_LEAN_CUT}-{BMI_OBESE_CUT})"
    return f"obese (>={BMI_OBESE_CUT})"


def hyperandrogenism_score(df: pd.DataFrame) -> pd.Series:
    """Count of the four clinical hyperandrogenism signs reported as positive."""
    return df[HYPERANDROGENISM_SIGN_COLUMNS].sum(axis=1).astype(int)


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``age_band``, ``bmi_band``, and
    ``hyperandrogenism_clinical`` columns appended."""
    out = df.copy()
    out["age_band"] = out["Age (yrs)"].apply(assign_age_band)
    out["bmi_band"] = out["BMI"].apply(assign_bmi_band)
    out["hyperandrogenism_n_signs"] = hyperandrogenism_score(out)
    # Binary: at least one sign present.
    out["hyperandrogenism_clinical"] = (out["hyperandrogenism_n_signs"] >= 1).astype(int)
    return out


# -----------------------------------------------------------------------------
# Rotterdam rule baseline
# -----------------------------------------------------------------------------

# Original 2003 Rotterdam follicle threshold.
ROTTERDAM_FOLLICLE_THRESHOLD = 12


@dataclass
class RotterdamResult:
    """Vectorized Rotterdam predictions plus the three component criteria."""

    oligo_anovulation: pd.Series  # criterion 1
    hyperandrogenism: pd.Series  # criterion 2
    pcom: pd.Series  # criterion 3
    n_criteria_met: pd.Series  # sum of the three, 0..3
    prediction: pd.Series  # >=2 of 3 → 1


def rotterdam_rule(df: pd.DataFrame) -> RotterdamResult:
    """Apply the Rotterdam 2-of-3 rule.

    Simplifications relative to clinical practice (documented in report):
    - "Oligo/anovulation" uses ``Cycle(R/I) == 0``. The dataset's
      ``Cycle length(days)`` column appears to encode bleed duration
      rather than cycle-to-cycle length, so cannot be used as ">35 days".
    - "Hyperandrogenism" is any of the four clinical signs.
    - "PCOM" uses the original 2003 threshold of ≥12 follicles per ovary.
    - The rule does NOT exclude mimics (CAH, thyroid, hyperprolactinemia)
      because the dataset's exclusion labs are not reliably present.
    """
    oligo = (df["Cycle(R/I)"] == 0).astype(int)
    hyperandro = (df[HYPERANDROGENISM_SIGN_COLUMNS].sum(axis=1) >= 1).astype(int)
    pcom = (
        (df["Follicle No. (L)"] >= ROTTERDAM_FOLLICLE_THRESHOLD)
        | (df["Follicle No. (R)"] >= ROTTERDAM_FOLLICLE_THRESHOLD)
    ).astype(int)
    n = oligo + hyperandro + pcom
    pred = (n >= 2).astype(int)
    return RotterdamResult(
        oligo_anovulation=oligo,
        hyperandrogenism=hyperandro,
        pcom=pcom,
        n_criteria_met=n,
        prediction=pred,
    )


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Sensitivity, specificity, PPV, NPV, accuracy from a binary prediction."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / (tn + fp + fn + tp)
    return {
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "accuracy": acc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def probabilistic_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """ROC-AUC, PR-AUC, Brier — defined only when both classes present."""
    if y_true.min() == y_true.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def sensitivity_at_specificity(
    y_true: np.ndarray, y_proba: np.ndarray, target_specificity: float = 0.8
) -> Tuple[float, float]:
    """Find the threshold that gives at least ``target_specificity`` and
    report the corresponding sensitivity. Returns ``(sensitivity, threshold)``.
    """
    if y_true.min() == y_true.max():
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificities = 1 - fpr
    feasible = specificities >= target_specificity
    if not feasible.any():
        return float("nan"), float("nan")
    # Pick the threshold that maximizes sensitivity among those meeting the spec floor.
    idx = np.argmax(np.where(feasible, tpr, -np.inf))
    return float(tpr[idx]), float(thresholds[idx])


def threshold_at_sensitivity(
    y_true: np.ndarray, y_proba: np.ndarray, target_sensitivity: float = 0.9
) -> Tuple[float, float]:
    """Find the (highest) threshold achieving at least ``target_sensitivity``.
    Returns ``(specificity, threshold)``.
    """
    if y_true.min() == y_true.max():
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    feasible = tpr >= target_sensitivity
    if not feasible.any():
        return float("nan"), float("nan")
    # Among feasible, pick the highest threshold (= highest specificity).
    idx = np.argmax(np.where(feasible, thresholds, -np.inf))
    return float(1 - fpr[idx]), float(thresholds[idx])


def metrics_for_subgroup(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    rotterdam_pred: np.ndarray,
    name: str,
    threshold_screen: float = 0.30,
    threshold_confirm: float = 0.70,
) -> Dict[str, float | str | int]:
    """Build one row of the per-subgroup summary table."""
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    prob = probabilistic_metrics(y_true, y_proba)
    # Model at the two clinical thresholds
    screen = binary_metrics(y_true, (y_proba >= threshold_screen).astype(int))
    confirm = binary_metrics(y_true, (y_proba >= threshold_confirm).astype(int))
    # Rotterdam baseline
    rott = binary_metrics(y_true, rotterdam_pred)
    sens_at_spec80, _ = sensitivity_at_specificity(y_true, y_proba, 0.8)
    return {
        "subgroup": name,
        "n": n,
        "n_positive": n_pos,
        "prevalence": round(n_pos / n, 3) if n else float("nan"),
        "roc_auc": round(prob["roc_auc"], 3),
        "pr_auc": round(prob["pr_auc"], 3),
        "brier": round(prob["brier"], 3),
        "sens@spec80": round(sens_at_spec80, 3),
        "model_screen_sens": round(screen["sensitivity"], 3),
        "model_screen_spec": round(screen["specificity"], 3),
        "model_confirm_sens": round(confirm["sensitivity"], 3),
        "model_confirm_spec": round(confirm["specificity"], 3),
        "rotterdam_sens": round(rott["sensitivity"], 3),
        "rotterdam_spec": round(rott["specificity"], 3),
    }
