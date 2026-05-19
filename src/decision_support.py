"""Patient report generation.

This module wires the calibrated classifier, the SHAP attributions, the
clinical thresholds, the per-feature explanation registry, the action
registry, and the comorbidity rules into one structured patient report.

Usage::

    from src.decision_support import generate_report, format_report_markdown
    from src.predict import load_bundle

    bundle = load_bundle("all")          # or "primary_care"
    report = generate_report(patient_dict, bundle)
    print(format_report_markdown(report, patient_id="PT-0042"))

The output is a deterministic, template-driven document. Every line traces
back to a registered template; there is no runtime text generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.actions import (  # noqa: E402
    ALWAYS_COLLECT,
    actions_for_feature,
    actions_for_missing,
    collect_comorbidity_flags,
)
from src.explanations import explain_feature  # noqa: E402
from src.predict import predict_patient, probability_to_tier  # noqa: E402


# Operating thresholds for tier mapping. The model is highly discriminative
# (see results/tables/operating_thresholds.csv — p≥0.179 already gives 90%
# sensitivity and 85% specificity), but we use round-number defaults here
# for UX clarity. Override per call to use the data-derived thresholds.
DEFAULT_SCREEN_THRESHOLD = 0.30  # below → LOW
DEFAULT_CONFIRM_THRESHOLD = 0.70  # above → HIGH; in-between → MEDIUM


def _series(patient) -> pd.Series:
    if isinstance(patient, pd.Series):
        return patient
    return pd.Series(patient)


def generate_report(
    patient,
    bundle: dict,
    top_n_shap: int = 5,
    screen_threshold: float = DEFAULT_SCREEN_THRESHOLD,
    confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD,
) -> Dict:
    """Build a structured decision-support report for a single patient.

    Returns a dict suitable for either ``format_report_markdown`` (for
    human consumption) or downstream UI rendering.
    """
    patient_series = _series(patient)

    # Run the classifier + SHAP via the existing inference function.
    pred = predict_patient(patient_series, bundle, top_n_shap=top_n_shap)

    # Re-derive tier with the (possibly overridden) thresholds.
    if pred["probability"] >= confirm_threshold:
        tier = "HIGH"
    elif pred["probability"] >= screen_threshold:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Build the driving-features section: explanation + actions per top SHAP contributor.
    driving = []
    for contributor in pred["top_contributors"]:
        feature = contributor["feature"]
        direction = "+" if contributor["contribution"] > 0 else "-"
        raw_value = patient_series.get(feature)
        explanation = explain_feature(feature, raw_value, direction)
        actions = actions_for_feature(feature, direction)
        driving.append(
            {
                "feature": feature,
                "value": contributor["value"],
                "contribution": contributor["contribution"],
                "direction": direction,
                "imputed": contributor["imputed"],
                "explanation": explanation,
                "actions": actions,
            }
        )

    # Comorbidity flags
    comorbidities = collect_comorbidity_flags(patient_series)

    # Recommended actions = union of per-feature actions + missing-data
    # actions + always-collect items. Dedupe in order.
    rec_actions: List[str] = []
    seen = set()
    for d in driving:
        for a in d["actions"]:
            if a not in seen:
                seen.add(a)
                rec_actions.append(a)
    for a in actions_for_missing(pred["missing_features"]):
        if a not in seen:
            seen.add(a)
            rec_actions.append(a)

    additional_info = [a for a in ALWAYS_COLLECT if a not in seen]

    # Headline recommendation by tier
    if tier == "HIGH":
        headline = "REFER for specialist evaluation. Confirmatory workup indicated."
    elif tier == "MEDIUM":
        headline = "Collect additional data; re-evaluate before referral decision."
    else:
        headline = "Low risk by current data. Routine follow-up; address modifiable risks as relevant."

    return {
        "probability": pred["probability"],
        "tier": tier,
        "headline": headline,
        "thresholds": {"screen": screen_threshold, "confirm": confirm_threshold},
        "driving_features": driving,
        "comorbidities": comorbidities,
        "recommended_actions": rec_actions,
        "additional_info_to_collect": additional_info,
        "missing_features": pred["missing_features"],
        "n_missing": len(pred["missing_features"]),
    }



def generate_report_oof(
    patient_row: pd.Series,
    oof_proba: float,
    oof_shap_row: pd.Series,
    top_n_shap: int = 5,
    screen_threshold: float = DEFAULT_SCREEN_THRESHOLD,
    confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD,
) -> Dict:
    """Like :func:`generate_report`, but uses precomputed out-of-fold
    probability and SHAP values from disk.

    Use for demoing training-set patients: the OOF probability and SHAP
    come from a model that did NOT see this patient in training, so the
    output is honest (no resubstitution optimism).
    """
    if oof_proba >= confirm_threshold:
        tier = "HIGH"
    elif oof_proba >= screen_threshold:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Identify imputed values: anything missing in the raw row.
    missing = [
        f for f in oof_shap_row.index
        if f in patient_row.index and pd.isna(patient_row[f])
    ]

    # Top SHAP contributors by absolute value.
    top = oof_shap_row.abs().sort_values(ascending=False).head(top_n_shap)
    driving = []
    for feature in top.index:
        contribution = float(oof_shap_row[feature])
        direction = "+" if contribution > 0 else "-"
        raw_value = patient_row.get(feature)
        value_display = "MISSING" if pd.isna(raw_value) else (
            f"{raw_value:.2f}" if isinstance(raw_value, float) else str(raw_value)
        )
        explanation = explain_feature(feature, raw_value, direction)
        actions = actions_for_feature(feature, direction)
        driving.append(
            {
                "feature": feature,
                "value": value_display,
                "contribution": contribution,
                "direction": direction,
                "imputed": pd.isna(raw_value),
                "explanation": explanation,
                "actions": actions,
            }
        )

    comorbidities = collect_comorbidity_flags(patient_row)

    rec_actions: List[str] = []
    seen = set()
    for d in driving:
        for a in d["actions"]:
            if a not in seen:
                seen.add(a)
                rec_actions.append(a)
    for a in actions_for_missing(missing):
        if a not in seen:
            seen.add(a)
            rec_actions.append(a)

    additional_info = [a for a in ALWAYS_COLLECT if a not in seen]

    if tier == "HIGH":
        headline = "REFER for specialist evaluation. Confirmatory workup indicated."
    elif tier == "MEDIUM":
        headline = "Collect additional data; re-evaluate before referral decision."
    else:
        headline = "Low risk by current data. Routine follow-up; address modifiable risks as relevant."

    return {
        "probability": oof_proba,
        "tier": tier,
        "headline": headline,
        "thresholds": {"screen": screen_threshold, "confirm": confirm_threshold},
        "driving_features": driving,
        "comorbidities": comorbidities,
        "recommended_actions": rec_actions,
        "additional_info_to_collect": additional_info,
        "missing_features": missing,
        "n_missing": len(missing),
        "_source": "out-of-fold",
    }


# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------


def _tier_badge(tier: str) -> str:
    return {"LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM", "HIGH": "🔴 HIGH"}[tier]


def _data_quality_caveat(n_missing: int) -> str:
    if n_missing <= 3:
        return ""
    if n_missing <= 10:
        return (
            f"> **Data caveat.** {n_missing} features were not recorded and "
            f"were median-imputed. Probability is usable but list the "
            f"missing items under additional info before relying on it."
        )
    return (
        f"> **⚠️ DATA WARNING — INTERPRET WITH CAUTION.** {n_missing} of the "
        f"model's inputs were missing and median-imputed. The probability "
        f"largely reflects assumed defaults rather than this patient's "
        f"specific physiology. Collect the listed data before acting."
    )


def format_report_markdown(report: Dict, patient_id: str = "Patient") -> str:
    """Render a generated report as a clinician-facing markdown document."""
    lines: List[str] = []
    lines.append(f"# PCOS Decision Support Report — {patient_id}")
    lines.append("")
    lines.append("## Risk assessment")
    lines.append("")
    lines.append(f"- **PCOS likelihood:** {_tier_badge(report['tier'])}")
    lines.append(f"- **Calibrated probability:** {report['probability']:.1%}")
    lines.append(
        f"- **Operating thresholds:** screen ≥ {report['thresholds']['screen']:.2f}, "
        f"confirm ≥ {report['thresholds']['confirm']:.2f}"
    )
    lines.append(f"- **Recommendation:** {report['headline']}")

    caveat = _data_quality_caveat(report["n_missing"])
    if caveat:
        lines.append("")
        lines.append(caveat)

    # Driving features
    lines.append("")
    lines.append("## Driving features")
    lines.append("")
    for d in report["driving_features"]:
        sign = "↑" if d["contribution"] > 0 else "↓"
        imputed = "  *(value imputed)*" if d["imputed"] else ""
        lines.append(
            f"### {sign} `{d['feature']}` = {d['value']}{imputed}"
        )
        lines.append(f"*Contribution to log-odds: {d['contribution']:+.3f}*")
        lines.append("")
        if d["explanation"]:
            lines.append(d["explanation"])
        if d["actions"]:
            lines.append("")
            lines.append("**Suggested actions:**")
            for a in d["actions"]:
                lines.append(f"- {a}")
        lines.append("")

    # Comorbidity flags
    if report["comorbidities"]:
        lines.append("## Comorbidity-aware risk flags")
        lines.append("")
        for category, flags in report["comorbidities"].items():
            lines.append(f"### {category}")
            for f in flags:
                lines.append(f"- {f}")
            lines.append("")

    # Aggregated recommended actions
    if report["recommended_actions"]:
        lines.append("## Recommended actions (summary)")
        lines.append("")
        for a in report["recommended_actions"]:
            lines.append(f"- {a}")
        lines.append("")

    # Additional info to collect (always shown)
    lines.append("## Additional information to collect")
    lines.append("")
    for info in report["additional_info_to_collect"]:
        lines.append(f"- {info}")
    if report["missing_features"]:
        lines.append("")
        lines.append(
            f"**Specifically missing for this patient ({len(report['missing_features'])}):** "
            + ", ".join(report["missing_features"][:10])
            + (", …" if len(report["missing_features"]) > 10 else "")
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report is a clinical decision support output, not a diagnosis. "
        "All recommendations remain at the discretion of the treating clinician. "
        "The model has not been externally validated.*"
    )
    return "\n".join(lines)
