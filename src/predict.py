from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Tier thresholds — operating points derived from subgroup calibration.
TIER_SCREEN = 0.30  # below this: LOW
TIER_CONFIRM = 0.70  # above this: HIGH; otherwise MEDIUM


def load_bundle(model_name: str = "all") -> dict:
    """Load a saved model bundle.

    ``model_name`` is either ``"all"`` or ``"primary_care"``.
    """
    path = REPO_ROOT / f"models/classifier_{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundle not found at {path}. Run notebooks/01_tabular_classifier.py first."
        )
    return joblib.load(path)


def probability_to_tier(p: float) -> str:
    if p >= TIER_CONFIRM:
        return "HIGH"
    if p >= TIER_SCREEN:
        return "MEDIUM"
    return "LOW"


def _aggregate_shap_to_original_features(
    shap_post: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.Series:
    """Sum one-hot-expanded SHAP back to original feature names."""
    agg: Dict[str, float] = {}
    for feat in numeric_features:
        agg[feat] = float(shap_post[feat])
    for feat in categorical_features:
        cols = [c for c in shap_post.index if c.startswith(f"{feat}_")]
        agg[feat] = float(shap_post[cols].sum())
    return pd.Series(agg)


def predict_patient(
    patient: pd.Series | Dict,
    bundle: dict,
    top_n_shap: int = 5,
) -> dict:
    """Run the full inference pipeline for one patient.

    Returns:
        dict with keys:
          - ``probability``: calibrated PCOS probability (float)
          - ``tier``: ``"LOW" | "MEDIUM" | "HIGH"``
          - ``top_contributors``: list of dicts with feature, value, contribution
          - ``missing_features``: list of feature names that were imputed
    """
    if isinstance(patient, dict):
        patient = pd.Series(patient)

    features: List[str] = bundle["features"]
    numeric: List[str] = bundle["numeric_features"]
    categorical: List[str] = bundle["categorical_features"]

    # Format as a single-row DataFrame in the model's expected column order.
    # Use np.nan (not pd.NA) so SimpleImputer can coerce numeric columns cleanly.
    row = pd.DataFrame(
        [{f: patient.get(f, np.nan) for f in features}], columns=features
    )
    # Coerce numeric columns to float so np.nan flows through downstream
    # transforms; categorical features stay as-is (handle_unknown='ignore').
    for f in numeric:
        row[f] = pd.to_numeric(row[f], errors="coerce")

    missing = [f for f in features if pd.isna(row.iloc[0][f])]

    # Calibrated probability (the headline output).
    proba = float(bundle["model"].predict_proba(row)[0, 1])

    # SHAP attribution from the raw XGBoost included in the bundle.
    xgb_pipeline = bundle["xgb_raw"]
    pre = xgb_pipeline.named_steps["preprocess"]
    clf = xgb_pipeline.named_steps["clf"]
    row_transformed = pre.transform(row)

    explainer = shap.TreeExplainer(clf)
    raw_shap = explainer.shap_values(row_transformed)[0]

    post_names = list(numeric)
    if categorical:
        ohe = pre.named_transformers_["cat"]
        post_names.extend(ohe.get_feature_names_out(categorical).tolist())
    shap_series = pd.Series(raw_shap, index=post_names)
    shap_per_feature = _aggregate_shap_to_original_features(
        shap_series, numeric, categorical
    )

    # Top contributors by absolute SHAP value.
    top = shap_per_feature.reindex(shap_per_feature.abs().sort_values(ascending=False).index).head(top_n_shap)
    contributors = [
        {
            "feature": feat,
            "value": _format_value(row.iloc[0][feat]),
            "contribution": float(top[feat]),
            "imputed": feat in missing,
        }
        for feat in top.index
    ]

    return {
        "probability": proba,
        "tier": probability_to_tier(proba),
        "top_contributors": contributors,
        "missing_features": missing,
    }


def _format_value(v) -> str:
    if pd.isna(v):
        return "MISSING"
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) >= 0.01 else f"{v:.3g}"
    return str(v)


def format_report(report: dict, patient_label: str = "Patient") -> str:
    """Render a single-patient prediction as a human-readable text report."""
    p = report["probability"]
    tier_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[report["tier"]]
    out = [
        "=" * 64,
        f"  {patient_label}",
        "=" * 64,
        f"  PCOS probability:  {p:.1%}",
        f"  Risk tier:         {tier_color} {report['tier']}",
        "",
        "  Top contributing features:",
    ]
    for c in report["top_contributors"]:
        sign = "↑" if c["contribution"] > 0 else "↓"
        imputed = "  [imputed]" if c["imputed"] else ""
        out.append(
            f"    {sign} {c['feature']:<28} = {c['value']:>10}"
            f"   (Δlogit {c['contribution']:+.3f}){imputed}"
        )
    if report["missing_features"]:
        out.append("")
        out.append(f"  Missing inputs (median-imputed): {len(report['missing_features'])}")
        for f in report["missing_features"][:5]:
            out.append(f"    - {f}")
        if len(report["missing_features"]) > 5:
            out.append(f"    ... and {len(report['missing_features']) - 5} more")
    out.append("")
    return "\n".join(out)


def predict_dataframe(
    df: pd.DataFrame, bundle: dict, top_n_shap: int = 5
) -> Tuple[pd.DataFrame, List[dict]]:
    """Run inference for every row in ``df``.

    Returns a tuple of (summary DataFrame, list of per-patient report dicts).
    """
    reports = [predict_patient(row, bundle, top_n_shap) for _, row in df.iterrows()]
    summary = pd.DataFrame(
        {
            "probability": [r["probability"] for r in reports],
            "tier": [r["tier"] for r in reports],
            "n_missing": [len(r["missing_features"]) for r in reports],
        }
    )
    return summary, reports


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "primary_care"],
        default="all",
        help="Which trained model to use (default: all features).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--row",
        type=int,
        help="Predict on a row index from data/processed/df_clean.parquet.",
    )
    src.add_argument(
        "--csv",
        type=Path,
        help="Predict on every row of a CSV with the xlsx-style column names.",
    )
    src.add_argument(
        "--json",
        type=str,
        help='Predict on a single patient described inline as JSON, e.g. \'{"Age (yrs)": 28, ...}\'',
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many SHAP contributors to display per patient (default 5).",
    )
    args = parser.parse_args()

    bundle = load_bundle(args.model)

    if args.row is not None:
        df = pd.read_parquet(REPO_ROOT / "data/processed/df_clean.parquet")
        if args.row >= len(df):
            raise SystemExit(f"Row {args.row} out of range (n={len(df)}).")
        patient = df.iloc[args.row]
        truth = patient.get("PCOS (Y/N)")
        report = predict_patient(patient, bundle, top_n_shap=args.top_n)
        label = f"Training row {args.row}"
        if truth is not None and not pd.isna(truth):
            label += f"  (actual PCOS = {int(truth)})"
        print(format_report(report, label))
    elif args.csv is not None:
        df = pd.read_csv(args.csv)
        df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
        summary, reports = predict_dataframe(df, bundle, top_n_shap=args.top_n)
        for i, report in enumerate(reports):
            print(format_report(report, f"Patient {i} from {args.csv.name}"))
        print("Summary:")
        print(summary.to_string())
    elif args.json is not None:
        data = json.loads(args.json)
        report = predict_patient(data, bundle, top_n_shap=args.top_n)
        print(format_report(report, "Inline JSON patient"))


if __name__ == "__main__":
    _cli()
