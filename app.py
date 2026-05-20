"""PCOS Clinical Decision Support — web application.

Serves a clinician-facing interface for retrieving a patient record and
viewing the PCOS risk assessment: estimated probability, risk tier,
contributing factors, Rotterdam criteria, comorbidity considerations, and
recommended clinical actions.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from src.clinical_metrics import rotterdam_rule  # noqa: E402
from src.decision_support import generate_report_oof  # noqa: E402

app = Flask(__name__)

# load
_DB = pd.read_parquet(APP_DIR / "data/patients.parquet")
_PROBA = pd.read_parquet(APP_DIR / "data/cv_probabilities.parquet")
_SHAP = pd.read_parquet(APP_DIR / "data/shap_oof.parquet")
_ROTTERDAM = rotterdam_rule(_DB)
_PATIENT_COUNT = len(_DB)
_FEATURE_COUNT = _SHAP.shape[1]

# dict of const labels
FEATURE_DISPLAY_NAMES = {
    "Age (yrs)": "Age",
    "Weight (Kg)": "Weight",
    "Height(Cm)": "Height",
    "BMI": "Body mass index (BMI)",
    "Blood Group": "Blood group",
    "Pulse rate(bpm)": "Pulse rate",
    "RR (breaths/min)": "Respiratory rate",
    "Hb(g/dl)": "Haemoglobin",
    "Cycle(R/I)": "Menstrual cycle regularity",
    "Cycle length(days)": "Menstrual cycle length",
    "Marraige Status (Yrs)": "Years married",
    "Pregnant(Y/N)": "Currently pregnant",
    "No. of abortions": "Previous pregnancy losses",
    "I beta-HCG(mIU/mL)": "Beta-hCG (first measurement)",
    "II beta-HCG(mIU/mL)": "Beta-hCG (second measurement)",
    "FSH(mIU/mL)": "Follicle-stimulating hormone (FSH)",
    "LH(mIU/mL)": "Luteinising hormone (LH)",
    "FSH/LH": "FSH to LH ratio",
    "Hip(inch)": "Hip circumference",
    "Waist(inch)": "Waist circumference",
    "Waist:Hip Ratio": "Waist-to-hip ratio",
    "TSH (mIU/L)": "Thyroid-stimulating hormone (TSH)",
    "AMH(ng/mL)": "Anti-Mullerian hormone (AMH)",
    "PRL(ng/mL)": "Prolactin",
    "Vit D3 (ng/mL)": "Vitamin D",
    "PRG(ng/mL)": "Progesterone",
    "RBS(mg/dl)": "Random blood glucose",
    "Weight gain(Y/N)": "Recent weight gain",
    "hair growth(Y/N)": "Hirsutism (excess hair growth)",
    "Skin darkening (Y/N)": "Acanthosis nigricans (skin darkening)",
    "Hair loss(Y/N)": "Androgenic hair loss",
    "Pimples(Y/N)": "Acne",
    "Fast food (Y/N)": "Frequent fast food consumption",
    "Reg.Exercise(Y/N)": "Regular exercise",
    "BP _Systolic (mmHg)": "Systolic blood pressure",
    "BP _Diastolic (mmHg)": "Diastolic blood pressure",
    "Follicle No. (L)": "Antral follicle count (left ovary)",
    "Follicle No. (R)": "Antral follicle count (right ovary)",
    "Avg. F size (L) (mm)": "Mean follicle size (left ovary)",
    "Avg. F size (R) (mm)": "Mean follicle size (right ovary)",
    "Endometrium (mm)": "Endometrial thickness",
}

TIER_META = {
    "HIGH": {"label": "High risk", "css": "high"},
    "MEDIUM": {"label": "Moderate risk", "css": "moderate"},
    "LOW": {"label": "Low risk", "css": "low"},
}


def _display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def _format_value(feature: str, raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Not recorded"
    if "(Y/N)" in feature or feature == "Pregnant(Y/N)":
        return "Yes" if int(raw) == 1 else "No"
    if feature == "Cycle(R/I)":
        return "Regular" if int(raw) == 1 else "Irregular"
    if isinstance(raw, float):
        return f"{raw:g}"
    return str(raw)


def _build_context(pid: int) -> dict:
    patient = _DB.iloc[pid]
    report = generate_report_oof(
        patient,
        float(_PROBA["all"].iloc[pid]),
        _SHAP.iloc[pid],
        top_n_shap=5,
    )
    tier_meta = TIER_META[report["tier"]]

    factors = []
    for d in report["driving_features"]:
        factors.append(
            {
                "name": _display_name(d["feature"]),
                "value": _format_value(d["feature"], patient.get(d["feature"])),
                "increases": d["contribution"] > 0,
                "explanation": d["explanation"],
                "actions": d["actions"],
                "imputed": d["imputed"],
            }
        )

    rotterdam_criteria = [
        ("Oligo- or anovulation", int(_ROTTERDAM.oligo_anovulation.iloc[pid])),
        ("Clinical hyperandrogenism", int(_ROTTERDAM.hyperandrogenism.iloc[pid])),
        ("Polycystic ovarian morphology", int(_ROTTERDAM.pcom.iloc[pid])),
    ]

    age = patient.get("Age (yrs)")
    bmi = patient.get("BMI")
    cycle = patient.get("Cycle(R/I)")

    return {
        "found": True,
        "nric": f"{pid:04d}",
        "age": f"{int(age)} years" if pd.notna(age) else "Not recorded",
        "bmi": f"{bmi:.1f}" if pd.notna(bmi) else "Not recorded",
        "cycle": (
            ("Regular" if int(cycle) == 1 else "Irregular")
            if pd.notna(cycle)
            else "Not recorded"
        ),
        "completeness": f"{_FEATURE_COUNT - report['n_missing']} of {_FEATURE_COUNT} fields",
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "probability_pct": round(report["probability"] * 100),
        "tier_label": tier_meta["label"],
        "tier_css": tier_meta["css"],
        "headline": report["headline"],
        "n_missing": report["n_missing"],
        "factors": factors,
        "comorbidities": report["comorbidities"],
        "recommended_actions": report["recommended_actions"],
        "additional_info": report["additional_info_to_collect"],
        "rotterdam_criteria": rotterdam_criteria,
        "rotterdam_count": sum(v for _, v in rotterdam_criteria),
    }


def _render_error(query: str, message: str, status: int = 200):
    return (
        render_template(
            "index.html",
            searched=True,
            found=False,
            query=query,
            message=message,
        ),
        status,
    )


@app.route("/")
def index():
    raw = request.args.get("nric", "").strip()
    if not raw:
        return render_template("index.html", searched=False, found=False)
    try:
        pid = int(raw)
    except ValueError:
        return _render_error(raw, "The patient NRIC must be a number.")
    if not 0 <= pid < _PATIENT_COUNT:
        return _render_error(raw, "No patient record was found for this NRIC.")
    try:
        context = _build_context(pid)
    except Exception:
        return _render_error(
            raw,
            "The record for this NRIC could not be retrieved. Please try again.",
            status=500,
        )
    return render_template("index.html", searched=True, query=raw, **context)


@app.errorhandler(404)
def _not_found(_e):
    return redirect(url_for("index"))


@app.errorhandler(Exception)
def _unexpected(e):
    if isinstance(e, HTTPException):
        return e
    return _render_error(
        request.args.get("nric", "").strip(),
        "An unexpected error occurred. Please try again.",
        status=500,
    )


def _find_free_port(preferred: int = 8000) -> int:
    import socket

    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def _open_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


if __name__ == "__main__":
    import logging
    import threading

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    port = _find_free_port(8000)
    url = f"http://127.0.0.1:{port}"

    threading.Timer(1.5, _open_browser, args=(url,)).start()

    print()
    print("  PCOS Clinical Decision Support")
    print(f"  Running at {url}")
    print("  A browser window will open shortly.")
    print("  Keep this window open while using the tool.")
    print("  Close it to stop the application.")
    print()

    app.run(host="127.0.0.1", port=port)
