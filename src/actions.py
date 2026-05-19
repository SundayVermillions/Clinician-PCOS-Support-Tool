"""Clinical action recommendations.

Two registries:

* :data:`ACTION_REGISTRY` — keyed by ``(feature, direction)`` tuples, lists
  concrete clinical actions a referring physician should take when this
  feature contributes meaningfully to a positive (or, occasionally,
  negative) prediction. Templates are hand-curated and grounded in the
  2023 International Evidence-Based Guideline for PCOS.

* :data:`MISSING_DATA_ACTIONS` — what to collect when a feature is absent
  from the patient record. Some of these are critical (e.g., 17-OHP for
  CAH exclusion); others are quality-of-care improvements.

* :data:`COMORBIDITY_RULES` — feature-based heuristics that flag long-term
  risks (T2DM, endometrial cancer, etc.) so the report includes a
  comorbidity-aware action plan even when the binary PCOS classification
  is settled.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Per-feature action registry
# -----------------------------------------------------------------------------

# Keys are (feature_name, direction). Direction "+" = positive contribution
# toward PCOS; "-" = negative. Most actions trigger only on "+".

ACTION_REGISTRY: Dict[Tuple[str, str], List[str]] = {
    ("AMH(ng/mL)", "+"): [
        "Confirm AMH elevation with a repeat measurement after a regular cycle.",
        "Order transvaginal ultrasound to corroborate polycystic ovarian morphology.",
        "Consider 17-OHP to rule out non-classical congenital adrenal hyperplasia.",
    ],
    ("Follicle No. (L)", "+"): [
        "Confirm imaging — document follicle count and ovarian volume.",
        "Apply revised Rotterdam criterion (≥20 antral follicles per ovary on high-resolution US) if available.",
    ],
    ("Follicle No. (R)", "+"): [
        "Confirm imaging — document follicle count and ovarian volume.",
        "Apply revised Rotterdam criterion (≥20 antral follicles per ovary on high-resolution US) if available.",
    ],
    ("hair growth(Y/N)", "+"): [
        "Document hirsutism with Ferriman-Gallwey scoring (threshold ≥6 in South Asian populations).",
        "Measure free testosterone and SHBG if clinical hirsutism is significant.",
    ],
    ("Skin darkening (Y/N)", "+"): [
        "Order fasting glucose / HbA1c — acanthosis nigricans is a visible insulin-resistance marker.",
        "Compute HOMA-IR if fasting insulin available.",
        "Counsel on lifestyle intervention (≥5% weight loss restores ovulation in many).",
    ],
    ("Weight gain(Y/N)", "+"): [
        "Quantify weight trajectory over 12 months; document interventions tried.",
        "Counsel on calorie-deficit + structured exercise; ≥150 min/week moderate activity.",
        "Consider metformin if BMI ≥25 and insulin resistance evident.",
    ],
    ("Hair loss(Y/N)", "+"): [
        "Examine scalp for female-pattern androgenic alopecia; differentiate from telogen effluvium.",
    ],
    ("Pimples(Y/N)", "+"): [
        "Adult-onset or treatment-resistant acne warrants androgen panel "
        "(total/free testosterone, DHEA-S, 17-OHP).",
    ],
    ("Cycle(R/I)", "+"): [
        "Document cycle history over ≥6 months; calculate average cycle length.",
        "If cycle >35 days persistently, meets Rotterdam oligo-anovulation criterion.",
        "Consider day-21 progesterone to confirm anovulation directly.",
    ],
    ("BMI", "+"): [
        "Counsel that South Asian populations face elevated metabolic risk at lower BMI thresholds.",
        "Order full metabolic panel: HbA1c, lipid panel, liver enzymes.",
    ],
    ("Waist:Hip Ratio", "+"): [
        "Document waist circumference. WHO Asia-Pacific threshold for women: >80 cm = increased risk.",
    ],
    ("TSH (mIU/L)", "+"): [
        "Investigate thyroid dysfunction before attributing menstrual irregularity to PCOS.",
        "Order free T4 and anti-TPO antibodies if TSH abnormal.",
    ],
    ("PRL(ng/mL)", "+"): [
        "Investigate hyperprolactinemia — pituitary MRI if prolactin persistently >50 ng/mL.",
        "Review medications: antipsychotics, antiemetics, opioids elevate prolactin.",
    ],
    ("FSH/LH", "+"): [
        "Confirm gonadotropins drawn in early follicular phase (cycle days 2–5) for valid interpretation.",
    ],
    ("RBS(mg/dl)", "+"): [
        "Replace random with fasting glucose or HbA1c — RBS is a coarse screen.",
        "OGTT if HbA1c 5.7–6.4% (prediabetes range).",
    ],
    ("Endometrium (mm)", "+"): [
        "If endometrium ≥15 mm in non-luteal phase, consider endometrial biopsy.",
        "Counsel on endometrial-cancer risk from chronic unopposed estrogen.",
        "Discuss cyclical progestin or combined OCP for endometrial protection.",
    ],
    ("BP _Systolic (mmHg)", "+"): [
        "Confirm with repeat measurements at separate visits.",
        "Initiate lifestyle counseling; pharmacotherapy if persistently ≥140/90.",
    ],
}


# -----------------------------------------------------------------------------
# Missing-data action templates
# -----------------------------------------------------------------------------

# Standard "additional info to collect" items, always shown.
ALWAYS_COLLECT: List[str] = [
    "Family history of PCOS, type 2 diabetes, and endometrial cancer.",
    "Hirsutism severity via Ferriman-Gallwey scoring (binary Y/N loses information).",
    "Cycle day at the time of hormone measurements (FSH, LH, AMH, progesterone).",
    "17-hydroxyprogesterone — required exclusion test for non-classical CAH.",
]

# Conditional items, shown only when the listed feature is missing or low quality.
MISSING_DATA_ACTIONS: Dict[str, List[str]] = {
    "AMH(ng/mL)": ["Measure AMH — strongest non-imaging PCOS biomarker."],
    "TSH (mIU/L)": ["Measure TSH — required to exclude thyroid dysfunction."],
    "PRL(ng/mL)": ["Measure prolactin — required to exclude hyperprolactinemia."],
    "RBS(mg/dl)": ["Order HbA1c or fasting glucose (preferred over random)."],
    "Endometrium (mm)": ["Measure endometrial thickness on ultrasound (timing-aware)."],
    "Follicle No. (L)": ["Transvaginal ultrasound for antral follicle count."],
    "Follicle No. (R)": ["Transvaginal ultrasound for antral follicle count."],
    "FSH/LH": ["Measure early-follicular FSH and LH (cycle days 2–5)."],
    "Vit D3 (ng/mL)": ["Measure 25-OH vitamin D — frequently low in PCOS; supplementation is benign."],
    "Hb(g/dl)": ["Measure hemoglobin — heavy bleeding can cause iron-deficiency anemia."],
}


# -----------------------------------------------------------------------------
# Comorbidity heuristics
# -----------------------------------------------------------------------------


def flag_metabolic_risk(patient: pd.Series) -> List[str]:
    """T2DM / metabolic-syndrome risk flags."""
    flags: List[str] = []
    bmi = patient.get("BMI")
    if bmi is not None and not pd.isna(bmi) and float(bmi) >= 27.5:
        flags.append("Obese by South Asian threshold — high T2DM risk; offer annual OGTT.")
    skin = patient.get("Skin darkening (Y/N)")
    if skin is not None and not pd.isna(skin) and int(skin) == 1:
        flags.append("Acanthosis nigricans present — clinical insulin resistance; metformin consideration.")
    rbs = patient.get("RBS(mg/dl)")
    if rbs is not None and not pd.isna(rbs) and float(rbs) >= 140:
        flags.append("Random glucose ≥140 mg/dL — confirm with HbA1c.")
    whr = patient.get("Waist:Hip Ratio")
    if whr is not None and not pd.isna(whr) and float(whr) >= 0.85:
        flags.append("Central adiposity pattern — independent cardiometabolic risk factor.")
    return flags


def flag_endometrial_cancer_risk(patient: pd.Series) -> List[str]:
    flags: List[str] = []
    cycle = patient.get("Cycle(R/I)")
    bmi = patient.get("BMI")
    endo = patient.get("Endometrium (mm)")
    if cycle is not None and not pd.isna(cycle) and int(cycle) == 0:
        flags.append(
            "Anovulatory cycles → unopposed estrogen exposure. "
            "Recommend cyclical progestin or combined OCP for endometrial protection."
        )
    if bmi is not None and not pd.isna(bmi) and float(bmi) >= 27.5 and cycle == 0:
        flags.append("Obesity + anovulation compounds endometrial cancer risk (peripheral aromatization).")
    if endo is not None and not pd.isna(endo) and float(endo) >= 15:
        flags.append("Endometrial thickness ≥15 mm — consider endometrial biopsy.")
    return flags


def flag_cardiovascular_risk(patient: pd.Series) -> List[str]:
    flags: List[str] = []
    sbp = patient.get("BP _Systolic (mmHg)")
    dbp = patient.get("BP _Diastolic (mmHg)")
    if sbp is not None and not pd.isna(sbp) and float(sbp) >= 130:
        flags.append(f"Systolic BP elevated ({float(sbp):.0f} mmHg).")
    if dbp is not None and not pd.isna(dbp) and float(dbp) >= 85:
        flags.append(f"Diastolic BP elevated ({float(dbp):.0f} mmHg).")
    return flags


def flag_mental_health(patient: pd.Series) -> List[str]:
    """Always recommend screening; PCOS patients have ~3× depression rates."""
    return [
        "Screen for depression and anxiety — PCOS patients have ~3× population rates.",
    ]


COMORBIDITY_RULES: Dict[str, Callable[[pd.Series], List[str]]] = {
    "Metabolic / T2DM risk": flag_metabolic_risk,
    "Endometrial cancer risk": flag_endometrial_cancer_risk,
    "Cardiovascular risk": flag_cardiovascular_risk,
    "Mental health": flag_mental_health,
}


def collect_comorbidity_flags(patient: pd.Series) -> Dict[str, List[str]]:
    """Run every comorbidity rule and return a dict of category → flags.
    Empty categories are omitted from the output."""
    out: Dict[str, List[str]] = {}
    for category, rule in COMORBIDITY_RULES.items():
        flags = rule(patient)
        if flags:
            out[category] = flags
    return out


# -----------------------------------------------------------------------------
# Public dispatchers
# -----------------------------------------------------------------------------


def actions_for_feature(feature: str, direction: str) -> List[str]:
    return ACTION_REGISTRY.get((feature, direction), [])


def actions_for_missing(features: List[str]) -> List[str]:
    """Return a deduplicated list of actions for the missing features."""
    seen = set()
    out: List[str] = []
    for f in features:
        for action in MISSING_DATA_ACTIONS.get(f, []):
            if action not in seen:
                seen.add(action)
                out.append(action)
    return out
