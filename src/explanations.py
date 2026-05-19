"""Per-feature clinical explanation templates.

Each ``explain_*`` function takes the patient's raw value plus the direction
of the SHAP contribution (``"+"`` = pushed toward PCOS, ``"-"`` = pushed
away), and returns a 1-2 sentence clinical explanation in plain English.

The :func:`explain_feature` dispatcher returns either a specific explanation
when one is registered, or a generic fallback that just states the value
and direction.

These explanations are hand-curated and grounded in the clinical
literature. They are fixed templates, not generated at runtime.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd



def explain_amh(value, direction: str) -> str:
    if pd.isna(value):
        return "AMH not measured. Recommend measurement — AMH is the single strongest non-imaging PCOS biomarker."
    v = float(value)
    if direction == "+":
        return (
            f"AMH is elevated at {v:.1f} ng/mL. AMH is produced by granulosa cells "
            f"of small antral follicles; the elevation reflects the expanded antral "
            f"follicle population characteristic of PCOS (typical PCOS cutoff ≥4 ng/mL)."
        )
    return (
        f"AMH is {v:.1f} ng/mL (within or below the typical PCOS range). "
        f"Argues mildly against PCOS but does not exclude it — phenotype-D patients "
        f"can have normal AMH."
    )


def explain_follicle_count(value, direction: str, side: str = "ovary") -> str:
    if pd.isna(value):
        return f"Antral follicle count for the {side} not recorded. Confirm with transvaginal ultrasound."
    v = int(value)
    if direction == "+":
        return (
            f"{v} antral follicles in the {side} — meets the original Rotterdam threshold (≥12) "
            f"and approaches the 2018 revised threshold (≥20). Direct evidence of polycystic ovarian morphology."
        )
    return (
        f"{v} antral follicles in the {side}, below the Rotterdam threshold. "
        f"Argues against polycystic ovarian morphology."
    )


def explain_follicle_l(value, direction: str) -> str:
    return explain_follicle_count(value, direction, "left ovary")


def explain_follicle_r(value, direction: str) -> str:
    return explain_follicle_count(value, direction, "right ovary")


def explain_cycle_regularity(value, direction: str) -> str:
    if pd.isna(value):
        return "Cycle regularity not recorded. Ask the patient: is the menstrual cycle regular (within ±3 days each month)?"
    if int(value) == 0:
        return (
            "Cycle reported as irregular. Meets the Rotterdam oligo-/anovulation criterion. "
            "Anovulatory cycles are the mechanistic basis for unopposed estrogen exposure "
            "and downstream endometrial-cancer risk."
        )
    return (
        "Cycle reported as regular. Argues against the oligo-/anovulation criterion — "
        "consider Rotterdam phenotype C (ovulatory PCOS) if other criteria are met."
    )


def explain_hair_growth(value, direction: str) -> str:
    if pd.isna(value):
        return (
            "Hirsutism not recorded. Recommend Ferriman-Gallwey scoring "
            "(threshold ≥6 in South Asian populations; ≥8 in most others). "
            "Binary Y/N loses important severity information."
        )
    if int(value) == 1:
        return (
            "Hirsutism reported (excess male-pattern body/facial hair). The most specific clinical "
            "hyperandrogenism sign; theca-cell androgen excess via CYP17A1/SRD5A1 pathway. "
            "Confirm severity with Ferriman-Gallwey scoring."
        )
    return "No hirsutism reported. Argues against clinical hyperandrogenism by this axis."


def explain_skin_darkening(value, direction: str) -> str:
    if pd.isna(value):
        return "Acanthosis nigricans not assessed. Check neck, axillae, and groin for dark velvety patches — visible marker of insulin resistance."
    if int(value) == 1:
        return (
            "Acanthosis nigricans present (dark, velvety skin patches). A visible marker of "
            "insulin resistance — frequently co-occurs with PCOS and predicts metabolic risk. "
            "Recommend HbA1c / OGTT screening."
        )
    return "No skin darkening reported. Less likely to have severe insulin resistance, though does not exclude it."


def explain_weight_gain(value, direction: str) -> str:
    if pd.isna(value):
        return "Weight history not recorded. Ask about unexplained or rapid weight gain in the past 12 months."
    if int(value) == 1:
        return (
            "Recent weight gain reported. Common PCOS manifestation, driven by insulin resistance and "
            "androgen-promoted central adiposity. Lifestyle intervention (5–10% weight loss) often "
            "restores ovulation."
        )
    return "No recent weight gain reported."


def explain_hair_loss(value, direction: str) -> str:
    if pd.isna(value):
        return "Scalp hair pattern not recorded. Assess for female-pattern (androgenic) alopecia."
    if int(value) == 1:
        return (
            "Female-pattern hair loss reported — androgenic alopecia is one of the four "
            "clinical hyperandrogenism signs. Less specific than hirsutism."
        )
    return "No hair loss reported."


def explain_pimples(value, direction: str) -> str:
    if pd.isna(value):
        return "Acne status not recorded."
    if int(value) == 1:
        return (
            "Persistent acne reported. Androgen-driven, especially in adult-onset cases. "
            "Note: acne is the *least* specific hyperandrogenism sign — common in the general population."
        )
    return "No persistent acne reported."


def explain_bmi(value, direction: str) -> str:
    if pd.isna(value):
        return "BMI not recorded."
    v = float(value)
    # South Asian thresholds
    if v >= 27.5:
        band = "obese (South Asian threshold)"
    elif v >= 23.0:
        band = "overweight (South Asian threshold)"
    else:
        band = "lean"
    return (
        f"BMI {v:.1f} — {band}. Note: South Asian populations face elevated metabolic risk "
        f"at lower BMI than Western thresholds. Lean PCOS is real and should not be discounted."
    )


def explain_waist_hip(value, direction: str) -> str:
    if pd.isna(value):
        return "Waist:hip ratio not recorded. Better metabolic-risk indicator than BMI alone."
    v = float(value)
    if v >= 0.85:
        return (
            f"Waist:hip ratio {v:.2f} — central adiposity pattern (≥0.85 in women is concerning). "
            f"Correlates with insulin resistance and cardiometabolic risk independent of BMI."
        )
    return f"Waist:hip ratio {v:.2f} — within normal range."


def explain_fsh_lh(value, direction: str) -> str:
    if pd.isna(value):
        return "FSH/LH ratio not computed (need both hormones in early follicular phase)."
    v = float(value)
    if v < 1.0:
        return (
            f"FSH/LH ratio {v:.2f} (<1) — LH exceeds FSH, the classic PCOS gonadotropin pattern "
            f"present in ~60% of cases. Tonic high estrogen disrupts the normal feedback that suppresses LH."
        )
    return f"FSH/LH ratio {v:.2f} (≥1) — normal early-follicular pattern."


def explain_tsh(value, direction: str) -> str:
    if pd.isna(value):
        return "TSH not measured — required as a PCOS exclusion (rule out thyroid dysfunction)."
    v = float(value)
    if v > 4.5:
        return f"TSH {v:.2f} mIU/L — elevated. Investigate hypothyroidism before attributing menstrual irregularity to PCOS."
    if v < 0.4:
        return f"TSH {v:.2f} mIU/L — suppressed. Investigate hyperthyroidism."
    return f"TSH {v:.2f} mIU/L — within normal range. Thyroid dysfunction excluded as alternative cause."


def explain_prolactin(value, direction: str) -> str:
    if pd.isna(value):
        return "Prolactin not measured — required as a PCOS exclusion (rule out hyperprolactinemia)."
    v = float(value)
    if v > 25:
        return f"Prolactin {v:.1f} ng/mL — elevated. Hyperprolactinemia mimics PCOS; investigate pituitary."
    return f"Prolactin {v:.1f} ng/mL — within normal range. Hyperprolactinemia excluded as alternative cause."


def explain_rbs(value, direction: str) -> str:
    if pd.isna(value):
        return "Random blood sugar not measured."
    v = float(value)
    if v >= 200:
        return f"RBS {v:.0f} mg/dL — diabetes range. Urgent metabolic follow-up."
    if v >= 140:
        return f"RBS {v:.0f} mg/dL — impaired tolerance. Order fasting glucose / HbA1c / OGTT."
    return f"RBS {v:.0f} mg/dL — within normal range, but RBS is a coarse screen; HbA1c is preferred."


def explain_age(value, direction: str) -> str:
    if pd.isna(value):
        return "Age not recorded."
    v = int(value)
    if v < 25:
        return f"Age {v} — early reproductive age. Note: in adolescents, normal pubertal cycle variability can mimic PCOS."
    if v >= 40:
        return f"Age {v} — approaching the perimenopausal window. AMH naturally declines; standard PCOS thresholds may be less informative."
    return f"Age {v} — reproductive prime; standard diagnostic criteria apply directly."


def explain_fast_food(value, direction: str) -> str:
    if pd.isna(value):
        return "Diet history not recorded."
    if int(value) == 1:
        return "Frequent fast food consumption reported — modifiable risk factor; counsel on diet change."
    return "Fast food not regularly consumed."


def explain_exercise(value, direction: str) -> str:
    if pd.isna(value):
        return "Exercise habits not recorded."
    if int(value) == 1:
        return "Regular exercise reported — protective; reinforce."
    return "No regular exercise. Aerobic + resistance training ≥150 min/week improves insulin sensitivity and ovulation."


def explain_cycle_length(value, direction: str) -> str:
    if pd.isna(value):
        return "Menstrual cycle length not recorded."
    v = float(value)
    return (
        f"Cycle-length field = {v:.0f}. Note: this dataset's column appears to encode bleed *duration* "
        f"rather than cycle-to-cycle length, so cannot reliably test the >35-day oligomenorrhea threshold. "
        f"Verify with patient."
    )


def explain_endometrium(value, direction: str) -> str:
    if pd.isna(value):
        return "Endometrial thickness not measured."
    v = float(value)
    if v > 15:
        return (
            f"Endometrial thickness {v:.1f} mm — markedly thickened. Sustained unopposed estrogen "
            f"raises endometrial hyperplasia / cancer risk. Recommend cyclical progestin therapy "
            f"and consider endometrial biopsy."
        )
    return f"Endometrial thickness {v:.1f} mm — interpret with cycle phase context."


def explain_bp_systolic(value, direction: str) -> str:
    if pd.isna(value):
        return "Blood pressure not recorded."
    v = float(value)
    if v >= 140:
        return f"Systolic blood pressure {v:.0f} mmHg — hypertensive range. Cardiovascular risk factor."
    if v >= 130:
        return f"Systolic blood pressure {v:.0f} mmHg — elevated. Monitor and address modifiable lifestyle factors."
    return f"Systolic blood pressure {v:.0f} mmHg — within normal range."


def explain_bp_diastolic(value, direction: str) -> str:
    if pd.isna(value):
        return "Blood pressure not recorded."
    v = float(value)
    if v >= 90:
        return f"Diastolic blood pressure {v:.0f} mmHg — hypertensive range. Cardiovascular risk factor."
    if v >= 85:
        return f"Diastolic blood pressure {v:.0f} mmHg — elevated. Monitor alongside other metabolic markers."
    return f"Diastolic blood pressure {v:.0f} mmHg — within normal range."


def explain_fsh(value, direction: str) -> str:
    if pd.isna(value):
        return "Follicle-stimulating hormone not measured."
    v = float(value)
    if v > 25:
        return (
            f"FSH {v:.1f} mIU/mL — elevated. This pattern suggests diminished ovarian "
            f"reserve rather than PCOS; consider primary ovarian insufficiency."
        )
    return (
        f"FSH {v:.1f} mIU/mL — within the normal-to-low range typical of PCOS, "
        f"where luteinising hormone usually predominates over FSH."
    )


def explain_lh(value, direction: str) -> str:
    if pd.isna(value):
        return "Luteinising hormone not measured."
    v = float(value)
    if v >= 10:
        return (
            f"LH {v:.1f} mIU/mL — elevated. Raised luteinising hormone is a characteristic "
            f"PCOS pattern and drives excess ovarian androgen production."
        )
    return f"LH {v:.1f} mIU/mL — interpret alongside FSH and cycle phase."


def explain_haemoglobin(value, direction: str) -> str:
    if pd.isna(value):
        return "Haemoglobin not measured."
    v = float(value)
    if v < 12:
        return (
            f"Haemoglobin {v:.1f} g/dL — below the normal range for women. "
            f"Consider iron-deficiency anaemia, which may indicate heavy menstrual bleeding."
        )
    return f"Haemoglobin {v:.1f} g/dL — within the normal range."


def explain_progesterone(value, direction: str) -> str:
    if pd.isna(value):
        return "Progesterone not measured."
    v = float(value)
    return (
        f"Progesterone {v:.1f} ng/mL. A low mid-luteal progesterone is consistent with "
        f"anovulation; interpret in the context of the cycle day on which it was drawn."
    )


def explain_vitamin_d(value, direction: str) -> str:
    if pd.isna(value):
        return "Vitamin D not measured."
    v = float(value)
    if v < 20:
        return (
            f"Vitamin D {v:.1f} ng/mL — deficient. Vitamin D deficiency is common alongside "
            f"PCOS and insulin resistance; supplementation is low-risk."
        )
    return f"Vitamin D {v:.1f} ng/mL — adequate."


def explain_follicle_size(value, direction: str, side: str = "ovary") -> str:
    if pd.isna(value):
        return f"Mean follicle size for the {side} not recorded."
    v = float(value)
    return (
        f"Mean follicle diameter {v:.1f} mm in the {side}. In PCOS, follicles are "
        f"characteristically small and uniform (2-9 mm) and arrested before a dominant "
        f"follicle develops."
    )


def explain_follicle_size_l(value, direction: str) -> str:
    return explain_follicle_size(value, direction, "left ovary")


def explain_follicle_size_r(value, direction: str) -> str:
    return explain_follicle_size(value, direction, "right ovary")


def explain_beta_hcg(value, direction: str) -> str:
    if pd.isna(value):
        return "Beta-hCG not measured."
    v = float(value)
    if v > 5:
        return (
            f"Beta-hCG {v:.1f} mIU/mL — above the non-pregnant range. Exclude pregnancy "
            f"before attributing menstrual changes to PCOS."
        )
    return f"Beta-hCG {v:.1f} mIU/mL — consistent with a non-pregnant state."




EXPLANATION_REGISTRY: Dict[str, Callable] = {
    "AMH(ng/mL)": explain_amh,
    "Follicle No. (L)": explain_follicle_l,
    "Follicle No. (R)": explain_follicle_r,
    "Cycle(R/I)": explain_cycle_regularity,
    "hair growth(Y/N)": explain_hair_growth,
    "Skin darkening (Y/N)": explain_skin_darkening,
    "Weight gain(Y/N)": explain_weight_gain,
    "Hair loss(Y/N)": explain_hair_loss,
    "Pimples(Y/N)": explain_pimples,
    "BMI": explain_bmi,
    "Waist:Hip Ratio": explain_waist_hip,
    "FSH/LH": explain_fsh_lh,
    "TSH (mIU/L)": explain_tsh,
    "PRL(ng/mL)": explain_prolactin,
    "RBS(mg/dl)": explain_rbs,
    "Age (yrs)": explain_age,
    "Fast food (Y/N)": explain_fast_food,
    "Reg.Exercise(Y/N)": explain_exercise,
    "Cycle length(days)": explain_cycle_length,
    "Endometrium (mm)": explain_endometrium,
    "BP _Systolic (mmHg)": explain_bp_systolic,
    "BP _Diastolic (mmHg)": explain_bp_diastolic,
    "FSH(mIU/mL)": explain_fsh,
    "LH(mIU/mL)": explain_lh,
    "Hb(g/dl)": explain_haemoglobin,
    "PRG(ng/mL)": explain_progesterone,
    "Vit D3 (ng/mL)": explain_vitamin_d,
    "Avg. F size (L) (mm)": explain_follicle_size_l,
    "Avg. F size (R) (mm)": explain_follicle_size_r,
    "I beta-HCG(mIU/mL)": explain_beta_hcg,
    "II beta-HCG(mIU/mL)": explain_beta_hcg,
}


def explain_feature(feature: str, value, direction: str) -> Optional[str]:
    fn = EXPLANATION_REGISTRY.get(feature)
    if fn is not None:
        return fn(value, direction)
    if direction == "+":
        return (
            "This measurement is among the factors most strongly associated "
            "with increased PCOS risk in this patient's profile."
        )
    return (
        "This measurement is among the factors that lower the assessed PCOS "
        "risk in this patient's profile."
    )
