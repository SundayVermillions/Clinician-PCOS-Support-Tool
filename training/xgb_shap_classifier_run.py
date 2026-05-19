
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.data_loading import (  # noqa: E402
    TARGET,
    get_feature_sets,
    load_pcos_data,
    split_feature_types,
)
from src.tabular import (  # noqa: E402
    RANDOM_STATE,
    build_logreg_pipeline,
    build_xgb_pipeline,
    compute_shap_from_fitted,
    evaluate_cv,
    metrics_dict,
    train_calibrated,
)


def run(repo_root: Path = REPO_ROOT) -> None:
    df = load_pcos_data()
    df.to_parquet(repo_root / "data/processed/df_clean.parquet")
    print(f"  Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"  Class balance: {df[TARGET].value_counts().to_dict()}")

    feature_sets = get_feature_sets(df)
    print(f"  all: {len(feature_sets['all'])} features")
    print(f"  primary_care: {len(feature_sets['primary_care'])} features")
    print(f"  specialist (excluded for PC): {feature_sets['specialist']}")

    y = df[TARGET].astype(int)

    all_metrics = []
    fitted_models = {}
    shap_tables = {}
    cv_results = {}

    for set_name in ["all", "primary_care"]:
        print(f"\n--- Feature set: {set_name} ---")
        features = feature_sets[set_name]
        types = split_feature_types(features)
        X = df[features]

        print("logreg start")
        logreg = build_logreg_pipeline(types["numeric"], types["categorical"])
        cv_lr = evaluate_cv(logreg, X, y)
        all_metrics.append(metrics_dict(f"logreg_{set_name}", cv_lr))
        print(f"    ROC-AUC: {cv_lr.roc_auc:.3f}  PR-AUC: {cv_lr.pr_auc:.3f}  Brier: {cv_lr.brier:.3f}")


        print("xgb start")
        xgb = build_xgb_pipeline(types["numeric"], types["categorical"])
        cv_xgb = evaluate_cv(xgb, X, y)
        all_metrics.append(metrics_dict(f"xgb_{set_name}", cv_xgb))
        print(f"    ROC-AUC: {cv_xgb.roc_auc:.3f}  PR-AUC: {cv_xgb.pr_auc:.3f}  Brier: {cv_xgb.brier:.3f}")
        calibrated = train_calibrated(xgb, X, y)
        calibrated_proba = calibrated.predict_proba(X)[:, 1]
        xgb_raw = build_xgb_pipeline(types["numeric"], types["categorical"])
        xgb_raw.fit(X, y)


        model_path = repo_root / f"models/classifier_{set_name}.pkl"
        joblib.dump(
            {
                "model": calibrated,
                "xgb_raw": xgb_raw,
                "features": features,
                "numeric_features": types["numeric"],
                "categorical_features": types["categorical"],
            },
            model_path,
        )
        print(f"  Saved bundle -> {model_path}")

        print("shap start")
        xgb_for_shap = build_xgb_pipeline(types["numeric"], types["categorical"])
        xgb_for_shap.fit(X, y)
        shap_deployed, _ = compute_shap_from_fitted(
            xgb_for_shap, X, types["numeric"], types["categorical"]
        )
        shap_path = repo_root / f"results/tables/shap_{set_name}.parquet"
        shap_deployed.to_parquet(shap_path)
        print(f"  Saved deployed SHAP ({shap_deployed.shape}) -> {shap_path}")


        print("shap test start")
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_shap = pd.DataFrame(index=X.index, columns=features, dtype=float)
        for fold, (train_idx, test_idx) in enumerate(cv_inner.split(X, y), 1):
            pipe = build_xgb_pipeline(types["numeric"], types["categorical"])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            sh, _ = compute_shap_from_fitted(
                pipe,
                X.iloc[test_idx],
                types["numeric"],
                types["categorical"],
            )
            oof_shap.iloc[test_idx] = sh.values
        oof_shap_path = repo_root / f"results/tables/shap_oof_{set_name}.parquet"
        oof_shap.to_parquet(oof_shap_path)
        print(f"  Saved OOF SHAP ({oof_shap.shape}) -> {oof_shap_path}")

        fitted_models[set_name] = calibrated
        shap_tables[set_name] = shap_deployed
        cv_results[set_name] = {
            "logreg": cv_lr,
            "xgb": cv_xgb,
            "calibrated_proba": calibrated_proba,
            "oof_proba": cv_xgb.y_proba,  # uncalibrated XGBoost OOF
        }

    print()
    print("export")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(repo_root / "results/tables/baseline_metrics.csv", index=False)
    print("  Metrics:")
    print(metrics_df.to_string(index=False))

    oof_proba_df = pd.DataFrame(
        {name: r["oof_proba"] for name, r in cv_results.items()},
        index=df.index,
    )
    oof_proba_df.to_parquet(repo_root / "results/tables/cv_probabilities.parquet")
    print(f"  Saved OOF probabilities -> results/tables/cv_probabilities.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (set_name, results) in zip(axes, cv_results.items()):
        prob_true_raw, prob_pred_raw = calibration_curve(
            y, results["xgb"].y_proba, n_bins=8, strategy="quantile"
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y, results["calibrated_proba"], n_bins=8, strategy="quantile"
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(prob_pred_raw, prob_true_raw, "o-", label="XGBoost (raw)")
        ax.plot(prob_pred_cal, prob_true_cal, "s-", label="XGBoost (calibrated)")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(f"Feature set: {set_name}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed PCOS frequency")
    fig.suptitle("Calibration curves (cross-validated predictions, isotonic refit)")
    fig.tight_layout()
    fig_path = repo_root / "results/figures/calibration_curves.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"  Saved calibration plot -> {fig_path}")

    print("\nDone.")
    print("\nArtifacts:")
    for p in [
        "data/processed/df_clean.parquet",
        "models/classifier_all.pkl",
        "models/classifier_primary_care.pkl",
        "results/tables/baseline_metrics.csv",
        "results/tables/cv_probabilities.parquet",
        "results/tables/shap_all.parquet",
        "results/tables/shap_primary_care.parquet",
        "results/tables/shap_oof_all.parquet",
        "results/tables/shap_oof_primary_care.parquet",
        "results/figures/calibration_curves.png",
    ]:
        full = repo_root / p
        status = "ok" if full.exists() else "error"
        print(f"  {status} {p}")


if __name__ == "__main__":
    run()
