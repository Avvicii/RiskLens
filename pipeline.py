"""
LendingClub Credit Default Prediction Pipeline
Usage: python pipeline.py --data path/to/accepted_2007_to_2018q4.csv
"""

import argparse
import json
import os
import warnings
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, brier_score_loss, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from shared import CalibratedXGB

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR   = "outputs"
MODEL_DIR    = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "fico_range_low", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "grade", "emp_length", "home_ownership",
    "purpose", "issue_d", "loan_status",
]

TARGET_POSITIVE = [
    "Charged Off", "Default", "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
EMP_LEN_ORDER = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5,  "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
CAT_ONEHOT = ["home_ownership", "purpose"]

RISK_THRESHOLDS = {
    "Low":      (0.00, 0.30),
    "Medium":   (0.30, 0.55),
    "High":     (0.55, 0.75),
    "Critical": (0.75, 1.01),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False,
                     usecols=lambda c: c in FEATURES,
                     skiprows=lambda i: i == 1)
    df = df.dropna(how="all").reset_index(drop=True)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()

    df["default"] = df["loan_status"].apply(lambda s: 1 if s in TARGET_POSITIVE else 0)
    df.drop(columns=["loan_status"], inplace=True)

    issue_dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df.drop(columns=["issue_d"], inplace=True, errors="ignore")

    if "grade" in df.columns:
        df["grade"] = df["grade"].str.strip().map(GRADE_ORDER).fillna(3).astype(int)

    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].str.strip().map(EMP_LEN_ORDER).fillna(5).astype(float)

    cats = [c for c in CAT_ONEHOT if c in df.columns]
    if cats:
        for col in cats:
            top = df[col].value_counts().nlargest(8).index
            df[col] = df[col].where(df[col].isin(top), other="OTHER")
        df = pd.get_dummies(df, columns=cats, drop_first=False, dtype=int)

    df = df.loc[:, df.isnull().mean() < 0.4]

    y = df.pop("default")
    X = df.select_dtypes(include=[np.number])

    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    print(f"Features: {X.shape[1]}  |  Default rate: {y.mean():.2%}")
    return X, y, issue_dates


def temporal_split(X, y, issue_dates, cutoff_year=2020):
    valid = issue_dates.notna()

    if valid.sum() > 100:
        min_date = issue_dates[valid].min()
        max_date = issue_dates[valid].max()
        cutoff   = pd.Timestamp(f"{cutoff_year}-01-01")
        if cutoff >= max_date:
            cutoff = min_date + (max_date - min_date) * 0.80

        train_mask = (issue_dates < cutoff) & valid
        test_mask  = (issue_dates >= cutoff) & valid

        if test_mask.sum() >= 100:
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask],  y[test_mask]
            print(f"Train: {len(X_tr):,} | Test: {len(X_te):,} | Cutoff: {cutoff.date()}")
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=RANDOM_STATE)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=RANDOM_STATE)

    feature_names = X_tr.columns.tolist()
    return (X_tr.values.astype(np.float64), X_te.values.astype(np.float64),
            y_tr.values, y_te.values, feature_names)


def train_logistic_regression(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   C=0.1, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> XGBClassifier:
    spw   = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="auc",
        random_state=RANDOM_STATE, n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def evaluate_model(name, model, X_test, y_test, threshold=0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    metrics = {
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "PR-AUC":  round(auc(rec, prec), 4),
        "Brier":   round(brier_score_loss(y_test, y_prob), 4),
        "F1":      round(f1_score(y_test, y_pred), 4),
    }
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    return metrics


def plot_confusion_matrix(name, model, X_test, y_test, threshold=0.5) -> None:
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Safe", "Pred Default"],
                yticklabels=["Actual Safe", "Actual Default"])
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.savefig(f"{OUTPUT_DIR}/cm_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr(models_dict, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, lw=2,
                     label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec, prec, lw=2,
                     label=f"{name} (AUC={auc(rec, prec):.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].axhline(y_test.mean(), color="k", linestyle="--", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_shap(model, X_train_df, X_test_df) -> np.ndarray:
    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.savefig(f"{OUTPUT_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    highest = int(np.argmax(model.predict_proba(X_test_df)[:, 1]))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[highest],
            base_values   = explainer.expected_value,
            data          = X_test_df.iloc[highest].values,
            feature_names = X_test_df.columns.tolist(),
        ),
        show=False,
    )
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    return shap_values


def calibrate_model(xgb_model, X_train, y_train) -> CalibratedXGB:
    cal = CalibratedXGB(xgb_model)
    cal.fit(X_train, y_train)
    return cal


def plot_calibration_curve(models, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        frac, mean = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean, frac, marker="o", lw=2,
                label=f"{name} (Brier={brier_score_loss(y_test, y_prob):.4f})")
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title="Calibration Curve")
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def tune_threshold(model, X_test, y_test) -> float:
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = np.where(
        (prec[:-1] + rec[:-1]) > 0,
        2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]),
        0,
    )
    best_idx = np.argmax(f1s)
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])

    print(f"Threshold  default=0.50 → F1: {f1_score(y_test, (y_prob>=0.5).astype(int)):.4f}")
    print(f"Threshold optimal={best_t:.3f} → F1: {best_f1:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(thresholds, prec[:-1], lw=2, color="#58a6ff", label="Precision")
    axes[0].plot(thresholds, rec[:-1],  lw=2, color="#f0a84b", label="Recall")
    axes[0].plot(thresholds, f1s,       lw=2, color="#3fb950", label="F1")
    axes[0].axvline(best_t, color="#f85149", linestyle="--", lw=1.5, label=f"Optimal={best_t:.3f}")
    axes[0].axvline(0.5, color="gray", linestyle=":", lw=1, label="Default=0.50")
    axes[0].set(xlabel="Threshold", title="Precision / Recall / F1 vs Threshold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#f0a84b", lw=2)
    axes[1].scatter(rec[best_idx], prec[best_idx], color="#f85149", s=80, zorder=5,
                    label=f"Optimal (F1={best_f1:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/threshold_optimisation.png", dpi=150, bbox_inches="tight")
    plt.close()
    return best_t


def score_to_category(score: float) -> str:
    for cat, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return cat
    return "Critical"


def save_artifacts(xgb_model, calibrated, lr_model,
                   feature_names, shap_values, metrics) -> None:
    joblib.dump(calibrated, f"{MODEL_DIR}/xgb_calibrated.joblib")
    joblib.dump(xgb_model,  f"{MODEL_DIR}/xgb_raw.joblib")
    joblib.dump(lr_model,   f"{MODEL_DIR}/logistic_regression.joblib")

    mean_shap = np.abs(shap_values).mean(axis=0)
    meta = {
        "feature_names": feature_names,
        "feature_shap_importance": [
            {"feature": f, "mean_abs_shap": round(float(v), 6)}
            for f, v in sorted(zip(feature_names, mean_shap), key=lambda x: -x[1])
        ],
        "risk_thresholds":   RISK_THRESHOLDS,
        "optimal_threshold": metrics.get("optimal_threshold", 0.5),
        "metrics":           metrics,
    }
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts saved → {MODEL_DIR}/")


def run_pipeline(data_path: str) -> None:
    df_raw = load_data(data_path)

    X, y, issue_dates = preprocess(df_raw)

    X_train, X_test, y_train, y_test, feature_names = temporal_split(X, y, issue_dates)

    lr_ckpt  = f"{MODEL_DIR}/checkpoint_lr.joblib"
    xgb_ckpt = f"{MODEL_DIR}/checkpoint_xgb.joblib"
    sig_path = f"{MODEL_DIR}/checkpoint_sig.json"
    sig      = {"features": feature_names, "n": len(feature_names)}
    saved    = json.load(open(sig_path)) if os.path.exists(sig_path) else {}
    valid    = (saved == sig)

    if valid and os.path.exists(lr_ckpt):
        lr_model = joblib.load(lr_ckpt)
        print("LR loaded from checkpoint")
    else:
        lr_model = train_logistic_regression(X_train, y_train)
        joblib.dump(lr_model, lr_ckpt)

    if valid and os.path.exists(xgb_ckpt):
        xgb_model = joblib.load(xgb_ckpt)
        print("XGBoost loaded from checkpoint")
    else:
        xgb_model = train_xgboost(X_train, y_train)
        joblib.dump(xgb_model, xgb_ckpt)

    with open(sig_path, "w") as f:
        json.dump(sig, f)

    metrics_lr  = evaluate_model("Logistic Regression", lr_model,  X_test, y_test)
    metrics_xgb = evaluate_model("XGBoost",             xgb_model, X_test, y_test)

    for name, model in [("Logistic Regression", lr_model), ("XGBoost", xgb_model)]:
        plot_confusion_matrix(name, model, X_test, y_test)
    plot_roc_pr({"Logistic Regression": lr_model, "XGBoost": xgb_model}, X_test, y_test)

    X_train_df  = pd.DataFrame(X_train, columns=feature_names)
    X_test_df   = pd.DataFrame(X_test,  columns=feature_names)
    shap_values = run_shap(xgb_model, X_train_df, X_test_df)

    calibrated = calibrate_model(xgb_model, X_train, y_train)
    plot_calibration_curve(
        {"XGBoost (raw)": xgb_model, "XGBoost (calibrated)": calibrated},
        X_test, y_test,
    )

    optimal_threshold = tune_threshold(calibrated, X_test, y_test)
    metrics_cal = evaluate_model("XGBoost (calibrated)", calibrated,
                                 X_test, y_test, threshold=optimal_threshold)

    all_metrics = {
        "logistic_regression": metrics_lr,
        "xgboost":             metrics_xgb,
        "xgboost_calibrated":  metrics_cal,
        "optimal_threshold":   optimal_threshold,
    }
    save_artifacts(xgb_model, calibrated, lr_model,
                   feature_names, shap_values, all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    run_pipeline(args.data)
