"""
Financial Risk Model Pipeline
LendingClub Dataset: EDA → Preprocessing → Logistic Regression → XGBoost → SHAP → Calibration

Usage:
    python pipeline.py --data data/lending_club.csv
    python pipeline.py --demo   (runs on synthetic data for testing)
"""

import argparse
import json
import os
import warnings
import joblib

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANDOM_STATE = 42
OUTPUT_DIR   = "outputs"
MODEL_DIR    = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# STEP 0 — DATA LOADING
# ═══════════════════════════════════════════════════════════════

# Core LendingClub columns we care about for credit risk
LENDING_CLUB_FEATURES = [
    "loan_amnt",        # Loan amount requested
    "int_rate",         # Interest rate
    "installment",      # Monthly payment
    "annual_inc",       # Annual income
    "dti",              # Debt-to-income ratio
    "delinq_2yrs",      # Delinquencies in past 2 years
    "fico_range_low",   # FICO score (low)
    "inq_last_6mths",   # Credit inquiries last 6 months
    "open_acc",         # Open credit lines
    "pub_rec",          # Public derogatory records
    "revol_bal",        # Revolving balance
    "revol_util",       # Revolving utilization rate
    "total_acc",        # Total credit lines
    "mort_acc",         # Mortgage accounts
    "pub_rec_bankruptcies",  # Bankruptcies on record
    "issue_d",          # Issue date (for time-based split)
    "loan_status",      # Target variable
]

TARGET_POSITIVE = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]


def load_lending_club(path: str) -> pd.DataFrame:
    """Load and minimally parse a LendingClub CSV."""
    print(f"\n[DATA] Loading: {path}")
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in LENDING_CLUB_FEATURES)
    print(f"[DATA] Raw shape: {df.shape}")
    return df


def make_synthetic_lending_club(n: int = 15000) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors LendingClub's schema.
    Used for testing / demo when real data isn't present.
    Class imbalance is intentionally realistic (~20% default).
    """
    print(f"\n[DATA] Generating synthetic LendingClub data (n={n})...")
    rng = np.random.default_rng(RANDOM_STATE)

    # Base features
    df = pd.DataFrame({
        "loan_amnt":          rng.integers(1000, 40000, n),
        "int_rate":           rng.uniform(5, 30, n),
        "installment":        rng.uniform(50, 1500, n),
        "annual_inc":         rng.lognormal(10.8, 0.6, n),          # realistic income dist
        "dti":                rng.uniform(0, 40, n),
        "delinq_2yrs":        rng.integers(0, 5, n),
        "fico_range_low":     rng.integers(620, 850, n),
        "inq_last_6mths":     rng.integers(0, 6, n),
        "open_acc":           rng.integers(2, 30, n),
        "pub_rec":            rng.integers(0, 3, n),
        "revol_bal":          rng.integers(0, 60000, n),
        "revol_util":         rng.uniform(0, 100, n),
        "total_acc":          rng.integers(5, 50, n),
        "mort_acc":           rng.integers(0, 5, n),
        "pub_rec_bankruptcies": rng.integers(0, 2, n),
    })

    # Synthetic issue dates (2015–2022 for temporal split testing)
    years  = rng.integers(2015, 2023, n)
    months = rng.integers(1, 13, n)
    df["issue_d"] = pd.to_datetime(
        {"year": years, "month": months, "day": 1}
    ).dt.strftime("%b-%Y")

    # Default probability driven by real risk factors
    log_odds = (
        -4.0
        + 0.06 * (df["int_rate"])
        + 0.04 * (df["dti"])
        + 0.30 * (df["delinq_2yrs"])
        - 0.02 * (df["fico_range_low"] - 700)
        + 0.15 * (df["pub_rec"])
        + 0.02 * (df["inq_last_6mths"])
        + 0.001 * (df["revol_util"])
        - 0.000008 * (df["annual_inc"])
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    df["loan_status"] = np.where(
        rng.random(n) < prob_default, "Charged Off", "Fully Paid"
    )

    default_rate = (df["loan_status"] == "Charged Off").mean()
    print(f"[DATA] Synthetic default rate: {default_rate:.1%}")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 1 — EDA
# ═══════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, target: str = "default") -> None:
    """
    EDA:
    - Class distribution
    - Missing values heatmap
    - Feature distributions by class
    - Correlation matrix
    """
    print("\n" + "═"*55)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("═"*55)

    print(f"\n[EDA] Dataset shape:  {df.shape}")
    print(f"[EDA] Default rate:   {df[target].mean():.2%}")
    print(f"\n[EDA] Missing values (top 10):")
    missing = df.isnull().mean().sort_values(ascending=False).head(10)
    print(missing[missing > 0].to_string())

    features         = [c for c in df.columns if c != target]
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Class distribution
    ax0 = fig.add_subplot(gs[0, 0])
    counts = df[target].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax0.bar(["No Default\n(0)", "Default\n(1)"], counts, color=colors, edgecolor="white", linewidth=1.5)
    ax0.set_title("Class Distribution", fontweight="bold")
    ax0.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax0.text(i, v + 30, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=9)

    # 2. Missing values
    ax1 = fig.add_subplot(gs[0, 1])
    missing_vals = df[numeric_features].isnull().mean() * 100
    missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=True)
    if len(missing_vals) > 0:
        ax1.barh(missing_vals.index, missing_vals.values, color="#3498db")
        ax1.set_title("Missing Values (%)", fontweight="bold")
        ax1.set_xlabel("% Missing")
    else:
        ax1.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=12)
        ax1.set_title("Missing Values", fontweight="bold")

    # 3. Correlation heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    corr = df[numeric_features[:8]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax2, cmap="RdYlGn", center=0,
                linewidths=0.5, annot=False, fmt=".2f", cbar_kws={"shrink": 0.8})
    ax2.set_title("Feature Correlation (top 8)", fontweight="bold")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2.tick_params(axis="y", rotation=0, labelsize=7)

    # 4–9: Distribution plots for top 6 features by correlation with target
    top_features = (
        df[numeric_features].corrwith(df[target]).abs().sort_values(ascending=False).head(6).index
    )
    for i, feat in enumerate(top_features):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        for label, color, alpha in [(0, "#2ecc71", 0.6), (1, "#e74c3c", 0.6)]:
            vals = df[df[target] == label][feat].dropna()
            ax.hist(vals, bins=30, alpha=alpha, color=color,
                    label=f"{'Default' if label else 'No Default'}",
                    density=True, edgecolor="none")
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel("Density", fontsize=7)

    fig.suptitle("LendingClub Credit Risk — Exploratory Data Analysis",
                 fontsize=14, fontweight="bold", y=1.01)

    path = f"{OUTPUT_DIR}/eda_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def preprocess(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    - Encode target
    - Parse issue_d for temporal split
    - Drop high-missingness columns
    - Impute remaining NaN
    - Return X, y, issue_dates
    """
    print("\n" + "═"*55)
    print("STEP 2: PREPROCESSING")
    print("═"*55)

    df = df_raw.copy()

    # — Target encoding
    df["default"] = df["loan_status"].apply(
        lambda s: 1 if s in TARGET_POSITIVE else 0
    )
    df.drop(columns=["loan_status"], inplace=True)

    # — Parse issue date
    issue_dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df.drop(columns=["issue_d"], inplace=True, errors="ignore")

    # — Drop columns with >40% missing
    thresh   = 0.4
    before   = df.shape[1]
    df       = df.loc[:, df.isnull().mean() < thresh]
    dropped  = before - df.shape[1]
    print(f"[PREP] Dropped {dropped} columns (>{thresh:.0%} missing)")

    # — Separate target
    y = df.pop("default")
    X = df.select_dtypes(include=[np.number])

    # — Impute with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    print(f"[PREP] Final feature matrix: {X.shape}")
    print(f"[PREP] Features: {list(X.columns)}")
    return X, y, issue_dates


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    issue_dates: pd.Series,
    cutoff_year: int = 2020,
) -> tuple:
    """
    Split data by time — train on pre-cutoff, test on cutoff+.
    Auto-detects cutoff if hard-coded year yields an empty test set.
    Falls back to 80/20 stratified split if dates are unavailable.
    """
    print("\n" + "="*55)
    print("STEP 3: TEMPORAL TRAIN / TEST SPLIT")
    print("="*55)

    from sklearn.model_selection import train_test_split

    valid_dates = issue_dates.notna()
    if valid_dates.sum() > 100:
        # Auto-detect: use the last 20% of the date range as test window
        min_date   = issue_dates[valid_dates].min()
        max_date   = issue_dates[valid_dates].max()
        date_range = max_date - min_date
        auto_cutoff = min_date + date_range * 0.80
        
        # If hard-coded cutoff year would leave test empty, use auto-cutoff
        hard_cutoff = pd.Timestamp(f"{cutoff_year}-01-01")
        cutoff = hard_cutoff if hard_cutoff < max_date else auto_cutoff
        
        train_mask = (issue_dates < cutoff) & valid_dates
        test_mask  = (issue_dates >= cutoff) & valid_dates

        # Safety: if test still too small, fall back to stratified split
        if test_mask.sum() < 100:
            print(f"[SPLIT] Date-based split too small — falling back to 80/20 stratified")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
            )
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test,  y_test  = X[test_mask],  y[test_mask]
            print(f"[SPLIT] Date range: {min_date.date()} → {max_date.date()}")
            print(f"[SPLIT] Cutoff: {cutoff.date()}")
            print(f"[SPLIT] Train: {len(X_train):,} rows | Default rate: {y_train.mean():.2%}")
            print(f"[SPLIT] Test:  {len(X_test):,} rows  | Default rate: {y_test.mean():.2%}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        print(f"[SPLIT] No valid dates — using 80/20 stratified split")
        print(f"[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    feature_names = X_train.columns.tolist()
    return (
        X_train.values.astype(np.float64),
        X_test.values.astype(np.float64),
        y_train.values,
        y_test.values,
        feature_names,
    )


SMOTE_MAX_ROWS = 50_000  # Above this, skip SMOTE — XGBoost scale_pos_weight handles it

def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Oversample minority class with SMOTE on training set only.
    Skipped for large datasets (>50k rows) where it would be prohibitively slow
    and memory-intensive. XGBoost's scale_pos_weight handles imbalance natively.
    """
    n_default    = int(y_train.sum())
    n_nodefault  = int((y_train == 0).sum())
    print(f"\n[SMOTE] Before — Default: {n_default:,}, Non-default: {n_nodefault:,}")

    if len(X_train) > SMOTE_MAX_ROWS:
        print(f"[SMOTE] Dataset too large ({len(X_train):,} rows) — skipping SMOTE.")
        print(f"[SMOTE] Class imbalance handled via XGBoost scale_pos_weight instead.")
        return X_train, y_train

    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] After  — Default: {int(y_res.sum()):,}, Non-default: {int((y_res==0).sum()):,}")
    return X_res, y_res


# ═══════════════════════════════════════════════════════════════
# STEP 4 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train) -> Pipeline:
    """Logistic Regression with standard scaling — interpretable baseline."""
    print("\n[MODEL] Training Logistic Regression...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=0.1,
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """XGBoost — production-grade gradient boosting for tabular data."""
    print("[MODEL] Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # handle imbalance natively
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    return model


# ═══════════════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax_roc=None,
    ax_pr=None,
) -> dict:
    """Return full metric dict and optionally plot ROC / PR curves."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc  = auc(rec, prec)
    brier   = brier_score_loss(y_test, y_prob)

    metrics = {
        "ROC-AUC":  round(roc_auc, 4),
        "PR-AUC":   round(pr_auc,  4),
        "Brier":    round(brier,   4),
        "F1":       round(f1_score(y_test, y_pred), 4),
    }

    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    if ax_roc is not None:
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    if ax_pr is not None:
        ax_pr.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})", linewidth=2)

    return metrics


def plot_confusion_matrix(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted\nNo Default", "Predicted\nDefault"],
                yticklabels=["Actual\nNo Default", "Actual\nDefault"])
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    path = f"{OUTPUT_DIR}/confusion_{name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {path}")


def plot_roc_pr(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Plot ROC and PR curves for multiple models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)
        axes[1].plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})", linewidth=2)

    # ROC
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # PR
    baseline = y_test.mean()
    axes[1].axhline(y=baseline, color="k", linestyle="--", linewidth=1,
                    label=f"Baseline ({baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Model Comparison — ROC & PR Curves", fontweight="bold")
    path = f"{OUTPUT_DIR}/roc_pr_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════

def run_shap(model: XGBClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values and generate:
    1. Summary (beeswarm) plot — global feature importance
    2. Bar plot — mean |SHAP| per feature
    3. Force plot for a single high-risk sample
    """
    print("\n" + "═"*55)
    print("STEP 6: SHAP EXPLAINABILITY")
    print("═"*55)

    print("[SHAP] Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # — Global: beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary — Global Feature Impact", fontweight="bold")
    path = f"{OUTPUT_DIR}/shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved: {path}")

    # — Global: bar
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontweight="bold")
    path = f"{OUTPUT_DIR}/shap_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved: {path}")

    # — Local: highest-risk sample
    y_prob  = model.predict_proba(X_test)[:, 1]
    highest = int(np.argmax(y_prob))
    print(f"\n[SHAP] Local explanation for sample #{highest} (p_default={y_prob[highest]:.3f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values     = shap_values[highest],
            base_values= explainer.expected_value,
            data       = X_test.iloc[highest].values,
            feature_names= X_test.columns.tolist(),
        ),
        show=False,
    )
    plt.title(f"SHAP Waterfall — Sample #{highest} (p={y_prob[highest]:.3f})", fontweight="bold")
    path = f"{OUTPUT_DIR}/shap_local_waterfall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved: {path}")

    return shap_values


def extract_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list,
    sample_idx: int = 0,
    top_n: int = 5,
) -> list[dict]:
    """
    Extract top-N most impactful features for a single prediction.
    Returns JSON-friendly list for the API response.
    """
    sv = shap_values[sample_idx]
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs[:top_n]]


# ═══════════════════════════════════════════════════════════════
# STEP 7 — CALIBRATION
# ═══════════════════════════════════════════════════════════════

from shared import CalibratedXGB  # must live in shared.py so pickle path is stable


def calibrate_model(model: XGBClassifier, X_train, y_train) -> "CalibratedXGB":
    """
    Apply Platt scaling (sigmoid calibration) to make probabilities trustworthy.
    Uses a manual wrapper to avoid the XGBoost float32 / sklearn float64 dtype clash.
    """
    print("\n" + "="*55)
    print("STEP 7: PROBABILITY CALIBRATION")
    print("="*55)
    print("[CALIB] Applying Platt scaling (sigmoid)...")
    calibrated = CalibratedXGB(model)
    calibrated.fit(X_train, y_train)
    print("[CALIB] Calibration complete.")
    return calibrated


def plot_calibration_curve(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Plot calibration curves for raw vs calibrated model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        brier  = brier_score_loss(y_test, y_prob)
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                label=f"{name} (Brier={brier:.4f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    path = f"{OUTPUT_DIR}/calibration_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CALIB] Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 8 — SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════

RISK_THRESHOLDS = {
    "Low":      (0.00, 0.30),
    "Medium":   (0.30, 0.55),
    "High":     (0.55, 0.75),
    "Critical": (0.75, 1.01),
}


def score_to_category(score: float) -> str:
    for cat, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return cat
    return "Critical"


def save_model_artifacts(
    xgb_model,
    calibrated_model,
    lr_model,
    feature_names: list,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    metrics: dict,
) -> None:
    """Persist all model artifacts for API consumption."""
    joblib.dump(calibrated_model, f"{MODEL_DIR}/xgb_calibrated.joblib")
    joblib.dump(lr_model,         f"{MODEL_DIR}/logistic_regression.joblib")
    joblib.dump(xgb_model,        f"{MODEL_DIR}/xgb_raw.joblib")

    # Feature metadata
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_meta = [
        {"feature": f, "mean_abs_shap": round(float(v), 6)}
        for f, v in sorted(zip(feature_names, mean_shap), key=lambda x: -x[1])
    ]

    meta = {
        "feature_names": feature_names,
        "feature_shap_importance": feature_meta,
        "risk_thresholds": RISK_THRESHOLDS,
        "metrics": metrics,
    }
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE] Artifacts persisted to: {MODEL_DIR}/")
    print(f"[SAVE] xgb_calibrated.joblib | logistic_regression.joblib | model_meta.json")


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(data_path: str = None, demo: bool = False) -> None:
    print("\n" + "█"*55)
    print("  FINANCIAL RISK MODEL — FULL PIPELINE")
    print("  LendingClub Credit Default Prediction")
    print("█"*55)

    # ── Load ──────────────────────────────────────────────────
    if demo or data_path is None:
        df_raw = make_synthetic_lending_club(n=15000)
    else:
        df_raw = load_lending_club(data_path)

    # ── EDA ───────────────────────────────────────────────────
    # Preview before preprocessing
    df_eda = df_raw.copy()
    df_eda["default"] = df_eda["loan_status"].apply(
        lambda s: 1 if s in TARGET_POSITIVE else 0
    )
    run_eda(df_eda)

    # ── Preprocess ────────────────────────────────────────────
    X, y, issue_dates = preprocess(df_raw)

    # ── Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = temporal_split(X, y, issue_dates)

    # ── SMOTE (training set only) ─────────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── Train (with checkpointing) ────────────────────────────
    print("\n" + "="*55)
    print("STEP 4: MODEL TRAINING")
    print("="*55)

    lr_ckpt  = os.path.join(MODEL_DIR, "checkpoint_lr.joblib")
    xgb_ckpt = os.path.join(MODEL_DIR, "checkpoint_xgb.joblib")

    # Checkpoint signature — invalidates saved models if features change
    sig_path = os.path.join(MODEL_DIR, "checkpoint_sig.json")
    current_sig = {"features": feature_names, "n_features": len(feature_names)}
    saved_sig = json.load(open(sig_path)) if os.path.exists(sig_path) else {}
    checkpoints_valid = (saved_sig == current_sig)
    if not checkpoints_valid and os.path.exists(sig_path):
        print("[CKPT] Feature signature changed — invalidating old checkpoints")

    if checkpoints_valid and os.path.exists(lr_ckpt):
        print("[CKPT] Loading Logistic Regression from checkpoint...")
        lr_model = joblib.load(lr_ckpt)
    else:
        lr_model = train_logistic_regression(X_train_res, y_train_res)
        joblib.dump(lr_model, lr_ckpt)
        print(f"[CKPT] Saved: {lr_ckpt}")

    if checkpoints_valid and os.path.exists(xgb_ckpt):
        print("[CKPT] Loading XGBoost from checkpoint...")
        xgb_model = joblib.load(xgb_ckpt)
    else:
        xgb_model = train_xgboost(X_train_res, y_train_res)
        joblib.dump(xgb_model, xgb_ckpt)
        print(f"[CKPT] Saved: {xgb_ckpt}")

    # Save signature after successful training
    with open(sig_path, "w") as f:
        json.dump(current_sig, f)

    # ── Evaluate ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 5: MODEL EVALUATION")
    print("="*55)
    metrics_lr  = evaluate_model("Logistic Regression", lr_model,  X_test, y_test)
    metrics_xgb = evaluate_model("XGBoost",             xgb_model, X_test, y_test)

    for m_name, model in [("Logistic Regression", lr_model), ("XGBoost", xgb_model)]:
        plot_confusion_matrix(m_name, model, X_test, y_test)

    plot_roc_pr(
        {"Logistic Regression": lr_model, "XGBoost": xgb_model},
        X_test, y_test,
    )

    # ── SHAP ──────────────────────────────────────────────────
    # Wrap X as DataFrames with feature names for SHAP plots only
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test,  columns=feature_names)
    shap_values = run_shap(xgb_model, X_train_df, X_test_df)

    # ── Calibrate ─────────────────────────────────────────────
    calibrated = calibrate_model(xgb_model, X_train, y_train)

    plot_calibration_curve(
        {"XGBoost (raw)":        xgb_model,
         "XGBoost (calibrated)": calibrated},
        X_test, y_test,
    )

    # ── Evaluate calibrated model ─────────────────────────────
    metrics_cal = evaluate_model("XGBoost (calibrated)", calibrated, X_test, y_test)

    # ── Save ──────────────────────────────────────────────────
    combined_metrics = {
        "logistic_regression":  metrics_lr,
        "xgboost":              metrics_xgb,
        "xgboost_calibrated":   metrics_cal,
    }
    save_model_artifacts(
        xgb_model, calibrated, lr_model,
        feature_names=feature_names,
        shap_values=shap_values,
        X_test=X_test_df,
        metrics=combined_metrics,
    )

    # ── Sample API payload ────────────────────────────────────
    print("\n" + "═"*55)
    print("STEP 8: SAMPLE API OUTPUT")
    print("═"*55)
    sample = X_test[[0]]  # X_test is now numpy
    score  = float(calibrated.predict_proba(sample)[0, 1])
    top_feats = extract_top_shap_features(shap_values, feature_names, 0)
    payload = {
        "risk_score":    round(score, 4),
        "risk_category": score_to_category(score),
        "top_features":  top_feats,
    }
    print(json.dumps(payload, indent=2))

    print("\n" + "█"*55)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs → {OUTPUT_DIR}/")
    print(f"  Models  → {MODEL_DIR}/")
    print("█"*55)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to LendingClub CSV")
    parser.add_argument("--demo", action="store_true",    help="Run on synthetic data")
    args = parser.parse_args()
    run_pipeline(data_path=args.data, demo=args.demo or args.data is None)
