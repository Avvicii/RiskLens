# Financial Risk Model
**LendingClub Credit Default Prediction — Phase 1**

Predicts `P(default | financial features)` using a calibrated XGBoost model
with full SHAP explainability and a FastAPI microservice.

---

## Architecture

```
pipeline.py          ← Full ML pipeline (EDA → Train → SHAP → Calibrate → Save)
api/main.py          ← FastAPI microservice (serves saved models)
models/              ← Saved artifacts (auto-created by pipeline)
outputs/             ← Plots (EDA, ROC, SHAP, calibration)
data/                ← Place your LendingClub CSV here
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run pipeline (demo mode — synthetic data)
```bash
python pipeline.py --demo
```

### 3. Run pipeline with real LendingClub data
Download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club
```bash
python pipeline.py --data data/lending_club.csv
```

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "int_rate": 24.9,
    "annual_inc": 38000,
    "dti": 35.2,
    "fico_range_low": 645,
    "delinq_2yrs": 3,
    "revol_util": 88.0
  }'

# Interactive docs
open http://localhost:8000/docs
```

---

## API Response Schema

```json
{
  "risk_score": 0.72,
  "risk_category": "High",
  "top_features": [
    {"feature": "int_rate",       "impact":  0.18},
    {"feature": "revol_util",     "impact":  0.15},
    {"feature": "dti",            "impact":  0.12},
    {"feature": "fico_range_low", "impact": -0.09},
    {"feature": "delinq_2yrs",    "impact":  0.08}
  ],
  "input_features": { ... },
  "model_version": "1.0.0"
}
```

**Risk Categories:**
| Category | Score Range |
|----------|-------------|
| Low      | 0.00 – 0.30 |
| Medium   | 0.30 – 0.55 |
| High     | 0.55 – 0.75 |
| Critical | 0.75 – 1.00 |

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| EDA | Class distribution, missing values, correlations, feature distributions |
| Preprocessing | Target encoding, temporal split, median imputation, SMOTE |
| Baseline | Logistic Regression (interpretable, regulatory-friendly) |
| Main Model | XGBoost (gradient boosted trees, handles imbalance via `scale_pos_weight`) |
| Evaluation | ROC-AUC, PR-AUC, F1, Brier score, confusion matrix |
| SHAP | Global summary + bar plots, local waterfall for highest-risk sample |
| Calibration | Platt scaling — makes probabilities statistically trustworthy |
| API | FastAPI with `/predict`, `/predict/batch`, `/model/info` |

---

## Temporal Split Logic

```
Train: issue_date < 2020-01-01
Test:  issue_date ≥ 2020-01-01
```

This mirrors real deployment. A model trained in 2019 should predict 2020+ defaults.
Random shuffling would inflate AUC by ~5–15 points.

---

## Outputs Generated

```
outputs/
  eda_overview.png              ← Class dist, missing values, correlations
  roc_pr_comparison.png         ← LR vs XGBoost ROC & PR curves
  confusion_logistic_*.png      ← Confusion matrices
  confusion_xgboost.png
  shap_summary.png              ← Beeswarm: global feature impact
  shap_importance.png           ← Bar: mean |SHAP| per feature
  shap_local_waterfall.png      ← Waterfall: highest-risk sample
  calibration_curve.png         ← Raw vs calibrated probabilities
```

---

## Phase 2 (Coming Next)
- Earnings call transcript ingestion
- Embedding pipeline (financial text → vectors)
- Tool-based agent (LLM + model as tool)
- News cross-verification layer
