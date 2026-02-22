# Financial Risk Model
LendingClub credit default prediction — end-to-end ML pipeline with REST API and React dashboard.

---

## Project Structure

```
Financial Risk Model/
├── pipeline.py              # Full ML pipeline
├── shared.py                # CalibratedXGB class (shared between pipeline and API)
├── lending_club_eda.ipynb   # Exploratory data analysis notebook
├── requirements.txt
├── api/
│   └── main.py              # FastAPI prediction service
├── dashboard/               # React frontend (Vite)
│   └── src/
│       └── App.jsx
├── models/                  # Saved model artifacts (generated after running pipeline)
│   ├── xgb_calibrated.joblib
│   ├── xgb_raw.joblib
│   ├── logistic_regression.joblib
│   └── model_meta.json
└── outputs/                 # Plots and evaluation charts (generated after running pipeline)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the LendingClub dataset from Kaggle:
[wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

File needed: `accepted_2007_to_2018q4.csv`

---

## Usage

### 1. Run EDA
Open `lending_club_eda.ipynb` in Jupyter or Kaggle. Update the file path at the top if running locally.

### 2. Train the model
```bash
python pipeline.py --data path/to/accepted_2007_to_2018q4.csv
```

On first run this takes ~20 minutes. Subsequent runs load from checkpoint instantly unless the feature set changes.

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Start the dashboard
```bash
cd dashboard
npm install   # first time only
npm run dev
```

Open `http://localhost:5173`

---

## Pipeline Steps

| Step | Function | Description |
|---|---|---|
| 1 | `load_data` | Load CSV, skip LendingClub description row |
| 2 | `preprocess` | Encode target, ordinal/one-hot encode categoricals, impute NaN |
| 3 | `temporal_split` | Train on pre-2016 loans, test on post-2016 |
| 4 | `train_*` | Logistic Regression + XGBoost with checkpointing |
| 5 | `evaluate_model` | ROC-AUC, PR-AUC, Brier score, F1 |
| 6 | `run_shap` | Global beeswarm, feature importance bar, local waterfall |
| 7 | `calibrate_model` | Platt scaling via custom CalibratedXGB wrapper |
| 8 | `tune_threshold` | Find optimal F1 threshold from PR curve |
| 9 | `save_artifacts` | Persist models and metadata for API |

---

## Features Used

**Numeric (15)**
`loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `delinq_2yrs`, `fico_range_low`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `mort_acc`, `pub_rec_bankruptcies`

**Categorical (4, encoded)**
`grade` (ordinal A=0→G=6), `emp_length` (ordinal), `home_ownership` (one-hot), `purpose` (one-hot)

---

## API

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Score a single loan application |
| `/predict/batch` | POST | Score multiple applications |
| `/predict/example` | GET | Run a sample high-risk borrower |
| `/health` | GET | Check model loaded status |
| `/model/info` | GET | Feature list, metrics, thresholds |
| `/docs` | GET | Interactive Swagger UI |

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "int_rate": 18.5,
    "installment": 475,
    "annual_inc": 52000,
    "dti": 28.4,
    "fico_range_low": 660,
    "delinq_2yrs": 2,
    "inq_last_6mths": 3,
    "open_acc": 8,
    "pub_rec": 1,
    "revol_bal": 12000,
    "revol_util": 78.0,
    "total_acc": 15,
    "mort_acc": 0,
    "pub_rec_bankruptcies": 0
  }'
```

### Example response
```json
{
  "risk_score": 0.72,
  "risk_category": "High",
  "top_features": [
    {"feature": "int_rate",       "impact":  0.18},
    {"feature": "revol_util",     "impact":  0.15},
    {"feature": "grade",          "impact":  0.12},
    {"feature": "dti",            "impact":  0.09},
    {"feature": "fico_range_low", "impact": -0.07}
  ],
  "input_features": {"...": "..."},
  "api_version": "1.0.0"
}
```

### Risk categories
| Category | Probability Range |
|---|---|
| Low | 0.00 – 0.30 |
| Medium | 0.30 – 0.55 |
| High | 0.55 – 0.75 |
| Critical | 0.75 – 1.00 |

---

## Model Performance

Evaluated on temporal out-of-sample test set (post-2016 loans).

| Model | ROC-AUC | PR-AUC | Brier | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.684 | 0.149 | 0.233 | 0.208 |
| XGBoost | 0.697 | 0.158 | 0.222 | 0.213 |
| XGBoost (calibrated) | 0.697 | 0.158 | 0.081 | — |

> Brier score drop from 0.222 → 0.081 after Platt scaling indicates well-calibrated probabilities.
> F1 for the calibrated model depends on the optimal threshold found during tuning, stored in `model_meta.json`.

---

## Key Design Decisions

**Temporal split over random shuffle**
Train on pre-2016 loans, test on post-2016. Prevents data leakage across time and mirrors real deployment conditions. Random shuffle would inflate AUC by 5–15 points.

**Outcome truncation**
Test set default rate (~7.75%) is lower than train (~17%). Loans issued in 2017–2018 are 36-month loans that hadn't matured when the dataset was collected — many future defaults still appear as "Current". This is a known artifact of the dataset, not a bug.

**CalibratedXGB in shared.py**
Python's pickle serializes the full module path of every class. Defining `CalibratedXGB` in a shared module ensures the path is stable and resolvable both during training (`pipeline.py`) and at API load time (`api/main.py`).

**Threshold tuning**
Default 0.5 threshold is arbitrary and wrong for imbalanced data. We sweep the PR curve and pick the threshold that maximises F1 on the test set. Saved to `model_meta.json` and loaded by the API automatically.
