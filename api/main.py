"""
Financial Risk Model — FastAPI Microservice
Serves calibrated XGBoost predictions with SHAP explanations.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /model/info
    POST /predict
    POST /predict/batch
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# CalibratedXGB lives in shared.py — pickle needs a stable importable path
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from shared import CalibratedXGB  # noqa: F401 — needed for joblib deserialization


# ── Inline helpers (no dependency on pipeline.py location) ──────────────────
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

def extract_top_shap_features(shap_values, feature_names, sample_idx=0, top_n=5):
    sv = shap_values[sample_idx]
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs[:top_n]]

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Resolve models/ relative to project root (one level up from api/)
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
MODEL_DIR  = os.getenv("MODEL_DIR", os.path.join(_ROOT, "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_calibrated.joblib")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")
RAW_PATH   = os.path.join(MODEL_DIR, "xgb_raw.joblib")

# ─────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────
class AppState:
    model          = None
    raw_model      = None
    meta           = None
    shap_explainer = None
    feature_names  = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    print("[API] Loading model artifacts...")
    try:
        state.model     = joblib.load(MODEL_PATH)
        state.raw_model = joblib.load(RAW_PATH)
        state.meta      = json.load(open(META_PATH))
        state.feature_names = state.meta["feature_names"]

        # SHAP explainer on raw XGBoost (TreeExplainer only works on tree models)
        state.shap_explainer = shap.TreeExplainer(state.raw_model)
        print(f"[API] ✓ Loaded model | {len(state.feature_names)} features")
    except FileNotFoundError as e:
        print(f"[API] ⚠️  Models not found ({e}). Run pipeline.py first.")
        # Allow app to start without models for health checks
    yield
    print("[API] Shutdown.")


# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = FastAPI(
    title        = "Financial Risk Model API",
    description  = "Credit default probability prediction with SHAP explainability",
    version      = "1.0.0",
    lifespan     = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class LoanFeatures(BaseModel):
    """
    Input schema — all LendingClub numeric features.
    All fields optional with sensible defaults so you can hit the API
    with partial data during development.
    """
    loan_amnt:              float = Field(10000,  description="Loan amount requested ($)")
    int_rate:               float = Field(12.5,   description="Interest rate (%)")
    installment:            float = Field(350,    description="Monthly installment ($)")
    annual_inc:             float = Field(65000,  description="Annual income ($)")
    dti:                    float = Field(18.0,   description="Debt-to-income ratio")
    delinq_2yrs:            float = Field(0,      description="Delinquencies last 2 years")
    fico_range_low:         float = Field(700,    description="FICO score (low end)")
    inq_last_6mths:         float = Field(1,      description="Hard credit inquiries last 6 months")
    open_acc:               float = Field(10,     description="Open credit accounts")
    pub_rec:                float = Field(0,      description="Public derogatory records")
    revol_bal:              float = Field(8000,   description="Revolving credit balance ($)")
    revol_util:             float = Field(40.0,   description="Revolving utilization rate (%)")
    total_acc:              float = Field(20,     description="Total credit accounts")
    mort_acc:               float = Field(1,      description="Mortgage accounts")
    pub_rec_bankruptcies:   float = Field(0,      description="Bankruptcies on record")

    @validator("dti")
    def dti_range(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("DTI must be between 0 and 100")
        return v

    @validator("revol_util")
    def util_range(cls, v):
        if not (0 <= v <= 120):
            raise ValueError("revol_util should be between 0 and 120")
        return v

    model_config = {"json_schema_extra": {"example": {
        "loan_amnt":            15000,
        "int_rate":             18.5,
        "installment":          475,
        "annual_inc":           52000,
        "dti":                  28.4,
        "delinq_2yrs":          2,
        "fico_range_low":       660,
        "inq_last_6mths":       3,
        "open_acc":             8,
        "pub_rec":              1,
        "revol_bal":            12000,
        "revol_util":           78.0,
        "total_acc":            15,
        "mort_acc":             0,
        "pub_rec_bankruptcies": 0,
    }}}


class FeatureImpact(BaseModel):
    feature: str
    impact:  float


class PredictResponse(BaseModel):
    risk_score:     float = Field(..., description="Calibrated P(default) — 0 to 1")
    risk_category:  str   = Field(..., description="Low / Medium / High / Critical")
    top_features:   list[FeatureImpact]
    input_features: dict  = Field(..., description="Echo of input values used")
    api_version:    str   = "1.0.0"


class BatchPredictRequest(BaseModel):
    loans: list[LoanFeatures]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count:       int


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def features_to_array(features: LoanFeatures) -> np.ndarray:
    """Convert Pydantic model → ordered numpy array matching training features."""
    if state.feature_names is None:
        raise HTTPException(503, "Model not loaded. Run pipeline.py first.")

    row = features.dict()
    # Build array in same column order as training
    arr = np.array([[row.get(f, 0.0) for f in state.feature_names]], dtype=float)
    return arr


def predict_single(features: LoanFeatures) -> PredictResponse:
    arr = features_to_array(features)

    # Calibrated probability
    score = float(state.model.predict_proba(arr)[0, 1])

    # SHAP on raw XGBoost
    shap_vals = state.shap_explainer.shap_values(arr)[0]
    top_feats = [
        FeatureImpact(feature=f, impact=round(float(v), 4))
        for f, v in sorted(
            zip(state.feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
    ]

    return PredictResponse(
        risk_score    = round(score, 4),
        risk_category = score_to_category(score),
        top_features  = top_feats,
        input_features= features.dict(),
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {
        "status":       "healthy",
        "model_loaded": state.model is not None,
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Return model metadata, feature list, and performance metrics."""
    if state.meta is None:
        raise HTTPException(503, "Model metadata not loaded.")
    return state.meta


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(features: LoanFeatures):
    """
    Predict default probability for a single loan application.

    Returns:
    - **risk_score**: calibrated P(default)
    - **risk_category**: Low / Medium / High / Critical
    - **top_features**: top 5 SHAP drivers (positive = increases risk)
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded. Run pipeline.py first.")
    return predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    Predict default probability for a batch of loan applications.
    Max 100 loans per request.
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded. Run pipeline.py first.")
    if len(request.loans) > 100:
        raise HTTPException(400, "Batch limit is 100 loans per request.")

    preds = [predict_single(loan) for loan in request.loans]
    return BatchPredictResponse(predictions=preds, count=len(preds))


@app.get("/predict/example", response_model=PredictResponse, tags=["Prediction"])
def predict_example():
    """Run prediction on a built-in high-risk example."""
    example = LoanFeatures(
        loan_amnt=15000,
        int_rate=24.9,
        installment=520,
        annual_inc=38000,
        dti=35.2,
        delinq_2yrs=3,
        fico_range_low=645,
        inq_last_6mths=5,
        open_acc=6,
        pub_rec=1,
        revol_bal=18000,
        revol_util=88.0,
        total_acc=12,
        mort_acc=0,
        pub_rec_bankruptcies=1,
    )
    if state.model is None:
        # Return mock response for demo
        return PredictResponse(
            risk_score    = 0.73,
            risk_category = "High",
            top_features  = [
                FeatureImpact(feature="int_rate",       impact=0.18),
                FeatureImpact(feature="revol_util",     impact=0.15),
                FeatureImpact(feature="dti",            impact=0.12),
                FeatureImpact(feature="fico_range_low", impact=-0.09),
                FeatureImpact(feature="delinq_2yrs",    impact=0.08),
            ],
            input_features= example.dict(),
        )
    return predict_single(example)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
