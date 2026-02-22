"""
shared.py — classes that must be importable from both pipeline.py and api/main.py.
Pickle serializes the full module path of every class. If CalibratedXGB is defined
in __main__ (pipeline.py run directly), the API cannot deserialize it.
Defining it here means pickle stores "shared.CalibratedXGB" which resolves anywhere.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibratedXGB:
    """
    Manual Platt scaling wrapper for XGBoost.
    Bypasses sklearn CalibratedClassifierCV float32/float64 dtype clash.
    """
    def __init__(self, xgb_model):
        self.xgb_model  = xgb_model
        self.calibrator = LogisticRegression(C=1e10)
        self.classes_   = np.array([0, 1])

    def fit(self, X, y):
        raw = self.xgb_model.predict_proba(X)[:, 1].astype(np.float64).reshape(-1, 1)
        self.calibrator.fit(raw, y)
        return self

    def predict_proba(self, X):
        raw = self.xgb_model.predict_proba(X)[:, 1].astype(np.float64).reshape(-1, 1)
        return self.calibrator.predict_proba(raw)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
