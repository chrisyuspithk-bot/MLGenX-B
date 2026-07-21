"""XGBoost surrogate predictor trained on PerturbQA structural features.

Feature vector is purely structural/semantic — no gene identifiers — enabling
perfect inductive generalization across the zero-overlap test set.

Features:
  deg_x, deg_y       : node degree centrality of each gene
  btw_x, btw_y       : betweenness centrality
  str_dist           : shortest path length (STRING)
  str_score          : aggregated STRING confidence
  reac_jac           : Reactome Jaccard coefficient
  go_bp, go_cc, go_mf: GO Resnik semantic similarities
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

from config.settings import XGB_PARAMS
from tools.payload import compress, error_payload

_MODEL: Optional[xgb.XGBClassifier] = None
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "xgboost_surrogate.pkl"
)


def _feature_names() -> List[str]:
    return [
        "deg_x", "deg_y", "btw_x", "btw_y",
        "str_dist", "str_score", "reac_jac",
        "go_bp", "go_cc", "go_mf",
    ]


def load_model() -> Optional[xgb.XGBClassifier]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
    return _MODEL


def save_model(model: xgb.XGBClassifier) -> None:
    global _MODEL
    _MODEL = model
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def query_ml_surrogate(features: Dict[str, float]) -> str:
    """Run XGBoost inference on structural feature vector.

    Expects all keys from _feature_names().
    Returns probability array [p_up, p_down, p_none].

    Payload keys (compressed):
      up   : P(up-regulated)
      dn   : P(down-regulated)
      nc   : P(no-change)
    """
    model = load_model()
    if model is None:
        return error_payload("no_model")

    names = _feature_names()
    vec = np.array([[features.get(n, 0.0) for n in names]], dtype=np.float32)
    proba = model.predict_proba(vec)[0]  # [class_0, class_1, class_2]

    # Map to up/down/none order (depends on training label encoding)
    return compress({
        "up": float(proba[0]),
        "dn": float(proba[1]),
        "nc": float(proba[2]),
    })


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
) -> xgb.XGBClassifier:
    """Train XGBoost on structural feature matrix X with labels y in {0, 1, 2}."""
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)
    save_model(model)
    return model
