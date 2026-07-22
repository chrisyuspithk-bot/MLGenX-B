"""Probability calibration layer for the LLM pipeline.

Maps LLM outputs (qualitative direction + confidence) to well-calibrated
continuous probabilities that maximize AUROC. Supports:

1. Rule-based calibration (default): maps direction/confidence to fixed ranges
2. Temperature scaling: learns a temperature parameter from validation data
3. XGBoost residual: combines LLM prediction with XGBoost surrogate
"""

from typing import Optional

import numpy as np


class RuleBasedCalibrator:
    """Map qualitative predictions to calibrated probability ranges.

    The LLM outputs 'up'/'down'/'no_change' + 'high'/'medium'/'low' confidence.
    This calibrator maps those to continuous probability values.

    High confidence: narrow range near extremes
    Medium confidence: broader range, some probability mass on alternatives
    Low confidence: near-uniform, with slight bias toward predicted direction
    """

    # fmt: off
    RANGES = {
        ("up", "high"):        (0.92, 0.98,  0.01, 0.03),
        ("up", "medium"):      (0.65, 0.85,  0.05, 0.15),
        ("up", "low"):         (0.35, 0.50,  0.15, 0.30),
        ("down", "high"):      (0.01, 0.03,  0.92, 0.98),
        ("down", "medium"):    (0.05, 0.15,  0.65, 0.85),
        ("down", "low"):       (0.15, 0.30,  0.35, 0.50),
        ("no_change", "high"): (0.01, 0.03,  0.01, 0.03),
        ("no_change", "medium"):(0.05, 0.15, 0.03, 0.10),
        ("no_change", "low"):  (0.15, 0.30,  0.10, 0.25),
    }
    # fmt: on

    def __call__(self, direction: str, confidence: str, surrogate: Optional[dict] = None):
        up_lo, up_hi, down_lo, down_hi = self.RANGES.get(
            (direction, confidence), (0.20, 0.40, 0.10, 0.25)
        )
        p_up = (up_lo + up_hi) / 2
        p_down = (down_lo + down_hi) / 2

        # Blend with surrogate if available (soft fallback)
        if surrogate:
            s_up = surrogate.get("up", 0.3)
            s_down = surrogate.get("down", 0.14)
            # 80% LLM, 20% surrogate for confident predictions
            # 50/50 for low confidence
            w = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(confidence, 0.5)
            p_up = w * p_up + (1 - w) * s_up
            p_down = w * p_down + (1 - w) * s_down

        return float(np.clip(p_up, 0.0, 1.0)), float(np.clip(p_down, 0.0, 1.0))


class TemperatureCalibrator:
    """Learn temperature scaling from validation data.

    After training, apply temperature to logits to improve calibration.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, log_probs: np.ndarray, labels: np.ndarray):
        """Learn optimal temperature via grid search on NLL."""
        best_temp, best_nll = 1.0, float("inf")
        for t in np.linspace(0.5, 5.0, 50):
            scaled = log_probs / t
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean()
            if nll < best_nll:
                best_nll = nll
                best_temp = t
        self.temperature = best_temp
        return self

    def calibrate(self, log_probs: np.ndarray) -> np.ndarray:
        scaled = log_probs / self.temperature
        return np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
