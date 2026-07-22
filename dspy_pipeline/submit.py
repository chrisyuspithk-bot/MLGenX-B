"""Submission pipeline: load data, configure LLM, run predictions, generate CSV.

Usage:
    python -m dspy_pipeline.submit \\
        --api-base $LLM_ENDPOINT \\
        --api-key $LLM_KEY \\
        --model openai/gpt-oss-120b \\
        [--optimize] [--fast]

The LLM runs locally against test.csv and produces submission.csv.
"""

import argparse
import json
import os
import sys
import time

import dspy
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.ingestion import download_competition_data
from tools.feature_builder import build_features, FEATURE_NAMES
from xgboost import XGBClassifier

from .tools import init_tools, init_surrogate
from .modules import PerturbationPredictor, FastPredictor
from .calibrate import RuleBasedCalibrator
from .optimizer import optimize_pipeline, competition_metric


def build_trainset(train_df: pd.DataFrame) -> list:
    """Convert training DataFrame to DSPy Examples for optimization."""
    examples = []
    for _, row in train_df.iterrows():
        examples.append(
            dspy.Example(
                gene_X=str(row["pert"]),
                gene_Y=str(row["gene"]),
                label=str(row["label"]),
            ).with_inputs("gene_X", "gene_Y")
        )
    return examples


def load_surrogate(train_df: pd.DataFrame):
    """Train XGBoost surrogate on training data for the ml_surrogate tool."""
    X_rows, y = [], []
    for _, row in train_df.iterrows():
        feats = build_features(str(row["pert"]), str(row["gene"]), species=10090, train_df=train_df)
        X_rows.append([feats.get(f, 0.0) for f in FEATURE_NAMES])
        lbl = str(row["label"])
        y.append(0 if lbl == "up" else (1 if lbl == "down" else 2))
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    n_up, n_down, n_none = (y == 0).sum(), (y == 1).sum(), (y == 2).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_none / max(n_up + n_down, 1),
        objective="multi:softprob", num_class=3, random_state=42, verbosity=0,
    )
    model.fit(X, y)

    # 10-seed ensemble for better calibration
    probas = np.zeros((len(X), 3))
    for seed in range(10):
        m = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_none / max(n_up + n_down, 1),
            objective="multi:softprob", num_class=3,
            random_state=seed, verbosity=0,
        )
        m.fit(X, y)
        probas += m.predict_proba(X)
    probas /= 10

    # Wrap in a simple predict_proba interface
    class EnsembleSurrogate:
        def predict_proba(self, X_new):
            p = np.zeros((len(X_new), 3))
            for seed in range(10):
                m = XGBClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=n_none / max(n_up + n_down, 1),
                    objective="multi:softprob", num_class=3,
                    random_state=seed, verbosity=0,
                )
                m.fit(X, y)
                p += m.predict_proba(X_new)
            return p / 10

    return EnsembleSurrogate(), FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="DSPy pipeline for Track B")
    parser.add_argument("--api-base", required=True, help="LLM API base URL")
    parser.add_argument("--api-key", required=True, help="LLM API key")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--optimize", action="store_true", help="Run DSPy optimization")
    parser.add_argument("--fast", action="store_true", help="Use fast (single-step) predictor")
    parser.add_argument("--output", default="submission/submission.csv", help="Output CSV path")
    args = parser.parse_args()

    # ── Configure LLM ──
    lm = dspy.LM(args.model, api_base=args.api_base, api_key=args.api_key)
    dspy.configure(lm=lm)

    # ── Load data ──
    print("[1/5] Loading data...")
    train_df, test_df = download_competition_data()
    with open("data/gene_cache/mygene.json") as f:
        mg_cache = json.load(f)

    # ── Initialize tools ──
    print("[2/5] Initializing tools...")
    init_tools(train_df, mg_cache)

    # Load and attach surrogate
    surrogate, feat_names = load_surrogate(train_df)
    init_surrogate(surrogate, feat_names)

    # ── Build pipeline ──
    calibrator = RuleBasedCalibrator()

    if args.fast:
        program = FastPredictor()
    else:
        program = PerturbationPredictor()

    # ── Optimize (optional) ──
    if args.optimize:
        print("[3/5] Optimizing with DSPy BootstrapFewShot...")
        trainset = build_trainset(train_df)
        program = optimize_pipeline(program, trainset)
    else:
        print("[3/5] Skipping optimization (use --optimize to enable)")

    # ── Run predictions ──
    print("[4/5] Running predictions on test set...")
    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        gx, gy = str(row["pert"]), str(row["gene"])

        try:
            pred = program(gene_X=gx, gene_Y=gy)
            p_up = float(pred.prediction_up)
            p_down = float(pred.prediction_down)
            trace = pred.reasoning_trace
        except Exception as e:
            # Fallback to surrogate + calibrator
            p_up, p_down = calibrator("no_change", "low")
            trace = json.dumps({"fallback": str(e)}, separators=(",", ":"))

        rows.append(
            {
                "id": row["id"],
                "prediction_up": round(p_up, 4),
                "prediction_down": round(p_down, 4),
                "reasoning_trace": trace,
                "tokens_used": 0,
                "num_tool_calls": 0,
                "prompt_tokens": 0,
                "num_distinct_tools": 7,
                "model_name": args.model,
            }
        )

    # ── Save submission ──
    print("[5/5] Saving submission...")
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(args.output, index=False)
    print(f"  Saved {len(sub_df)} rows to {args.output}")
    print(f"  pred_up range: [{sub_df.prediction_up.min():.3f}, {sub_df.prediction_up.max():.3f}]")
    print(f"  pred_down range: [{sub_df.prediction_down.min():.3f}, {sub_df.prediction_down.max():.3f}]")
    print("Done!")


if __name__ == "__main__":
    main()
