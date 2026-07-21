"""Master orchestration pipeline for MLGenX Track B.

Coordinates the full multi-agent graph retrieval pipeline:
  1. Data ingestion (Kaggle + PerturbQA)
  2. Feature engineering (structural, identifier-agnostic)
  3. XGBoost surrogate training with GroupKFold zero-overlap validation
  4. Multi-agent inference loop (Planner -> Execution -> Synthesizer)
  5. Continuous probability extraction and submission generation
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from config.determinism import configure_determinism
from config.settings import N_FOLDS
from data.ingestion import (
    download_competition_data,
    fetch_perturbqa_metadata,
    setup_kaggle_credentials,
    KAGGLE_COMPETITION,
)
from agents.execution_agent import ExecutionAgent
from agents.planner import PlannerAgent
from agents.synthesizer import synthesize
from agents.prompts import FINAL_EVALUATION_PROMPT
from tools.feature_builder import build_features
from tools.surrogate_tool import train_surrogate, query_ml_surrogate, load_model

# Track B uses mouse genes (Mus musculus)
SPECIES_STRING = 10090
SPECIES_REACTOME = "MMU"


def build_training_matrix(
    train_df: pd.DataFrame,
    perturbqa_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix X and labels y from training + PerturbQA data.

    Labels: 0=up, 1=down, 2=no_change.
    Returns (X, y, groups) where groups are perturbation gene names for GroupKFold.
    """
    rows = []
    labels = []
    groups = []

    # Process competition training data
    for _, row in train_df.iterrows():
        gene_x = str(row.get("pert", row.get("perturbation_gene", row.get("gene_x", ""))))
        gene_y = str(row.get("gene", row.get("target_gene", row.get("gene_y", ""))))
        direction = str(row.get("label", row.get("direction", "none")))

        features = build_features(gene_x, gene_y, species=SPECIES_STRING)
        rows.append(list(features.values()))
        groups.append(gene_x)

        if direction == "up":
            labels.append(0)
        elif direction == "down":
            labels.append(1)
        else:
            labels.append(2)

    # Process PerturbQA data
    for _, row in perturbqa_df.iterrows():
        gene_x = str(row.get("perturbation", ""))
        gene_y = str(row.get("target", ""))
        if not gene_x or not gene_y:
            continue

        features = build_features(gene_x, gene_y, species=SPECIES_STRING)
        rows.append(list(features.values()))
        groups.append(gene_x)

        direction = row.get("direction", 0)
        if direction == 1 or direction == "up":
            labels.append(0)
        elif direction == -1 or direction == "down":
            labels.append(1)
        else:
            labels.append(2)

    return (
        np.array(rows, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(groups),
    )


def validate_zero_overlap(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """GroupKFold validation respecting zero-overlap gene split.

    Returns mean DE AUROC and DIR AUROC across folds.
    """
    from tools.surrogate_tool import _feature_names

    gkf = GroupKFold(n_splits=N_FOLDS)
    de_scores = []
    dir_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_surrogate(X_train, y_train)
        proba = model.predict_proba(X_val)

        # DE AUROC: P(up) + P(down)  vs  P(none) — binary classification
        de_true = (y_val != 2).astype(int)
        de_pred = 1.0 - proba[:, 2]
        try:
            de_auc = roc_auc_score(de_true, de_pred)
        except ValueError:
            de_auc = 0.5
        de_scores.append(de_auc)

        # DIR AUROC: P(up) / (P(up) + P(down)) for non-none cases
        mask = y_val != 2
        if mask.sum() > 1:
            dir_true = (y_val[mask] == 0).astype(int)
            dir_pred = proba[mask, 0] / (proba[mask, 0] + proba[mask, 1] + 1e-9)
            try:
                dir_auc = roc_auc_score(dir_true, dir_pred)
            except ValueError:
                dir_auc = 0.5
        else:
            dir_auc = 0.5
        dir_scores.append(dir_auc)

    return {
        "de_auroc_mean": np.mean(de_scores),
        "de_auroc_std": np.std(de_scores),
        "dir_auroc_mean": np.mean(dir_scores),
        "dir_auroc_std": np.std(dir_scores),
    }


def run_inference(
    test_df: pd.DataFrame,
    execution_agent: ExecutionAgent,
) -> List[Dict[str, float]]:
    """Run multi-agent inference for every test row.

    Returns list of {up, down, none} probability dicts.
    """
    predictions = []

    for idx, row in test_df.iterrows():
        gene_x = str(row.get("pert", row.get("perturbation_gene", row.get("gene_x", ""))))
        gene_y = str(row.get("gene", row.get("target_gene", row.get("gene_y", ""))))

        # Reset agent for each query
        planner = PlannerAgent(gene_x, gene_y)
        agent = ExecutionAgent()

        # Run the planning-execution loop
        while not planner.is_complete():
            batch = planner.plan_next()
            for call in batch:
                tool = call["tool"]
                params = call["params"]

                # Fill surrogate features from accumulated results if needed
                if tool == "query_ml_surrogate" and not params.get("features"):
                    features = build_features(gene_x, gene_y, species=SPECIES_STRING)
                    params["features"] = features

                agent.execute(tool, params)

        # Synthesize context
        context = synthesize(agent.get_all_results(), gene_x, gene_y)

        # Extract prediction from surrogate (primary signal)
        surrogate_result = None
        for r in agent.get_all_results():
            if r.get("tool") == "query_ml_surrogate" and "err" not in str(
                r.get("result", "")
            ):
                try:
                    surrogate_result = json.loads(r["result"])
                except Exception:
                    pass

        if surrogate_result:
            predictions.append({
                "up": surrogate_result.get("up", 0.33),
                "down": surrogate_result.get("dn", 0.33),
                "none": surrogate_result.get("nc", 0.34),
            })
        else:
            predictions.append({"up": 0.33, "down": 0.33, "none": 0.34})

    return predictions


def generate_submission(
    test_df: pd.DataFrame,
    predictions: List[Dict[str, float]],
    output_dir: str = "submission",
) -> str:
    """Generate submission.csv and metadata JSON artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # submission.csv
    rows = []
    for i, (_, test_row) in enumerate(test_df.iterrows()):
        pred = predictions[i]
        rows.append({
            "id": test_row.get("id", i),
            "prediction_up": pred["up"],
            "prediction_down": pred["down"],
            "prediction_no_change": pred["none"],
        })
    sub_df = pd.DataFrame(rows)
    sub_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)

    # metadata.json
    metadata = {
        "pipeline": "multi-agent-graph-retrieval",
        "track": "B",
        "model": "GPT-OSS-120B",
        "quantization": "MXFP4",
        "reasoning_effort": "high",
        "tools_used": [
            "query_string_db",
            "query_reactome",
            "query_go_semantics",
            "query_ml_surrogate",
        ],
        "surrogate": "XGBoost on PerturbQA structural features",
        "features": [
            "deg_x", "deg_y", "btw_x", "btw_y",
            "str_dist", "str_score", "reac_jac",
            "go_bp", "go_cc", "go_mf",
        ],
        "conflict_resolution": "hard-coded-matrix",
        "token_optimization": "json-minification-float-truncation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return sub_path


def main(kaggle_key: Optional[str] = None):
    """Execute the full MLGenX Track B pipeline."""
    configure_determinism(seed=42)

    # --- Data ingestion ---
    print("[1/5] Ingesting data...")
    if kaggle_key:
        setup_kaggle_credentials(kaggle_key)

    perturbqa = fetch_perturbqa_metadata()
    print(f"  PerturbQA records: {len(perturbqa)}")

    train_df, test_df = download_competition_data()
    print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # --- Feature engineering & training ---
    print("[2/5] Building training matrix...")
    X, y, groups = build_training_matrix(train_df, perturbqa)
    print(f"  Feature matrix: {X.shape}, labels: {y.shape}")

    print("[3/5] Validating with GroupKFold (zero-overlap)...")
    metrics = validate_zero_overlap(X, y, groups)
    print(f"  DE  AUROC: {metrics['de_auroc_mean']:.4f} +/- {metrics['de_auroc_std']:.4f}")
    print(f"  DIR AUROC: {metrics['dir_auroc_mean']:.4f} +/- {metrics['dir_auroc_std']:.4f}")

    # Train final model on all data
    print("[4/5] Training final XGBoost surrogate on full data...")
    train_surrogate(X, y)
    print("  Model saved.")

    # --- Inference ---
    print("[5/5] Running inference on test set...")
    agent = ExecutionAgent()
    predictions = run_inference(test_df, agent)

    # --- Submission ---
    sub_path = generate_submission(test_df, predictions)
    print(f"\nSubmission saved to: {sub_path}")
    print("Pipeline complete.")


if __name__ == "__main__":
    import sys
    kaggle_key = sys.argv[1] if len(sys.argv) > 1 else None
    main(kaggle_key)
