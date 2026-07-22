"""DSPy optimizer for the perturbation prediction pipeline.

Uses BootstrapFewShot with metric-driven optimization against the training set.
The metric is (DE_AUROC + DIR_AUROC) / 2, matching the competition scoring.
"""

import dspy
import numpy as np
from sklearn.metrics import roc_auc_score


def competition_metric(y_true_labels, y_pred_proba_up, y_pred_proba_down):
    """Compute the competition metric: (micro_AUROC_DE + micro_AUROC_DIR) / 2.

    DE_AUROC: ability to distinguish (up or down) from no_change
    DIR_AUROC: ability to distinguish up from down among DE-positive rows

    Args:
        y_true_labels: array of 'up', 'down', 'none'
        y_pred_proba_up: array of P(up) floats
        y_pred_proba_down: array of P(down) floats
    """
    y_true_labels = np.array(y_true_labels)
    y_pred_up = np.array(y_pred_proba_up, dtype=float)
    y_pred_down = np.array(y_pred_proba_down, dtype=float)

    # DE: (up or down) vs none, scored by p_up + p_down
    de_true = (y_true_labels != "none").astype(int)
    de_pred = y_pred_up + y_pred_down
    de_auc = roc_auc_score(de_true, de_pred)

    # DIR: up vs down among DE-positive rows
    de_mask = y_true_labels != "none"
    if de_mask.sum() < 2 or (y_true_labels[de_mask] == "up").all() or (y_true_labels[de_mask] == "down").all():
        dir_auc = 0.5
    else:
        dir_true = (y_true_labels[de_mask] == "up").astype(int)
        dir_pred = y_pred_up[de_mask] / (y_pred_up[de_mask] + y_pred_down[de_mask] + 1e-9)
        dir_auc = roc_auc_score(dir_true, dir_pred)

    return (de_auc + dir_auc) / 2


def dspy_metric(example, pred, trace=None):
    """DSPy-compatible metric function."""
    score = competition_metric(
        [example.label],
        [pred.prediction_up],
        [pred.prediction_down],
    )
    return score


def dspy_batch_metric(examples, preds, trace=None):
    """DSPy-compatible batch metric for efficiency."""
    labels = [ex.label for ex in examples]
    ups = [float(p.prediction_up) for p in preds]
    downs = [float(p.prediction_down) for p in preds]
    return competition_metric(labels, ups, downs)


def optimize_pipeline(
    program: dspy.Module,
    trainset: list,
    metric=dspy_metric,
    num_threads: int = 4,
):
    """Optimize a DSPy program using BootstrapFewShot.

    Args:
        program: DSPy module to optimize
        trainset: list of dspy.Example with gene_X, gene_Y, label
        metric: DSPy metric function
        num_threads: parallel compile threads
    """
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        num_threads=num_threads,
    )
    optimized = optimizer.compile(program, trainset=trainset)
    return optimized


def optimize_with_mipro(
    program: dspy.Module,
    trainset: list,
    metric=dspy_batch_metric,
    num_threads: int = 4,
):
    """Optimize using MIPROv2 for better prompt optimization.

    MIPRO jointly optimizes instructions and few-shot examples.
    Use when BootstrapFewShot results plateau.
    """
    optimizer = dspy.MIPROv2(
        metric=metric,
        num_threads=num_threads,
        auto="light",
    )
    optimized = optimizer.compile(program, trainset=trainset)
    return optimized
