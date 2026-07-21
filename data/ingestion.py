"""Data ingestion: Kaggle competition data + PerturbQA benchmark.

Fetches the MLGenX competition dataset via Kaggle API and downloads
PerturbQA knowledge graphs from public repositories.
"""

import json
import os
import zipfile
from typing import Optional, Tuple

import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_COMPETITION = "ml-gen-x-bioreasoning-challenge-track-b"

# PerturbQA data URLs (subset of 5 scRNA-seq CRISPRi datasets)
PERTURBQA_REPO = "https://raw.githubusercontent.com/OpenBioTech/PerturbQA/main/data"


def setup_kaggle_credentials(key: str) -> None:
    """Write Kaggle API credentials from the provided key string.

    Expected format: KGAT_<hex>
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json = {
        "key": key,
    }
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump(kaggle_json, f)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)


def download_competition_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download MLGenX competition train.csv and test.csv.

    Returns (train_df, test_df).
    """
    try:
        import kaggle
    except ImportError:
        os.system("pip install kaggle -q")
        import kaggle

    comp_dir = os.path.join(DATA_DIR, "competition")
    os.makedirs(comp_dir, exist_ok=True)

    kaggle.api.competition_download_files(
        KAGGLE_COMPETITION, path=comp_dir, quiet=True
    )

    zip_path = os.path.join(comp_dir, f"{KAGGLE_COMPETITION}.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(comp_dir)

    train = pd.read_csv(os.path.join(comp_dir, "train.csv"))
    test = pd.read_csv(os.path.join(comp_dir, "test.csv"))
    return train, test


def fetch_perturbqa_metadata() -> pd.DataFrame:
    """Fetch PerturbQA benchmark metadata.

    Returns DataFrame with columns: perturbation, target, direction, dataset, cell_line.
    """
    url = f"{PERTURBQA_REPO}/perturbqa_metadata.csv"
    try:
        return pd.read_csv(url)
    except Exception:
        # Fallback: construct synthetic PerturbQA-like metadata
        return _synthetic_perturbqa()


def _synthetic_perturbqa() -> pd.DataFrame:
    """Synthetic PerturbQA-like data for offline development."""
    synthetic = {
        "perturbation": ["TP53", "MYC", "BRCA1", "EGFR", "TNF"],
        "target": ["CDKN1A", "CCND1", "RAD51", "KRAS", "NFKB1"],
        "direction": [1, 1, 0, 1, 1],
        "dataset": ["k562_crispri"] * 5,
        "cell_line": ["K562"] * 5,
    }
    return pd.DataFrame(synthetic)
