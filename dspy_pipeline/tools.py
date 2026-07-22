"""DSPy tool wrappers for the Track B agent pipeline.

All tools operate on gene symbols and return identifier-agnostic, token-compressed JSON.
Each tool is decorated with DSPy's @tool for use in ReAct/ChainOfThought modules.
"""

import json
import sys
import os
from typing import Optional

import dspy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.string_tool import query_string_db as _string
from tools.reactome_tool import query_reactome as _reactome
from tools.go_tool import query_go_semantics as _go

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_mg_cache: Optional[dict] = None
_train_cache: Optional[pd.DataFrame] = None
_gene_profiles: dict = {}
_knn_profile: Optional[tuple] = None


def init_tools(train_df: pd.DataFrame, mg_cache: dict):
    """Pre-load training data and gene annotations for fast tool responses."""
    global _train_cache, _mg_cache, _gene_profiles, _knn_profile

    _train_cache = train_df
    _mg_cache = mg_cache

    # Build gene-level perturbation profiles for lookup_similar
    pert_stats = train_df.groupby("pert")["label"].value_counts(
        normalize=True
    ).unstack(fill_value=0)
    gene_stats = train_df.groupby("gene")["label"].value_counts(
        normalize=True
    ).unstack(fill_value=0)
    for c in ["down", "none", "up"]:
        if c not in pert_stats.columns:
            pert_stats[c] = 0.0
        if c not in gene_stats.columns:
            gene_stats[c] = 0.0

    # Gene property features for KNN
    def _gene_vector(g):
        e = mg_cache.get(g, {})
        go = e.get("go", {})
        bp_n = len([x for x in (go.get("BP", []) or []) if isinstance(x, dict)])
        cc_n = len([x for x in (go.get("CC", []) or []) if isinstance(x, dict)])
        mf_n = len([x for x in (go.get("MF", []) or []) if isinstance(x, dict)])
        kw = e.get("pathway", {}).get("kegg", []) or []
        kw_n = len([x for x in kw if isinstance(x, dict)])
        summary = e.get("summary", "") or ""
        name = e.get("name", "") or ""
        gt = e.get("type_of_gene", "unknown") or "unknown"
        return [
            bp_n,
            cc_n,
            mf_n,
            kw_n,
            len(summary),
            len(name),
            1.0 if gt == "protein-coding" else 0.0,
        ]

    all_genes = sorted(
        set(train_df["pert"]) | set(train_df["gene"]) | set(mg_cache.keys())
    )
    train_genes = sorted(set(train_df["pert"]) | set(train_df["gene"]))
    X_gene = np.array([_gene_vector(g) for g in train_genes], dtype=np.float32)

    y_profiles = np.zeros((len(train_genes), 6), dtype=np.float32)
    for i, g in enumerate(train_genes):
        if g in pert_stats.index:
            y_profiles[i, :3] = pert_stats.loc[g, ["up", "down", "none"]].values.astype(
                np.float32
            )
        else:
            y_profiles[i, :3] = [0.306, 0.141, 0.553]
        if g in gene_stats.index:
            y_profiles[i, 3:] = gene_stats.loc[g, ["up", "down", "none"]].values.astype(
                np.float32
            )
        else:
            y_profiles[i, 3:] = [0.306, 0.141, 0.553]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_gene)
    nn = NearestNeighbors(n_neighbors=min(30, len(train_genes)), metric="cosine")
    nn.fit(X_scaled)

    _gene_profiles = {"genes": train_genes, "X": X_scaled, "y": y_profiles}
    _knn_profile = (scaler, nn, y_profiles, train_genes)


# ---------------------------------------------------------------------------
# Tool 1: Gene characterization from mygene.info
# ---------------------------------------------------------------------------


@dspy.Tool
def gene_info(gene_symbol: str) -> str:
    """Return detailed functional annotation for a gene.

    Returns GO terms, KEGG pathways, gene type, and summary.
    Use this for BOTH the perturbation gene X and target gene Y.
    """
    if _mg_cache is None:
        return json.dumps({"err": "cache_not_initialized"})

    entry = _mg_cache.get(gene_symbol, {})
    go = entry.get("go", {})
    bp_terms = []
    for bp in (go.get("BP", []) or [])[:50]:
        if isinstance(bp, dict):
            bp_terms.append(bp.get("term", bp.get("id", "")))

    kw = entry.get("pathway", {}).get("kegg", []) or []
    pathways = []
    for p in kw[:20]:
        if isinstance(p, dict):
            pathways.append(p.get("name", ""))

    result = {
        "symbol": gene_symbol,
        "name": entry.get("name", ""),
        "type": entry.get("type_of_gene", ""),
        "summary": (entry.get("summary", "") or "")[:300],
        "go_bp": bp_terms[:10],
        "go_bp_n": len(bp_terms),
        "pathways": pathways[:10],
        "pathway_n": len(pathways),
    }
    return json.dumps(result, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Tool 2: STRING protein-protein interaction
# ---------------------------------------------------------------------------


@dspy.Tool
def protein_interactions(gene_symbol: str) -> str:
    """Return STRING interaction partners for a gene.

    Returns top interaction partners with confidence scores (0-1000).
    Use to check if X and Y physically interact or share interaction partners.
    """
    try:
        result = json.loads(_string(gene_symbol, species=10090))
        return json.dumps(result, separators=(",", ":"))
    except Exception:
        return json.dumps({"err": "string_timeout"})


# ---------------------------------------------------------------------------
# Tool 3: Reactome pathway overlap
# ---------------------------------------------------------------------------


@dspy.Tool
def pathway_overlap(gene_x: str, gene_y: str) -> str:
    """Return Reactome pathway Jaccard similarity between two genes.

    1.0 = identical pathways, 0.0 = no shared pathways.
    High overlap suggests co-regulation potential.
    """
    try:
        result = json.loads(_reactome(gene_x, gene_y))
        return json.dumps(result, separators=(",", ":"))
    except Exception:
        return json.dumps({"err": "reactome_timeout"})


# ---------------------------------------------------------------------------
# Tool 4: GO semantic similarity
# ---------------------------------------------------------------------------


@dspy.Tool
def go_similarity(gene_x: str, gene_y: str) -> str:
    """Return Resnik semantic similarity for GO BP, CC, MF.

    Higher values = more similar functions. CC < 0.1 means different
    cellular compartments → unlikely direct interaction.
    """
    try:
        return _go(gene_x, gene_y, species="mouse")
    except Exception:
        return json.dumps({"err": "go_timeout"})


# ---------------------------------------------------------------------------
# Tool 5: XGBoost surrogate prediction
# ---------------------------------------------------------------------------

_surrogate_model = None
_surrogate_features = None


def init_surrogate(model, feature_names: list):
    global _surrogate_model, _surrogate_features
    _surrogate_model = model
    _surrogate_features = feature_names


@dspy.Tool
def ml_surrogate(gene_x: str, gene_y: str) -> str:
    """Return XGBoost surrogate prediction for this gene pair.

    The surrogate was trained on 7,705 perturbation experiments using
    identifier-agnostic features (GO overlap, pathway similarity, etc.).
    Returns calibrated probabilities for up/down/none.

    Trust this when: direct binding evidence is weak, long-range cascade,
    or when other tools give contradictory signals.
    """
    if _surrogate_model is None:
        return json.dumps({"err": "surrogate_not_initialized"})

    from tools.feature_builder import build_features

    feats = build_features(gene_x, gene_y, species=10090)
    X = np.array([[feats.get(f, 0.0) for f in _surrogate_features]], dtype=np.float32)
    proba = _surrogate_model.predict_proba(X)[0]
    return json.dumps(
        {"up": round(float(proba[0]), 3), "down": round(float(proba[1]), 3)},
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Tool 6: Similar perturbation lookup (anti-leakage KNN)
# ---------------------------------------------------------------------------


@dspy.Tool
def lookup_similar_perturbations(gene_symbol: str) -> str:
    """Find genes with similar properties in the training set and report
    their perturbation-effect profiles.

    Uses KNN in gene-property space (GO counts, pathway counts, gene type).
    Returns: what fraction of similar genes' targets went up/down/none.
    """
    if _knn_profile is None:
        return json.dumps({"err": "not_initialized"})

    scaler, nn, y_profiles, train_genes = _knn_profile

    # Build query vector
    e = _mg_cache.get(gene_symbol, {}) if _mg_cache else {}
    go = e.get("go", {})
    bp_n = len([x for x in (go.get("BP", []) or []) if isinstance(x, dict)])
    cc_n = len([x for x in (go.get("CC", []) or []) if isinstance(x, dict)])
    mf_n = len([x for x in (go.get("MF", []) or []) if isinstance(x, dict)])
    kw = e.get("pathway", {}).get("kegg", []) or []
    kw_n = len([x for x in kw if isinstance(x, dict)])
    summary = e.get("summary", "") or ""
    name = e.get("name", "") or ""
    gt = e.get("type_of_gene", "unknown") or "unknown"
    query = np.array(
        [
            [
                bp_n,
                cc_n,
                mf_n,
                kw_n,
                len(summary),
                len(name),
                1.0 if gt == "protein-coding" else 0.0,
            ]
        ],
        dtype=np.float32,
    )

    query_scaled = scaler.transform(query)
    _, idxs = nn.kneighbors(query_scaled)
    profiles = y_profiles[idxs[0]]
    mean_profile = profiles.mean(axis=0)

    # Get names of 3 most similar genes
    similar = [train_genes[i] for i in idxs[0][:3]]

    return json.dumps(
        {
            "similar": similar,
            "pert_up": round(float(mean_profile[0]), 3),
            "pert_down": round(float(mean_profile[1]), 3),
            "gene_up": round(float(mean_profile[3]), 3),
            "gene_down": round(float(mean_profile[4]), 3),
        },
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Tool 7: Training data lookup (anti-leakage for test genes)
# ---------------------------------------------------------------------------


@dspy.Tool
def lookup_training(gene_x: str, gene_y: str) -> str:
    """Check if this exact gene pair appears in the training data.

    Returns label distribution if found, otherwise reports no match.
    Training genes are disjoint from test genes, so matches are rare.
    """
    if _train_cache is None:
        return json.dumps({"found": 0})

    mask = (_train_cache["pert"].str.lower() == gene_x.lower()) & (
        _train_cache["gene"].str.lower() == gene_y.lower()
    )
    n = mask.sum()
    if n == 0:
        return json.dumps({"found": 0, "note": "zero_overlap_expected"})

    labels = _train_cache.loc[mask, "label"].value_counts(normalize=True).to_dict()
    return json.dumps(
        {"found": int(n), "up": round(labels.get("up", 0), 3), "down": round(labels.get("down", 0), 3)},
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Tool list for DSPy
# ---------------------------------------------------------------------------

TOOLS = [
    gene_info,
    protein_interactions,
    pathway_overlap,
    go_similarity,
    ml_surrogate,
    lookup_similar_perturbations,
    lookup_training,
]
