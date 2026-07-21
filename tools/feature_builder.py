"""Structural + statistical feature vector builder for the XGBoost surrogate.

Constructs identifier-agnostic features from knowledge graph topology AND
training-data statistical priors, enabling zero-overlap generalization.

Fast path: training-data priors are instant. External APIs (mygene.info,
STRING partners) are loaded from pre-fetched JSON caches.
"""

import json
import os
from typing import Any, Dict, Optional

import networkx as nx
import pandas as pd

from tools.string_tool import _build_graph
from tools.reactome_tool import _pathways_for_gene
from tools.go_tool import query_go_semantics

# Global caches (lazy-filled)
_PERT_STATS: Optional[pd.DataFrame] = None
_GENE_STATS: Optional[pd.DataFrame] = None
_GLOBAL_PRIOR: Dict[str, float] = {}
_MY_GENE_CACHE: Dict[str, Dict[str, Any]] = {}
_STRING_PARTNERS_CACHE: Dict[str, list] = {}
_CACHES_LOADED = False


def _load_caches():
    """Load pre-fetched gene data from JSON caches."""
    global _MY_GENE_CACHE, _STRING_PARTNERS_CACHE, _CACHES_LOADED
    if _CACHES_LOADED:
        return
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "gene_cache")
    for fname, target in [
        ("mygene.json", _MY_GENE_CACHE),
        ("string_partners.json", _STRING_PARTNERS_CACHE),
    ]:
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    target.update(json.load(f))
            except Exception:
                pass
    _CACHES_LOADED = True


def _init_train_stats(train_df: pd.DataFrame):
    global _PERT_STATS, _GENE_STATS, _GLOBAL_PRIOR
    if _PERT_STATS is not None:
        return
    _PERT_STATS = train_df.groupby("pert")["label"].value_counts(normalize=True).unstack(fill_value=0)
    _GENE_STATS = train_df.groupby("gene")["label"].value_counts(normalize=True).unstack(fill_value=0)
    for col in ["down", "none", "up"]:
        if col not in _PERT_STATS.columns:
            _PERT_STATS[col] = 0.0
        if col not in _GENE_STATS.columns:
            _GENE_STATS[col] = 0.0
    _GLOBAL_PRIOR = {
        "up": float((train_df["label"] == "up").mean()),
        "down": float((train_df["label"] == "down").mean()),
        "none": float((train_df["label"] == "none").mean()),
    }
    _load_caches()


def build_features(
    gene_x: str, gene_y: str, species: int = 10090, train_df: pd.DataFrame = None,
) -> Dict[str, float]:
    """Build feature vector for a gene pair (zero-overlap safe, 26 features).

    Graph topology (10): deg_x/y, btw_x/y, str_dist, str_score, reac_jac, go_bp/cc/mf
    Training priors (9): pert_up/down/none, gene_up/down/none, interactions, prior_up/down_prod
    Gene annotations (7): mg_go_bp_x/y, mg_pathway_x/y, mg_summary_len_x/y, mg_pathway_overlap
    """
    if train_df is not None:
        _init_train_stats(train_df)

    features: Dict[str, float] = {
        "deg_x": 0.0, "deg_y": 0.0, "btw_x": 0.0, "btw_y": 0.0,
        "str_dist": -1.0, "str_score": 0.0, "reac_jac": 0.0,
        "go_bp": 0.0, "go_cc": 0.0, "go_mf": 0.0,
        "pert_up": _GLOBAL_PRIOR.get("up", 0.306),
        "pert_down": _GLOBAL_PRIOR.get("down", 0.141),
        "pert_none": _GLOBAL_PRIOR.get("none", 0.553),
        "gene_up": _GLOBAL_PRIOR.get("up", 0.306),
        "gene_down": _GLOBAL_PRIOR.get("down", 0.141),
        "gene_none": _GLOBAL_PRIOR.get("none", 0.553),
        "interactions": 0.0,
        "mg_go_bp_x": 0.0, "mg_go_bp_y": 0.0,
        "mg_pathway_x": 0.0, "mg_pathway_y": 0.0,
        "mg_summary_len_x": 0.0, "mg_summary_len_y": 0.0,
        "mg_pathway_overlap": 0.0,
        "prior_up_prod": 0.0, "prior_down_prod": 0.0,
    }

    # --- Training-data priors (instant, zero-overlap safe) ---
    if _PERT_STATS is not None:
        x_lower = gene_x.lower()
        pert_match = _PERT_STATS[_PERT_STATS.index.str.lower() == x_lower]
        if len(pert_match) > 0:
            features["pert_up"] = float(pert_match["up"].values[0])
            features["pert_down"] = float(pert_match["down"].values[0])
            features["pert_none"] = float(pert_match["none"].values[0])
        y_lower = gene_y.lower()
        gene_match = _GENE_STATS[_GENE_STATS.index.str.lower() == y_lower]
        if len(gene_match) > 0:
            features["gene_up"] = float(gene_match["up"].values[0])
            features["gene_down"] = float(gene_match["down"].values[0])
            features["gene_none"] = float(gene_match["none"].values[0])
        pair_mask = (train_df["pert"].str.lower() == x_lower) & (train_df["gene"].str.lower() == y_lower)
        features["interactions"] = float(pair_mask.sum())

    # --- STRING topology ---
    try:
        G = _build_graph([gene_x, gene_y], species)
        if gene_x in G:
            features["deg_x"] = float(G.degree(gene_x))
        if gene_y in G:
            features["deg_y"] = float(G.degree(gene_y))
        if G.number_of_nodes() >= 2:
            try:
                btw = nx.betweenness_centrality(G, normalized=True)
                features["btw_x"] = float(btw.get(gene_x, 0.0))
                features["btw_y"] = float(btw.get(gene_y, 0.0))
            except Exception:
                pass
        try:
            path = nx.dijkstra_path(G, gene_x, gene_y, weight="weight")
            length = len(path) - 1
            features["str_dist"] = float(length)
            if length > 0:
                scores = [G[path[i]][path[i + 1]].get("score", 0) for i in range(length)]
                features["str_score"] = round(sum(scores) / length, 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            features["str_dist"] = -1.0
    except Exception:
        pass

    # --- STRING interaction partners fallback (from cache) ---
    if features["deg_x"] == 0.0 and gene_x in _STRING_PARTNERS_CACHE:
        features["deg_x"] = float(len(_STRING_PARTNERS_CACHE.get(gene_x, [])))
    if features["deg_y"] == 0.0 and gene_y in _STRING_PARTNERS_CACHE:
        features["deg_y"] = float(len(_STRING_PARTNERS_CACHE.get(gene_y, [])))

    # --- Reactome Jaccard ---
    try:
        px = _pathways_for_gene(gene_x, species="MMU")
        py = _pathways_for_gene(gene_y, species="MMU")
        inter = px & py
        union = px | py
        features["reac_jac"] = round(len(inter) / len(union), 4) if union else 0.0
    except Exception:
        pass

    # --- GO semantics ---
    try:
        go_result = json.loads(query_go_semantics(gene_x, gene_y, species="mouse"))
        features["go_bp"] = float(go_result.get("bp", 0.0))
        features["go_cc"] = float(go_result.get("cc", 0.0))
        features["go_mf"] = float(go_result.get("mf", 0.0))
    except Exception:
        pass

    # --- mygene.info annotations (from pre-fetched cache) ---
    mg_x = _MY_GENE_CACHE.get(gene_x, {})
    mg_y = _MY_GENE_CACHE.get(gene_y, {})
    if mg_x:
        go_x = mg_x.get("go", {})
        features["mg_go_bp_x"] = float(len(go_x.get("BP", [])))
        kw_x = mg_x.get("pathway", {}).get("kegg", [])
        features["mg_pathway_x"] = float(len(kw_x))
        features["mg_summary_len_x"] = float(len(mg_x.get("summary", "") or ""))
    if mg_y:
        go_y = mg_y.get("go", {})
        features["mg_go_bp_y"] = float(len(go_y.get("BP", [])))
        kw_y = mg_y.get("pathway", {}).get("kegg", [])
        features["mg_pathway_y"] = float(len(kw_y))
        features["mg_summary_len_y"] = float(len(mg_y.get("summary", "") or ""))
        if mg_x:
            px_names = {p.get("name", "") for p in (kw_x if isinstance(kw_x, list) else [])}
            py_names = {p.get("name", "") for p in (kw_y if isinstance(kw_y, list) else [])}
            features["mg_pathway_overlap"] = float(len(px_names & py_names))

    # --- Derived features ---
    features["prior_up_prod"] = features["pert_up"] * features["gene_up"]
    features["prior_down_prod"] = features["pert_down"] * features["gene_down"]

    return features


FEATURE_NAMES = [
    "deg_x", "deg_y", "btw_x", "btw_y", "str_dist", "str_score", "reac_jac",
    "go_bp", "go_cc", "go_mf",
    "pert_up", "pert_down", "pert_none",
    "gene_up", "gene_down", "gene_none", "interactions",
    "mg_go_bp_x", "mg_go_bp_y", "mg_pathway_x", "mg_pathway_y",
    "mg_summary_len_x", "mg_summary_len_y", "mg_pathway_overlap",
    "prior_up_prod", "prior_down_prod",
]
