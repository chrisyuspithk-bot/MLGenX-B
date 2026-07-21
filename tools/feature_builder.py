"""Structural feature vector builder for the XGBoost surrogate.

Constructs identifier-agnostic features from knowledge graph topology,
enabling zero-overlap generalization across disjoint gene sets.
"""

import json
from typing import Any, Dict

import networkx as nx

from tools.string_tool import _build_graph, _fetch_interactions
from tools.reactome_tool import _pathways_for_gene
from tools.go_tool import query_go_semantics


def build_features(gene_x: str, gene_y: str, species: int = 10090) -> Dict[str, float]:
    """Build the full structural feature vector for a gene pair.

    Features:
      deg_x, deg_y   : degree centrality in STRING PPI network
      btw_x, btw_y   : betweenness centrality
      str_dist       : shortest path length (-1 if disconnected)
      str_score      : average STRING confidence along shortest path
      reac_jac       : Reactome pathway Jaccard coefficient
      go_bp, go_cc,
      go_mf          : GO Resnik semantic similarities
    """
    features: Dict[str, float] = {
        "deg_x": 0.0, "deg_y": 0.0,
        "btw_x": 0.0, "btw_y": 0.0,
        "str_dist": -1.0, "str_score": 0.0,
        "reac_jac": 0.0,
        "go_bp": 0.0, "go_cc": 0.0, "go_mf": 0.0,
    }

    # --- STRING topology ---
    try:
        G = _build_graph([gene_x, gene_y], species)

        if gene_x in G:
            features["deg_x"] = float(G.degree(gene_x))
        if gene_y in G:
            features["deg_y"] = float(G.degree(gene_y))

        # Betweenness (approximate on local subgraph)
        if G.number_of_nodes() >= 2:
            try:
                btw = nx.betweenness_centrality(G, normalized=True)
                features["btw_x"] = float(btw.get(gene_x, 0.0))
                features["btw_y"] = float(btw.get(gene_y, 0.0))
            except Exception:
                pass

        # Shortest path
        try:
            path = nx.dijkstra_path(G, gene_x, gene_y, weight="weight")
            length = len(path) - 1
            features["str_dist"] = float(length)
            if length > 0:
                scores = [
                    G[path[i]][path[i + 1]].get("score", 0)
                    for i in range(length)
                ]
                features["str_score"] = round(sum(scores) / length, 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            features["str_dist"] = -1.0
    except Exception:
        pass

    # --- Reactome Jaccard ---
    try:
        px = _pathways_for_gene(gene_x, species="MMU")
        py = _pathways_for_gene(gene_y, species="MMU")
        intersection = px & py
        union = px | py
        features["reac_jac"] = round(
            len(intersection) / len(union), 4
        ) if union else 0.0
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

    return features
