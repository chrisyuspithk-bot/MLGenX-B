"""STRING DB tool: PPI confidence, shortest-path traversal via Dijkstra.

Filters for experimental + curated database channels with score >= 700.
Edge weight = 1000 - combined_score so high-confidence edges are "shorter".
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import requests

from config.settings import STRING_API_BASE

# STRING returns scores as floats in [0, 1]; threshold at 0.7 (high confidence)
STRING_THRESHOLD = 0.7
from tools.payload import compress, error_payload

_STRING_CACHE: Dict[str, dict] = {}
_GRAPH_CACHE: Optional[nx.Graph] = None
_GRAPH_SPECIES: Optional[int] = None


def _fetch_interactions(
    genes: List[str], species: int = 10090, min_score: int = 0
) -> dict:
    """Fetch STRING interaction network for a gene list (human=9606)."""
    key = f"{','.join(sorted(genes))}@{min_score}"
    if key in _STRING_CACHE:
        return _STRING_CACHE[key]

    url = f"{STRING_API_BASE}/json/network"
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": min_score,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        data = {}
    _STRING_CACHE[key] = data
    return data


def _build_graph(
    genes: List[str], species: int = 10090, min_score: int = 0
) -> nx.Graph:
    global _GRAPH_CACHE, _GRAPH_SPECIES
    cache_key = (species, min_score)
    if _GRAPH_CACHE is not None and _GRAPH_SPECIES == cache_key:
        return _GRAPH_CACHE

    G = nx.Graph()
    data = _fetch_interactions(genes, species, min_score)
    for edge in data:
        src = edge.get("preferredName_A") or edge.get("stringId_A", "")
        tgt = edge.get("preferredName_B") or edge.get("stringId_B", "")
        combined = float(edge.get("score", 0))
        weight = max(0.001, 1.0 - combined)
        G.add_edge(src, tgt, weight=weight, score=combined)
    _GRAPH_CACHE = G
    _GRAPH_SPECIES = cache_key
    return G


def query_string_db(gene_x: str, gene_y: str, species: int = 10090) -> str:
    """Return direct interaction confidence + shortest regulatory path.

    Payload keys (compressed):
      ic   : direct interaction confidence 0-1000 (absent if no direct edge)
      dist : shortest-path hop count
      path : abbreviated node list of shortest path
      score: aggregated confidence along path
    """
    G_full = _build_graph([gene_x, gene_y], species, min_score=0)

    # Filter to high-confidence edges only
    G = nx.Graph()
    for u, v, data in G_full.edges(data=True):
        if data.get("score", 0) >= STRING_THRESHOLD:
            G.add_edge(u, v, **data)

    result: dict = {}

    # Direct edge check
    if G.has_edge(gene_x, gene_y):
        result["ic"] = G[gene_x][gene_y]["score"]

    # Dijkstra shortest path
    try:
        path = nx.dijkstra_path(G, gene_x, gene_y, weight="weight")
        length = len(path) - 1
        result["dist"] = length
        if length <= 5:
            result["path"] = path
        agg_score = 0
        if length > 0:
            scores = [
                G[path[i]][path[i + 1]].get("score", 0) for i in range(length)
            ]
            agg_score = round(sum(scores) / length)
        result["score"] = agg_score
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        result["dist"] = -1

    return compress(result)
