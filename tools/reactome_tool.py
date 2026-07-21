"""Reactome pathway tool: Jaccard similarity of pathway membership sets.

Uses the Reactome Content Service REST API to fetch pathways per gene,
then computes Jaccard = |intersection| / |union|.
"""

from typing import Dict, List, Set

import requests

from tools.payload import compress, error_payload

_REACTOME_CACHE: Dict[str, Set[str]] = {}
_REACTOME_BASE = "https://reactome.org/ContentService"


def _pathways_for_gene(gene: str, species: str = "MMU") -> Set[str]:
    if gene in _REACTOME_CACHE:
        return _REACTOME_CACHE[gene]

    pathways: Set[str] = set()
    try:
        url = f"{_REACTOME_BASE}/data/query/ids"
        resp = requests.post(
            url,
            headers={"Content-Type": "text/plain"},
            data=gene,
            params={"species": species},
            timeout=15,
        )
        resp.raise_for_status()
        for mapping in resp.json():
            for entry in mapping.get("entries", []):
                if entry.get("type") == "Pathway":
                    pathways.add(entry["id"])
    except Exception:
        pass

    _REACTOME_CACHE[gene] = pathways
    return pathways


def query_reactome(gene_x: str, gene_y: str, species: str = "MMU") -> str:
    """Return Jaccard coefficient for shared Reactome pathway membership.

    Payload keys (compressed):
      jac : Jaccard similarity [0, 1]
      nx  : number of pathways for gene_x
      ny  : number of pathways for gene_y
      shared : count of shared pathways
    """
    px = _pathways_for_gene(gene_x, species)
    py = _pathways_for_gene(gene_y, species)

    intersection = px & py
    union = px | py

    jaccard = len(intersection) / len(union) if union else 0.0

    return compress({
        "jac": jaccard,
        "nx": len(px),
        "ny": len(py),
        "shared": len(intersection),
    })
