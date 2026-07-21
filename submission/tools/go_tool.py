"""Gene Ontology semantic similarity tool.

Computes Resnik semantic similarity across three sub-ontologies:
Biological Process (BP), Cellular Component (CC), Molecular Function (MF).

Resnik similarity = IC(MICA) where MICA is the Most Informative Common Ancestor.
Information Content IC(c) = -log(P(c)) where P(c) is annotation frequency.

Returns a compressed numerical matrix instead of verbose GO term descriptions.
"""

import gzip
import os
import pickle
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import requests

from tools.payload import compress, error_payload

# Cache paths
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "go_cache")
_GO_OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
_GOA_URLS = {
    "human": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz",
    "mouse": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/MOUSE/goa_mouse.gaf.gz",
}
_DEFAULT_SPECIES = "mouse"  # Track B uses mouse genes

_GO_GRAPH: Optional[Dict[str, dict]] = None
_GO_ANNOTATIONS: Optional[Dict[str, Set[str]]] = None
_GO_IC: Optional[Dict[str, float]] = None
_GO_NS: Dict[str, str] = {}  # term_id -> namespace


def _ensure_go_loaded(species: str = ""):
    global _GO_GRAPH, _GO_ANNOTATIONS, _GO_IC, _GO_NS

    sp = species or _DEFAULT_SPECIES

    if _GO_GRAPH is not None:
        return

    os.makedirs(_CACHE_DIR, exist_ok=True)
    obo_path = os.path.join(_CACHE_DIR, "go-basic.obo")
    goa_url = _GOA_URLS.get(sp, _GOA_URLS[_DEFAULT_SPECIES])
    goa_fname = f"goa_{sp}.gaf.gz"
    goa_path = os.path.join(_CACHE_DIR, goa_fname)
    cache_path = os.path.join(_CACHE_DIR, f"go_resnik_{sp}.pkl")

    # Try loading from cache
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
            _GO_GRAPH = cached["graph"]
            _GO_ANNOTATIONS = cached["annotations"]
            _GO_IC = cached["ic"]
            _GO_NS = cached["ns"]
            return

    _parse_obo(obo_path)
    _parse_annotations(goa_path, goa_url)
    _compute_ic()

    with open(cache_path, "wb") as f:
        pickle.dump({
            "graph": _GO_GRAPH,
            "annotations": _GO_ANNOTATIONS,
            "ic": _GO_IC,
            "ns": _GO_NS,
        }, f)


def _parse_obo(obo_path: str):
    global _GO_GRAPH, _GO_NS
    _GO_GRAPH = {}
    _GO_NS = {}

    if not os.path.exists(obo_path):
        _download(_GO_OBO_URL, obo_path)

    current: Optional[dict] = None
    with open(obo_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "[Term]":
                if current and current.get("id"):
                    _GO_GRAPH[current["id"]] = current
                current = {"is_a": [], "alt_ids": []}
            elif line == "[Typedef]":
                if current and current.get("id"):
                    _GO_GRAPH[current["id"]] = current
                current = None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("namespace: "):
                    ns = line[11:]
                    current["namespace"] = ns
                    _GO_NS[current.get("id", "")] = ns
                elif line.startswith("is_a: "):
                    parts = line[6:].split("!")
                    go_id = parts[0].strip()
                    current["is_a"].append(go_id)
                elif line.startswith("alt_id: "):
                    current["alt_ids"].append(line[8:])
                elif line.startswith("is_obsolete:"):
                    current["obsolete"] = True
    if current and current.get("id"):
        _GO_GRAPH[current["id"]] = current


def _parse_annotations(goa_path: str, goa_url: str = ""):
    global _GO_ANNOTATIONS
    _GO_ANNOTATIONS = defaultdict(set)

    if not os.path.exists(goa_path):
        url = goa_url or _GOA_URLS.get(_DEFAULT_SPECIES, "")
        if url:
            _download(url, goa_path)

    with gzip.open(goa_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            gene_symbol = cols[2].strip()
            go_id = cols[4].strip()
            qualifier = cols[3].strip() if len(cols) > 3 else ""
            if "NOT" in qualifier:
                continue
            _GO_ANNOTATIONS[gene_symbol].add(go_id)


def _compute_ic():
    global _GO_IC

    # Count annotations per GO term (propagated to ancestors)
    term_counts: Dict[str, int] = defaultdict(int)
    total_genes = len(_GO_ANNOTATIONS)

    for gene, terms in _GO_ANNOTATIONS.items():
        propagated: Set[str] = set()
        for t in terms:
            _propagate_ancestors(t, propagated)
        for t in propagated:
            term_counts[t] += 1

    _GO_IC = {}
    for term_id, count in term_counts.items():
        freq = count / max(total_genes, 1)
        if freq > 0:
            _GO_IC[term_id] = -math.log(freq)


def _propagate_ancestors(term_id: str, collected: Set[str]):
    if term_id in collected:
        return
    collected.add(term_id)
    graph = _GO_GRAPH or {}
    node = graph.get(term_id, {})
    for parent_id in node.get("is_a", []):
        _propagate_ancestors(parent_id, collected)


def _ancestors(term_id: str) -> Set[str]:
    result: Set[str] = set()
    _propagate_ancestors(term_id, result)
    return result


def _mica_ic(terms_a: Set[str], terms_b: Set[str]) -> float:
    """Resnik similarity: max IC among common ancestors."""
    ancestors_a: Set[str] = set()
    ancestors_b: Set[str] = set()
    for t in terms_a:
        _propagate_ancestors(t, ancestors_a)
    for t in terms_b:
        _propagate_ancestors(t, ancestors_b)

    common = ancestors_a & ancestors_b
    if not common:
        return 0.0

    ic_map = _GO_IC or {}
    return max((ic_map.get(c, 0.0) for c in common), default=0.0)


def _filter_by_namespace(terms: Set[str], namespace: str) -> Set[str]:
    ns_map = _GO_NS or {}
    return {t for t in terms if ns_map.get(t) == namespace}


def query_go_semantics(gene_x: str, gene_y: str, species: str = "") -> str:
    """Compute Resnik semantic similarity across BP, CC, MF sub-ontologies.

    Payload keys (compressed):
      bp : Resnik IC for Biological Process
      cc : Resnik IC for Cellular Component
      mf : Resnik IC for Molecular Function
    """
    try:
        _ensure_go_loaded(species)
    except Exception:
        return error_payload("go_load_failed")

    annotations = _GO_ANNOTATIONS or {}
    terms_x = annotations.get(gene_x, set())
    terms_y = annotations.get(gene_y, set())

    result = {}
    ns_map = {
        "bp": "biological_process",
        "cc": "cellular_component",
        "mf": "molecular_function",
    }

    for key, ns in ns_map.items():
        tx = _filter_by_namespace(terms_x, ns)
        ty = _filter_by_namespace(terms_y, ns)
        if tx and ty:
            result[key] = round(_mica_ic(tx, ty), 2)
        else:
            result[key] = 0.0

    return compress(result)


def _download(url: str, dest: str):
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
