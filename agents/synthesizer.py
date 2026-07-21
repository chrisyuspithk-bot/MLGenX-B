"""Context Synthesizer – aggregates tool-call results into structured narrative.

Translates raw compressed JSON from tool calls into cohesive biological prose
that the LLM can reason over effectively (inspired by SUMMER architecture).

Also implements the hard-coded Conflict Resolution Matrix to resolve
contradictions between STRING, surrogate, and ontology evidence.
"""

import json
from typing import Any, Dict, List


def _parse_result(result_str: str) -> Dict[str, Any]:
    try:
        return json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def synthesize(results: List[Dict[str, Any]], gene_x: str, gene_y: str) -> str:
    """Build a concise biological narrative from tool-call results.

    Returns a structured text block suitable for injection into the final
    evaluation prompt.
    """
    if not results:
        return f"No tool data retrieved for {gene_x} -> {gene_y}."

    string_data = {}
    reactome_data = {}
    go_data = {}
    surrogate_data = {}

    for entry in results:
        tool = entry.get("tool", "")
        parsed = _parse_result(entry.get("result", "{}"))
        if "err" in parsed:
            continue
        if tool == "query_string_db":
            string_data = parsed
        elif tool == "query_reactome":
            reactome_data = parsed
        elif tool == "query_go_semantics":
            go_data = parsed
        elif tool == "query_ml_surrogate":
            surrogate_data = parsed

    lines = [f"Perturbation: {gene_x} -> Target: {gene_y}"]

    # STRING section
    if string_data:
        ic = string_data.get("ic")
        dist = string_data.get("dist", -1)
        score = string_data.get("score", 0)
        if ic is not None:
            lines.append(f"STRING direct interaction confidence: {ic:.2f}.")
        if dist > 0:
            lines.append(
                f"Shortest regulatory path: {dist} hops (avg confidence {score:.2f})."
            )
        elif dist == 0:
            lines.append("STRING: Gene X and Y are the same node.")
        elif dist == -1:
            lines.append("STRING: No connected path found in the PPI network.")

    # Reactome section
    if reactome_data:
        jac = reactome_data.get("jac", 0)
        shared = reactome_data.get("shared", 0)
        lines.append(
            f"Reactome Jaccard similarity: {jac:.2f} ({shared} shared pathways)."
        )

    # GO section
    if go_data:
        bp = go_data.get("bp", 0)
        cc = go_data.get("cc", 0)
        mf = go_data.get("mf", 0)
        lines.append(
            f"GO Resnik similarity — BP: {bp:.2f}, CC: {cc:.2f}, MF: {mf:.2f}."
        )
        if cc < 0.1:
            lines.append(
                "WARNING: Very low Cellular Component similarity. "
                "Proteins likely reside in different compartments — "
                "direct interaction improbable regardless of pathway overlap."
            )

    # Surrogate section
    if surrogate_data:
        up = surrogate_data.get("up", 0)
        dn = surrogate_data.get("dn", 0)
        nc = surrogate_data.get("nc", 0)
        lines.append(
            f"XGBoost Surrogate prediction: up={up:.2f}, down={dn:.2f}, "
            f"no-change={nc:.2f}."
        )

    # --- Conflict Resolution Matrix ---
    directive = _apply_conflict_resolution(string_data, surrogate_data, go_data)
    if directive:
        lines.append(f"CONFLICT DIRECTIVE: {directive}")

    return "\n".join(lines)


def _apply_conflict_resolution(
    string_data: dict,
    surrogate_data: dict,
    go_data: dict,
) -> str:
    """Hard-coded conflict resolution per the documented matrix.

    Returns a directive string or empty string if no conflict.
    """
    dist = string_data.get("dist")
    ic = string_data.get("ic")
    cc = go_data.get("cc", 1.0)

    # Spatial mismatch override
    if cc < 0.1:
        return (
            "Spatial mismatch detected (CC sim < 0.1). "
            "Override all other signals: classify as no-change. "
            "Proteins in isolated compartments cannot interact post-perturbation."
        )

    # Direct binding clash: STRING dist=1 overrides surrogate
    if dist is not None and dist == 1 and ic is not None and ic >= 0.7:
        return (
            f"Direct binding detected (STRING distance=1, confidence={ic}). "
            "Override surrogate prediction. Trust STRING direct physical interaction."
        )

    # Long-range cascade: surrogate overrides STRING for dist > 3
    if dist is not None and dist > 3:
        return (
            f"Long-range cascade (STRING distance={dist}). "
            "Override STRING. Trust XGBoost surrogate for systemic propagation."
        )

    return ""
