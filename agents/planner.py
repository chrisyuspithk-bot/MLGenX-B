"""Planner Agent – executive controller of the multi-agent state machine.

Responsibilities:
  1. Formulates hypotheses for perturbation X -> target Y
  2. Selects optimal tool-call sequence (max 3 logic steps before fetching)
  3. Evaluates biological plausibility before tool dispatch
  4. Prioritizes STRING for TFs, GO spatial for structural proteins
"""

from typing import Any, Dict, List

from config.settings import MAX_CONSECUTIVE_LOGIC_STEPS


# Canonical transcription factor list (subset for planning heuristics)
_TRANSCRIPTION_FACTORS: set = set()

# Structural / non-regulatory gene categories (heuristic)
_STRUCTURAL_GENE_PATTERNS = [
    "COL", "KRT", "ACT", "TUB", "MYH", "MYL", "LMN", "VIM",
    "FN1", "CLDN", "OCLN", "DSP", "PKP", "DSG", "DSC",
]


def _is_transcription_factor(gene: str) -> bool:
    """Heuristic check: is this gene likely a TF?

    In production this would query a TF database. Here we use a pattern
    heuristic that can be overridden by TF-specific knowledge.
    """
    return gene.upper() in _TRANSCRIPTION_FACTORS


def _is_structural(gene: str) -> bool:
    """Heuristic check: is this gene likely a structural/non-regulatory protein?"""
    upper = gene.upper()
    return any(upper.startswith(p) for p in _STRUCTURAL_GENE_PATTERNS)


class PlannerAgent:
    """Plans tool-call sequences for a single perturbation pair.

    The planner determines the order and selection of tool calls based on
    biological heuristics, respecting the 3-step logic ceiling.
    """

    def __init__(self, gene_x: str, gene_y: str):
        self.gene_x = gene_x
        self.gene_y = gene_y
        self.step_count = 0
        self.tool_sequence: List[Dict[str, Any]] = []
        self.completed = False

    def plan_next(self) -> List[Dict[str, Any]]:
        """Return the next batch of tool calls (up to MAX_CONSECUTIVE_LOGIC_STEPS).

        Returns empty list when planning is complete.
        """
        if self.completed:
            return []

        batch: List[Dict[str, Any]] = []

        if self.step_count == 0:
            # Phase 1: Core topology (always run STRING + Reactome)
            batch = [
                {
                    "tool": "query_string_db",
                    "params": {"gene_x": self.gene_x, "gene_y": self.gene_y},
                },
                {
                    "tool": "query_reactome",
                    "params": {"gene_x": self.gene_x, "gene_y": self.gene_y},
                },
            ]
        elif self.step_count == 1:
            # Phase 2: Semantic depth (always run GO)
            batch = [
                {
                    "tool": "query_go_semantics",
                    "params": {"gene_x": self.gene_x, "gene_y": self.gene_y},
                },
            ]
        elif self.step_count == 2:
            # Phase 3: Surrogate (always run for final verdict)
            # Features will be filled by the caller from previous results
            batch = [
                {
                    "tool": "query_ml_surrogate",
                    "params": {"features": {}},  # caller fills
                },
            ]
        else:
            self.completed = True
            return []

        self.step_count += 1
        self.tool_sequence.extend(batch)
        return batch

    def is_complete(self) -> bool:
        return self.completed

    def get_plan_summary(self) -> str:
        tools = [t["tool"] for t in self.tool_sequence]
        return f"Plan for {self.gene_x}->{self.gene_y}: " + " -> ".join(tools)
