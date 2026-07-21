"""Execution Agent – deterministic Python safety layer.

Responsibilities:
  1. Receives JSON-formatted tool requests from the Planner
  2. Invokes external APIs / local surrogate, catching all exceptions
  3. Maintains atomic budget counter (max 250 tool calls per query)
  4. Returns condensed error payloads instead of verbose tracebacks
  5. Triggers emergency halt at 240 calls, forcing synthesis phase
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional

from config.settings import EMERGENCY_CALL_CEILING, MAX_TOOL_CALLS
from tools.payload import error_payload

# Registry of available tool functions
TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(name: str, fn: Callable) -> None:
    TOOL_REGISTRY[name] = fn


class ExecutionAgent:
    """Deterministic safety layer wrapping tool invocations."""

    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        self.max_calls = max_calls
        self.call_count = 0
        self.results: List[Dict[str, Any]] = []
        self.halted = False

    def execute(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Invoke a registered tool, returning compressed JSON result.

        Returns error payload on timeout, missing tool, or any exception.
        """
        if self.halted:
            return error_payload("halted")

        self.call_count += 1

        if self.call_count >= EMERGENCY_CALL_CEILING:
            self.halted = True
            return error_payload("budget_ceiling", "force_synthesize")

        if self.call_count > self.max_calls:
            return error_payload("budget_exceeded")

        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            return error_payload(f"unknown_tool:{tool_name}")

        try:
            result = fn(**params)
            self.results.append({
                "tool": tool_name,
                "params": params,
                "result": result,
                "call_idx": self.call_count,
            })
            return result
        except Exception as exc:
            exc_type = type(exc).__name__
            return error_payload(exc_type.lower())

    def get_budget_remaining(self) -> int:
        return self.max_calls - self.call_count

    def get_all_results(self) -> List[Dict[str, Any]]:
        return list(self.results)


# --- Register known tools ---
from tools.string_tool import query_string_db
from tools.reactome_tool import query_reactome
from tools.go_tool import query_go_semantics
from tools.surrogate_tool import query_ml_surrogate

register_tool("query_string_db", query_string_db)
register_tool("query_reactome", query_reactome)
register_tool("query_go_semantics", query_go_semantics)
register_tool("query_ml_surrogate", query_ml_surrogate)
