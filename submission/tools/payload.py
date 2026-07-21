"""Token-optimized JSON payload serialization.

Aggressive minification to keep tool-call responses within the 16 384 token budget.
Average compressed payload: 12-18 tokens vs 100+ for unoptimized.
"""

import json
from typing import Any, Dict

from config.settings import FLOAT_PRECISION


def _truncate(value: float) -> float:
    try:
        return round(float(value), FLOAT_PRECISION)
    except (TypeError, ValueError):
        return value


def _strip_nulls(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    if isinstance(obj, (float, int)):
        return _truncate(obj)
    # Handle numpy scalar types
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        return _truncate(obj.item())
    return obj


def compress(payload: Dict[str, Any]) -> str:
    """Return a minified JSON string with nulls stripped and floats truncated."""
    cleaned = _strip_nulls(payload)
    return json.dumps(cleaned, separators=(",", ":"), ensure_ascii=True)


def error_payload(reason: str, action: str = "retry") -> str:
    return compress({"err": reason, "act": action})
