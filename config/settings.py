"""Central configuration constants for the multi-agent pipeline."""

# Competition constraints
MAX_TOOLS = 100
MAX_TOOL_CALLS = 250
PROMPT_TOKEN_LIMIT = 16384

# STRING DB thresholds (scores are floats in [0, 1])
STRING_CONFIDENCE_THRESHOLD = 0.7
STRING_API_BASE = "https://string-db.org/api"

# Gene Ontology namespaces
GO_NAMESPACES = ("biological_process", "cellular_component", "molecular_function")

# Agent planning limits
MAX_CONSECUTIVE_LOGIC_STEPS = 3
EMERGENCY_CALL_CEILING = 240

# XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
}

# GroupKFold splits
N_FOLDS = 5

# Logit extraction tokens (vocabulary ids depend on model tokenizer)
TARGET_TOKENS = ["up", "down", "none"]

# Payload compression: float precision
FLOAT_PRECISION = 2
