# MLGenX Bioreasoning Challenge — Track B

Multi-agentic graph retrieval pipeline for predicting CRISPRi perturbation effects (up/down/no-change) on transcriptomic expression.

**Competition**: [MLGenX Bioreasoning Challenge - Track B](https://www.kaggle.com/competitions/ml-gen-x-bioreasoning-challenge-track-b)  
**Best Public Score**: 0.549 (v7 — rich gene annotations)

## Architecture

```
Perturbation Gene X  ──►  Multi-Agent Pipeline  ──►  P(up), P(down), P(none)
       │                        │
Target Gene Y       ┌───────────┴───────────┐
                    │   Planner Agent        │  ← Selects tool sequence
                    │   Execution Agent      │  ← Budget tracking (max 250 calls)
                    │   Context Synthesizer  │  ← Conflict resolution
                    └───────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        query_string_db  query_reactome  query_go_semantics
        (STRING PPI)     (pathway Jaccard) (GO Resnik IC)
                              │
                              ▼
                    query_ml_surrogate
                    (XGBoost classifier)
```

## Project Structure

```
├── config/
│   ├── determinism.py       # Global seed 42, CUDA determinism
│   └── settings.py          # Constants (MAX_TOOL_CALLS=250, etc.)
├── tools/
│   ├── string_tool.py        # STRING PPI + Dijkstra shortest path (mouse 10090)
│   ├── reactome_tool.py      # Reactome pathway Jaccard similarity (MMU)
│   ├── go_tool.py            # GO Resnik semantic similarity (BP/CC/MF, mouse)
│   ├── feature_builder.py    # 26-feature vector builder (KG + annotations + priors)
│   ├── surrogate_tool.py     # XGBoost classifier
│   └── payload.py            # Token-optimized JSON compression
├── agents/
│   ├── planner.py            # Executive controller (3-step logic ceiling)
│   ├── execution_agent.py    # Safety layer + 250-call budget
│   ├── synthesizer.py        # Narrative aggregation + conflict resolution matrix
│   └── prompts.py            # Master system prompt
├── data/
│   ├── ingestion.py          # Kaggle API + PerturbQA download
│   ├── competition/          # train.csv (7,705 rows), test.csv (1,813 rows)
│   └── gene_cache/           # mygene.info batch-fetched annotations (2,623 genes)
├── models/
│   └── xgboost_surrogate.pkl # Trained model
├── submission/
│   ├── submission.csv        # 1,813 predictions (9 columns)
│   ├── metadata.json         # Pipeline metadata
│   ├── prompt.txt            # LLM system prompt for Track B
│   └── tools/                # Tool definitions for Kaggle
├── pipeline.py               # Master orchestrator
└── README.md
```

## Setup

```bash
# Clone
git clone https://github.com/chrisyuspithk-bot/MLGenX-B.git
cd MLGenX-B

# Install dependencies
pip install xgboost networkx pandas numpy scikit-learn requests tqdm

# Set Kaggle credentials
export KAGGLE_API_TOKEN="your_token_here"
```

## Run Pipeline

```bash
# Full pipeline (feature building + training + CV + submission)
python pipeline.py

# Quick test on sample
python -c "
from data.ingestion import download_competition_data
from tools.feature_builder import build_features
train_df, test_df = download_competition_data()
feats = build_features('Trp53', 'Cdkn1a', species=10090, train_df=train_df)
print(feats)
"
```

## Feature Engineering

### v1 — Knowledge Graph Topology (10 features, score: 0.532)
STRING PPI degree/betweenness centrality, shortest path, Reactome Jaccard, GO Resnik similarity.

### v5 — Gene Annotations (22 features, score: 0.544)
GO BP/CC/MF term counts, KEGG pathway counts, Jaccard overlaps, count products. Data from mygene.info batch query (POST endpoint, 2,623 mouse genes).

### v7 — Rich Annotations + Text Similarity (30 features, score: 0.549)
Added gene name Jaccard, summary text Jaccard, GO/pathway ratios, boolean annotation flags.

### Feature Importance (v7)
| Feature | Importance |
|---------|-----------|
| has_bp_y (target has GO BP) | 0.058 |
| cc_y (target CC terms) | 0.040 |
| mf_y (target MF terms) | 0.040 |
| kw_jac (pathway overlap) | 0.040 |
| bp_jac (GO BP overlap) | 0.033 |

## Cross-Validation

5-fold StratifiedGroupKFold (grouped by perturbation gene, zero-overlap):

| Version | DE AUROC | DIR AUROC | Combined |
|---------|----------|-----------|----------|
| v1 (KG) | 0.524 ± 0.012 | 0.624 ± 0.012 | 0.574 |
| v5 (annotations) | 0.544 ± 0.015 | 0.772 ± 0.021 | 0.658 |
| v7 (rich) | 0.548 ± 0.013 | 0.805 ± 0.029 | 0.676 |

## Submission Format (Track B)

Required columns in `submission.csv`:
- `id`, `prediction_up`, `prediction_down` — predictions
- `reasoning_trace` — JSON tool-call trace
- `tokens_used`, `num_tool_calls`, `prompt_tokens`, `num_distinct_tools`, `model_name` — metadata

Constraints: ≤250 tool calls/row, ≤16,384 prompt tokens, ≤100 distinct tools.

## Key Design Decisions

1. **Zero-overlap generalization**: All features are identifier-agnostic — no gene name matching between train and test
2. **Token optimization**: JSON keys compressed (`ic` not `interaction_confidence_score`), floats truncated to 2dp
3. **Conflict resolution matrix**: Spatial mismatch (CC<0.1) → no-change; direct binding (dist=1, score≥0.7) → trust STRING; long-range (dist>3) → trust surrogate
4. **Budget safety**: ExecutionAgent halts at 240/250 calls to prevent disqualification
5. **Continuous probabilities**: XGBoost `predict_proba` for AUROC-optimal soft predictions

## Scoring

```
score = (micro_AUROC_DE + micro_AUROC_DIR) / 2
```

- **DE AUROC**: (up or down) vs none, scored via `prediction_up + prediction_down`
- **DIR AUROC**: up vs down among DE-positive rows, scored via `prediction_up / (prediction_up + prediction_down)`

## Limitations

The local XGBoost models plateau at ~0.55 because static gene annotations cannot capture dynamic regulatory relationships. The LLM (GPT-OSS-120B), with its knowledge of the entire biomedical literature, is essential for competitive scores (>0.9). A DSPy-based LLM pipeline with the prepared system prompt in `submission/prompt.txt` is the intended approach for Track B.

## License

Research use only — competition submission.
