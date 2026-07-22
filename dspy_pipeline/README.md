# DSPy LLM Pipeline — Track B

DSPy-based multi-agent pipeline that uses GPT-OSS-120B (or any OpenAI-compatible LLM) to predict CRISPRi perturbation effects through structured biological reasoning.

## Architecture

```
gene_X, gene_Y
     │
     ├── Characterize (gene_info tool)
     │   ├── gene_role (TF/kinase/receptor/...)
     │   ├── functional_summary
     │   ├── is_regulator
     │   └── pathway_membership
     │
     ├── RetrieveEvidence (ReAct + 7 tools)
     │   ├── gene_info(X), gene_info(Y)
     │   ├── protein_interactions
     │   ├── pathway_overlap
     │   ├── go_similarity
     │   ├── ml_surrogate (XGBoost ensemble)
     │   ├── lookup_similar_perturbations
     │   └── lookup_training
     │
     ├── Synthesize (ChainOfThought + conflict matrix)
     │   ├── biological_reasoning
     │   ├── conflict_resolution
     │   ├── predicted_direction
     │   └── confidence
     │
     └── Calibrate (rule-based + surrogate blend)
         ├── prediction_up ∈ [0, 1]
         ├── prediction_down ∈ [0, 1]
         └── reasoning_trace (JSON)
```

## Tools

| Tool | Description | Source |
|------|-------------|--------|
| `gene_info` | GO terms, KEGG pathways, gene type, summary | mygene.info |
| `protein_interactions` | STRING PPI partners + confidence scores | STRING DB |
| `pathway_overlap` | Reactome Jaccard similarity | Reactome |
| `go_similarity` | GO Resnik semantic similarity (BP/CC/MF) | GO |
| `ml_surrogate` | XGBoost 10-seed ensemble probabilities | Local model |
| `lookup_similar_perturbations` | KNN in gene-property space → effect profiles | Training data |
| `lookup_training` | Exact-match training data lookup | Training data |

## Conflict Resolution Matrix

| Scenario | Primary Evidence | Fallback | Directive |
|----------|-----------------|----------|-----------|
| Direct binding (STRNG dist=1, score>0.7) | STRING | Surrogate | Trust STRING |
| Long-range cascade (dist>3) | Surrogate | STRING | Trust surrogate |
| Spatial mismatch (GO CC<0.1) | — | — | Classify no-change |
| Shared pathways + TF-target | Pathway | Surrogate | Infer direction |
| Unrelated + no interactions | — | — | Classify no-change |

## Usage

```bash
pip install dspy

python -m dspy_pipeline.submit \
    --api-base $LLM_ENDPOINT \
    --api-key $LLM_KEY \
    --model openai/gpt-oss-120b \
    --optimize
```

### Without LLM access (testing)

```bash
# Uses a mock/fake LM for pipeline structure testing
python -m dspy_pipeline.submit \
    --api-base http://localhost:8000 \
    --api-key test \
    --model test/model \
    --fast
```

## Optimization

DSPy BootstrapFewShot optimizes the prompt instructions and few-shot examples
against the training set using the competition metric `(DE_AUROC + DIR_AUROC)/2`.

## Calibration

Three strategies (in `calibrate.py`):
1. **Rule-based** (default): maps direction/confidence to fixed probability ranges
2. **Temperature scaling**: learns optimal temperature from validation log-probs
3. **XGBoost residual**: blends LLM probabilities with surrogate predictions
   (80% LLM for high confidence, 50/50 for low confidence)
