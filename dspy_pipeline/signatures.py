"""DSPy signatures for the multi-agent perturbation prediction pipeline.

Each signature defines the input/output contract for a reasoning module.
"""

import dspy


class Characterize(dspy.Signature):
    """Characterize the molecular function, biological role, and regulatory
    potential of a single gene in mouse bone marrow-derived macrophages.

    Determine: Is it a transcription factor? A kinase? A signaling adaptor?
    A structural protein? What pathways does it participate in? What is its
    known role in macrophage biology or immune function?

    Use gene_info() and protein_interactions() to gather evidence.
    """

    gene_symbol: str = dspy.InputField(desc="Gene symbol (e.g., Trp53)")
    gene_role: str = dspy.OutputField(
        desc="Label: TF, kinase, signaling, receptor, structural, enzyme, or other"
    )
    functional_summary: str = dspy.OutputField(
        desc="1-2 sentence summary of biological function and macrophage relevance"
    )
    is_regulator: bool = dspy.OutputField(
        desc="True if this gene is likely to regulate other genes (TF, kinase, signaling)"
    )
    pathway_membership: str = dspy.OutputField(
        desc="Comma-separated list of key pathways (e.g., 'NF-kB, apoptosis, MAPK')"
    )
    interactor_count: int = dspy.OutputField(
        desc="Approximate number of protein interaction partners"
    )


class RetrieveEvidence(dspy.Signature):
    """Retrieve all relevant evidence for predicting how CRISPRi knockdown
    of gene_X will affect expression of gene_Y.

    Use ALL available tools: gene_info for both genes, protein_interactions,
    pathway_overlap, go_similarity, ml_surrogate, lookup_similar_perturbations.

    Gather data methodically. Do not predict yet — just collect evidence.
    """

    gene_X: str = dspy.InputField(desc="Perturbation gene (CRISPRi knockdown)")
    gene_Y: str = dspy.InputField(desc="Target gene (expression measured)")

    X_info: str = dspy.OutputField(
        desc="JSON: gene_info(gene_X) result with type, pathways, GO terms"
    )
    Y_info: str = dspy.OutputField(
        desc="JSON: gene_info(gene_Y) result with type, pathways, GO terms"
    )
    string_evidence: str = dspy.OutputField(
        desc="JSON: STRING interaction data for both genes, including any direct edge"
    )
    pathway_overlap: str = dspy.OutputField(
        desc="JSON: Reactome pathway Jaccard similarity and shared pathways"
    )
    go_similarity: str = dspy.OutputField(
        desc="JSON: GO Resnik semantic similarity for BP, CC, MF"
    )
    surrogate_prediction: str = dspy.OutputField(
        desc="JSON: XGBoost surrogate predicted up/down probabilities"
    )
    similar_perturbations: str = dspy.OutputField(
        desc="JSON: Training-set genes similar to X and their effect profiles"
    )


class Synthesize(dspy.Signature):
    """Synthesize all retrieved evidence into a biological reasoning chain
    and apply the conflict resolution matrix to determine the likely outcome.

    CONFLICT MATRIX:
    1. Direct binding (STRING distance=1, score>0.7) → trust STRING over surrogate
    2. Long-range cascade (STRING distance>3) → trust surrogate over STRING
    3. Spatial mismatch (GO CC similarity <0.1) → classify as no-change regardless
    4. Shared pathways + known TF-target → infer regulatory direction
    5. Unrelated functions + no interactions → likely no-change

    Consider cell-type context: bone marrow-derived macrophages.
    Consider that CRISPRi causes knockdown (loss of function) of gene X.
    """

    gene_X: str = dspy.InputField()
    gene_Y: str = dspy.InputField()
    X_info: str = dspy.InputField()
    Y_info: str = dspy.InputField()
    string_evidence: str = dspy.InputField()
    pathway_overlap: str = dspy.InputField()
    go_similarity: str = dspy.InputField()
    surrogate_prediction: str = dspy.InputField()
    similar_perturbations: str = dspy.InputField()

    biological_reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning: what X does, what Y does, how they relate, "
        "which evidence was weighted most, and why"
    )
    conflict_resolution: str = dspy.OutputField(
        desc="Which conflict rule was triggered and how the evidence was resolved"
    )
    predicted_direction: str = dspy.OutputField(
        desc="One of: up, down, no_change"
    )
    confidence: str = dspy.OutputField(
        desc="One of: high, medium, low"
    )


class Calibrate(dspy.Signature):
    """Convert qualitative prediction and confidence into well-calibrated
    continuous probabilities that maximize AUROC.

    Guidelines:
    - high confidence + up → pred_up=0.92-0.98, pred_down=0.01-0.03
    - high confidence + down → pred_up=0.01-0.03, pred_down=0.92-0.98
    - high confidence + no_change → pred_up=0.01-0.03, pred_down=0.01-0.03
    - medium confidence → intermediate values (0.6-0.8 for primary, 0.05-0.2 for secondary)
    - low confidence → near-uniform (0.3-0.4 each)
    - Always sum p_up + p_down <= 1.0 (p_none = 1 - p_up - p_down)
    """

    predicted_direction: str = dspy.InputField()
    confidence: str = dspy.InputField()
    biological_reasoning: str = dspy.InputField()
    surrogate_prediction: str = dspy.InputField()

    prediction_up: float = dspy.OutputField(desc="P(up) ∈ [0, 1]")
    prediction_down: float = dspy.OutputField(desc="P(down) ∈ [0, 1]")
    reasoning_trace: str = dspy.OutputField(
        desc="JSON summary of tools used, evidence, and confidence calibration"
    )
